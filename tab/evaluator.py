from typing import Any, Dict, List, Sequence, Tuple, Union
import csv
from pathlib import Path
from warnings import warn

from tqdm import tqdm
import torch

from kaitorch.typing import TorchTensor, TorchFloat
from kaitorch.data import squared_euclidean_distance

from .tab import TAB
from .utils import KMMatch


SIZE = (64, 128)


class Evaluator:
    R = 0.1
    T = (0.3, 0.5, 0.8)

    SEMANTICS: Dict[str, Sequence[str]] = {}
    for c in TAB.SEMANTICS:
        SEMANTICS[c] = [c]
    SEMANTICS['w/o semantics'] = TAB.SEMANTICS

    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'test',
        *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - root: path to the TAB dataset.
            - split: "train", "val" or "test".

        ### Methods:
            - __call__
            - tabulate

        __call__
        ### Args:
            - x: a sequence of prediction results listed frame by frame.
                For each frame, it should be in the form as bellow.

                {
                    "sequence": "xxxx-xx-xx-xx-xx-xx-xx",
                    "id": "1234567890.123456",
                    "boundaries": [
                        {
                            "semantics": "xxx",
                            "points": [
                                [12.345, 12.345],
                                [12.345, 12.345],
                                ...
                            ],
                            "scores": [12.345, 12.345, ...]
                        }
                    ]
                }

        ### Returns:
            - Point-level F1 scores.
            - Line-level F1 scores.

        '''
        self._tab = TAB(root, split)
        self._match = KMMatch()

    @torch.no_grad()
    def __call__(
        self, x: Sequence[Dict[str, Any]]
    ) -> Tuple[
        Dict[str, Dict[float, float]],
        Dict[str, Dict[float, Dict[str, float]]]
    ]:
        _preds = {}
        for pred in x:
            id = pred['id']
            if id in _preds:
                warn(f'a duplicated prediction {pred["sequence"]}/{id}.')
            else:
                _preds[id] = pred['boundaries']

        scores_p = {}
        scores_b = {}
        for c in self.SEMANTICS.keys():
            scores_p[c] = []
            _scores_b = {}
            for t in self.T:
                _scores_b[t] = [0., 0., 0.]  # tp, num_pred, num_bound
            scores_b[c] = _scores_b

        for f in tqdm(self._tab):
            _id = f.id
            preds, bounds = self._preprocess_(
                _preds[_id] if _id in _preds else [],
                self._tab.get_boundaries(f)
            )

            for c, cats in self.SEMANTICS.items():
                _scores_p, _scores_b = self._eval_one(
                    [p for p in preds if p['semantics'] in cats],
                    [
                        bound
                        for bound in bounds
                        if bound['semantics'] in cats or bound['fuzzy']
                    ]
                )

                scores_p[c] += _scores_p
                for t in self.T:
                    for i in range(3):
                        scores_b[c][t][i] += _scores_b[t][i]

        for c in self.SEMANTICS.keys():
            scores_p[c] = torch.mean(torch.as_tensor(scores_p[c])).item()

            _scores_b = scores_b[c]
            for t in self.T:
                s = _scores_b[t]
                tp = s[0]
                if 0 == tp:
                    scores_b[c][t] = 0
                else:
                    pre = tp / s[1]
                    rec = tp / s[2]
                    scores_b[c][t] = 2 * pre * rec / (pre + rec)
        return scores_p, scores_b

    @torch.no_grad()
    def _eval_one(
        self,
        preds: Sequence[Dict[str, Any]],
        bounds: Sequence[Dict[str, Any]]
    ) -> Tuple[
        Dict[float, List[float]],
        Dict[float, Dict[float, Dict[str, int]]]
    ]:
        num_bound = len(bounds)
        num_pred = len(preds)

        scores_b = {}
        if 0 == num_bound or 0 == num_pred:
            for bound in bounds:
                num_bound -= int(bound['fuzzy'])
            for t in self.T:
                scores_b[t] = (0., num_pred, num_bound)
            return [0.] * num_bound, scores_b

        mask_non_fuzzy = []
        sims = torch.zeros((num_bound, num_pred))
        for i, bound in enumerate(bounds):
            mask_non_fuzzy.append(not bound['fuzzy'])
            keypoints = bound['keypoints'].unsqueeze(1)
            radii = bound['radii']
            num_key = len(keypoints)

            for j, pred in enumerate(preds):
                points = pred['points']
                distances = squared_euclidean_distance(
                    keypoints, points.unsqueeze(0)
                )  # (num_key, num_point)

                pre = torch.sum(
                    torch.any(distances < radii.unsqueeze(1), dim=0)
                ) / len(points)
                rec = torch.sum(
                    torch.min(distances, dim=-1)[0] < radii
                ) / num_key
                if 0 == (t := pre + rec):
                    sims[i, j] = 0
                else:
                    sims[i, j] = 2 * pre * rec / t

        mask_non_fuzzy = torch.as_tensor(mask_non_fuzzy)

        # Do one-to-one matching.
        indices = torch.as_tensor(self._match(sims)[0])
        mask = indices != -1
        sims = sims[torch.arange(num_bound)[mask], indices[mask]]
        num_bound = torch.sum(mask_non_fuzzy).item()
        mask_non_fuzzy = mask_non_fuzzy[mask]

        for t in self.T:
            m = sims >= t
            tp = torch.sum(m[mask_non_fuzzy]).item()
            scores_b[t] = (tp, num_pred - torch.sum(m).item() + tp, num_bound)

        sims = sims[mask_non_fuzzy]
        return sims.tolist() + [0.] * (num_bound - len(sims)), scores_b

    @torch.no_grad()
    def _preprocess_(
        self, preds: Sequence[Dict[str, Any]], bounds: Sequence[Dict[str, Any]]
    ) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
        _preds = []
        for p in preds:
            points = p['points']
            if 0 == len(points):
                continue
            p['points'] = torch.as_tensor(points)
            _preds.append(p)

        for bound in bounds:
            points_xy, radii = self.decode_keypoints(bound)

            bound['keypoints'] = points_xy
            bound['radii'] = torch.pow(radii, 2)
        return _preds, bounds

    @classmethod
    def decode_keypoints(
        cls, boundary: Dict[str, Any]
    ) -> Tuple[TorchTensor[TorchFloat], TorchTensor[TorchFloat]]:
        r'''Get 2D coordinates of keypoints and their tolerant radii.

        ### Args:
            - boundary

        ### Returns:
            - 2D coordinates of keypoints. Its shape is `(N, 2)`.
            - Tolerant radii of keypoints. Its shape is `(N,)`.

        '''
        points_xy = []
        ns = []
        ends = []
        single = int(boundary['single'])
        for p in boundary['keypoints']:
            points_xy.append(p['xy'])

            ns.append(
                int(p['curve'])
                + int(p['unstructured'])
                + int(
                    p['irregular'] or p['occluded']
                    or p['blind'] or p['distorted'] or p['lengthened']
                )
                + single
            )
            ends.append(int(p['end']))

        points_xy = torch.as_tensor(points_xy)
        return points_xy, cls.R * (
            torch.as_tensor(ends) + 1 + (
                torch.sqrt(torch.sum(torch.pow(points_xy, 2), dim=1)) / 40 + 1
            ) * torch.log10(
                # torch.as_tensor(ns, dtype=points_xy.dtype) * 2.25 + 1
                torch.as_tensor(ns, dtype=points_xy.dtype) * 9 + 1
            )
        )

    @classmethod
    def tabulate(
        cls, x: Tuple[Dict[str, float], Dict[str, Dict[float, float]]]
    ) -> str:
        r'''

        ### Args:
            - scores_point: point-level F1 scores.
            - scores_bound: line-level F1 scores.

        ### Returns:
            - Tabulated results.

        '''
        scores_point, scores_bound = x

        t_0 = ''
        t_1 = ''
        num_t = len(cls.T) + 1
        for c in cls.SEMANTICS.keys():
            t_0 += f'{c}' + '\t' * num_t + '| '
            t_1 += 'F_P | '
            for t in cls.T:
                t_1 += f'F_{t} | '
        t_0 += '\n'
        t_1 += '\n'

        s = t_0 + t_1
        for c in cls.SEMANTICS.keys():
            s += f'{scores_point[c]} | '

            _scores = scores_bound[c]
            for t in cls.T:
                s += f'{_scores[t]} | '
        return s

    @classmethod
    def csv(
        cls,
        x: Tuple[Dict[str, float], Dict[str, Dict[float, float]]],
        dst: Union[Path, str]
    ):
        scores_point, scores_bound = x
        t_0 = []
        t_1 = []
        tmp = [''] * (len(cls.T) + 1)
        s = []
        for c in cls.SEMANTICS.keys():
            t_0.append(c)
            t_0 += tmp

            t_1.append('F_P')
            for t in cls.T:
                t_1.append(f'F_{t}')

            s.append(scores_point[c])

            _scores = scores_bound[c]
            for t in cls.T:
                s.append(_scores[t])

        csv.writer(Path(dst).open('w')).writerows((t_0, t_1, s))
