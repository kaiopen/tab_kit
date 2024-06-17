from typing import Any, Dict, Sequence, Tuple

import torch

from kaitorch.typing import TorchTensorLike, Bool, Real
from kaitorch.data import Group, cell_from_size

from .tab import TAB
from .evaluator import SIZE


class Sampler:
    def __init__(
        self,
        size: TorchTensorLike[Real] = SIZE,
        error: TorchTensorLike[Real] = torch.as_tensor(0),
        closed: TorchTensorLike[Bool] = torch.as_tensor(False),
        *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - size: size of a BEV.
            - error: tolerant error. Its shape should be `(C,)` or its length
                should be larger than 1. An error should be within [0, 1).
            - closed: Whether do loop closing. Its shape should be `(C,)` or
                its length should be larger than 1. If `True`, the last group
                will be merged into the first group.

        ### Methods:
            - __call__

        __call__
        ### Args:
            - linestrip: a boundary in a linestrip.

        ### Returns:
            - Keypoints.

        '''
        r = torch.as_tensor(list(TAB.RANGE_X) + list(TAB.RANGE_Y))
        lower_bound = r[[0, 2]]
        upper_bound = r[[1, 3]]
        size = torch.as_tensor(size)
        error = torch.as_tensor(error)
        closed = torch.as_tensor(closed)
        self._group = Group(
            lower_bound,
            cell_from_size(lower_bound, upper_bound, size, error, closed),
            error, closed, upper_bound
        )

        self._w = size.tolist()[1]

    def __call__(
        self, points: Sequence[Dict[str, Any]]
    ) -> Sequence[Tuple[float, float]]:
        points_xy = []
        for p in points:
            points_xy.append(p['xy'])
        points_xy = torch.as_tensor(points_xy)

        groups = self._group(points_xy)
        # Assign a pillar ID to each point.
        # Sort points by IDs.
        ids, indices = torch.sort(groups[:, 0] * self._w + groups[:, 1])
        # Count the number of the points included in a pillar.
        _, counts = torch.unique_consecutive(ids, return_counts=True)

        i = 0
        samples = []
        for c in torch.cumsum(counts, dim=0).tolist():
            inds = indices[i: c]

            straight = True
            structured = True
            regular = True
            non_occluded = True
            non_blind = True
            non_distorted = True
            non_lengthened = True
            non_end = True

            for j in inds.tolist():
                p = points[j]

                if straight:
                    straight = not p['curve']

                if structured:
                    structured = not p['unstructured']

                if regular:
                    regular = not p['irregular']
                if non_occluded:
                    non_occluded = not p['occluded']
                if non_blind:
                    non_blind = not p['blind']
                if non_distorted:
                    non_distorted = not p['distorted']
                if non_lengthened:
                    non_lengthened = not p['lengthened']

                if non_end:
                    non_end = not p['end']

            samples.append(
                {
                    'xy': torch.mean(points_xy[inds], dim=0).tolist(),
                    'curve': not straight,
                    'unstructured': not structured,
                    'irregular': not regular,
                    'occluded': not non_occluded,
                    'blind': not non_blind,
                    'distorted': not non_distorted,
                    'lengthened': not non_lengthened,
                    'end': not non_end
                }
            )

            i = c
        return samples
