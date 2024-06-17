from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from enum import Enum
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import torch

from kaitorch.typing import TorchTensorLike, Real
from kaitorch.data import Group, cell_from_size
from kaitorch.pcd import PointClouds

from .tab import TAB
from .bev import BEV


class Visualizer:
    class Mode(Enum):
        SEMANTICS = 0
        CURVE = 1
        UNSTRUCTURED = 2
        UNCLEAR = 3
        SINGLE = 4
        END = 5
        PREDICTION = 6

    class Palette:
        DEFAULT = (64. / 255, 64. / 255, 64. / 255)
        SEMANTICS = {
            'straight-going_side': (143. / 255, 188. / 255, 120. / 255),
            'turning': (239. / 255, 65. / 255., 67. / 255)
        }

        UNCLEAR = {
            'irregular': (195. / 255, 0. / 255, 120. / 255),
            'occluded': (158. / 255, 49. / 255, 80. / 255),
            'blind': (199. / 255, 109. / 255, 162. / 255),
            'distorted': (42. / 255, 157. / 255, 142. / 255),
            'lengthened': (248. / 255, 172. / 255, 140. / 255),
        }

        CURVE = (255. / 255, 194. / 255, 75. / 255)
        UNSTRUCTURED = (165. / 255, 181. / 255, 93. / 255)
        SINGLE = (204. / 255, 1. / 255, 31. / 255)
        END = (253. / 255, 141. / 255, 60. / 255)
        PREDICTION = SEMANTICS

    def __init__(
        self,
        bev_generator: BEV,
        size: TorchTensorLike[Real] = (256, 512),
        mode: Mode = Mode.SEMANTICS,
        palette: Union[
            Dict[str, Tuple[float, float, float]],
            Tuple[float, float, float]
        ] = Palette.SEMANTICS,
        default: Tuple[float, float, float] = Palette.DEFAULT,
        width: float = 1.,
        show: bool = False,
        save: bool = True,
        dst: Union[Path, str] = Path.cwd().joinpath('vis'),
        *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - bev_generator: function generating a BEV from point cloud.
            - size: size of the BEV.
            - mode: visualization mode for additional annotations.
            - palette
            - default: default color for rendering the one is not highlighted.
            - width: width of lines or size of scatters.
            - show: Whether to show as images.
            - save: Whether to save as images in the specified path `dst`.
            - dst: path for saving visualization results.

        ### Methods:
            - __call__

        __call__
        ### Args:
            - sequence: sequence number.
            - id
            - pcd: point cloud
            - anno: additional annotations. If `None`, the binary segmentation
                annotation will be visualized only.

        '''
        self._bev = bev_generator

        lower_bound = torch.as_tensor((TAB.RANGE_X[0], TAB.RANGE_Y[0]))
        upper_bound = torch.as_tensor((TAB.RANGE_X[1], TAB.RANGE_Y[1]))
        self._group = Group(
            lower_bound,
            cell=cell_from_size(
                lower_bound, upper_bound, torch.as_tensor(size)
            ),
            upper_bound=upper_bound,
            return_offset=True
        )

        self._mode = mode
        self._palette = palette
        self._default = default
        self._width = width

        self._show = show
        self._save = save
        if save:
            if isinstance(dst, str):
                self._dir = Path(dst)
            else:
                self._dir = dst
            self._dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        sequence: str,
        id: str,
        pcd: PointClouds,
        annotations: Optional[
            Union[Sequence[Dict[str, Any]], Sequence[Sequence[Dict[str, Any]]]]
        ] = None
    ) -> None:
        bev = self._bev(pcd)

        _, ax = plt.subplots()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.imshow(bev)

        def _semantics():
            if not isinstance(self._palette, dict):
                warn('an inappropriate palette.')

                color = self._palette
                self._palette = {}
                for k in self.Palette.SEMANTICS.keys():
                    self._palette[k] = color

            for bound in annotations:
                points = torch.add(
                    *self._group(torch.as_tensor(bound['linestrip']))
                )
                ax.plot(
                    points[:, 0], points[:, 1],
                    color=self._palette[bound['semantics']],
                    linewidth=self._width
                )

        def _curve():
            if not isinstance(self._palette, (tuple, list)):
                warn('an inappropriate palette.')
                self._palette = self._palette.values()[0]

            curve = []
            non = []
            for bound in annotations:
                for p in bound['points']:
                    if p['curve']:
                        curve.append(p['xy'])
                    else:
                        non.append(p['xy'])

            if non:
                points = torch.add(*self._group(torch.as_tensor(non)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._default
                )

            if curve:
                points = torch.add(*self._group(torch.as_tensor(curve)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._palette
                )

        def _unstructured():
            if not isinstance(self._palette, (tuple, list)):
                warn('an inappropriate palette.')
                self._palette = self._palette.values()[0]

            u = []
            s = []
            for bound in annotations:
                for p in bound['points']:
                    if p['unstructured']:
                        u.append(p['xy'])
                    else:
                        s.append(p['xy'])

            if s:
                points = torch.add(*self._group(torch.as_tensor(s)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._default
                )

            if u:
                points = torch.add(*self._group(torch.as_tensor(u)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._palette
                )

        def _unclear():
            if not isinstance(self._palette, dict):
                warn('an inappropriate palette.')

                color = self._palette
                self._palette = {}
                for k in self.Palette.UNCLEAR.keys():
                    self._palette[k] = color

            vis: Dict[str, List] = {}
            for k in self._palette.keys():
                vis[k] = []
            default = []

            for bound in annotations:
                for p in bound['points']:
                    if p['irregular']:
                        vis['irregular'].append(p['xy'])
                    elif p['occluded']:
                        vis['occluded'].append(p['xy'])
                    elif p['blind']:
                        vis['blind'].append(p['xy'])
                    elif p['distorted']:
                        vis['distorted'].append(p['xy'])
                    elif p['lengthened']:
                        vis['lengthened'].append(p['xy'])
                    else:
                        default.append(p['xy'])

            if default:
                points = torch.add(*self._group(torch.as_tensor(default)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._default
                )

            for k, points in vis.items():
                if points:
                    points = torch.add(*self._group(torch.as_tensor(points)))
                    ax.scatter(
                        points[:, 0], points[:, 1],
                        s=self._width,
                        color=self._palette[k]
                    )

        def _single():
            if not isinstance(self._palette, (tuple, list)):
                warn('an inappropriate palette.')
                self._palette = self._palette.values()[0]

            for bound in annotations:
                points = torch.add(
                    *self._group(torch.as_tensor(bound['linestrip']))
                )
                ax.plot(
                    points[:, 0], points[:, 1],
                    color=self._palette if bound['single'] else self._default,
                    linewidth=self._width
                )

        def _end():
            if not isinstance(self._palette, (tuple, list)):
                warn('an inappropriate palette.')
                self._palette = self._palette.values()[0]

            end = []
            non = []
            for bound in annotations:
                for p in bound['points']:
                    if p['end']:
                        end.append(p['xy'])
                    else:
                        non.append(p['xy'])

            if non:
                points = torch.add(*self._group(torch.as_tensor(non)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._default
                )

            if end:
                points = torch.add(*self._group(torch.as_tensor(end)))
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._palette
                )

        def _prediction():
            if not isinstance(self._palette, dict):
                warn('an inappropriate palette.')

                color = self._palette
                self._palette = {}
                for k in self.Palette.SEMANTICS.keys():
                    self._palette[k] = color

            for bound in annotations:
                points = torch.add(
                    *self._group(torch.as_tensor(bound['points']))
                )
                ax.scatter(
                    points[:, 0], points[:, 1],
                    s=self._width,
                    color=self._palette[bound['semantics']]
                )

        if annotations is not None:
            match self._mode:
                case self.Mode.SEMANTICS:
                    _semantics()
                case self.Mode.CURVE:
                    _curve()
                case self.Mode.UNSTRUCTURED:
                    _unstructured()
                case self.Mode.UNCLEAR:
                    _unclear()
                case self.Mode.SINGLE:
                    _single()
                case self.Mode.END:
                    _end()
                case self.Mode.PREDICTION:
                    _prediction()

        ax.invert_yaxis()
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
        )

        if self._save:
            p = self._dir.joinpath(sequence, id + '.png')
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=500, bbox_inches='tight', pad_inches=0)
        if self._show:
            plt.show()
        plt.close()
