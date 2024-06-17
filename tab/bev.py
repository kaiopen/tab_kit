from enum import Enum

import torch
from kaitorch.typing import TorchTensor, TorchTensorLike, \
    TorchFloat, Float, Real
from kaitorch.data import Group, cell_from_size, mask_in_range, \
    min_max_normalize, reverse_group, xy_to_rt
from kaitorch.pcd import PointClouds
from kaitorch.utils import pseudo_colors

from .tab import TAB


class BEV:
    class Mode(Enum):
        CONSTANT = 0
        INTENSITY = 1
        Z = 2

    def __init__(
        self,
        size: TorchTensorLike[Real] = (256, 512),
        mode: Mode = Mode.INTENSITY,
        color: TorchTensorLike[Float] = (0, 128. / 255., 1.),
        *args, **kwargs
    ) -> None:
        lower_bound = torch.as_tensor((TAB.RANGE_X[0], TAB.RANGE_Y[0]))
        upper_bound = torch.as_tensor((TAB.RANGE_X[1], TAB.RANGE_Y[1]))
        size = torch.as_tensor(size)
        cell = cell_from_size(lower_bound, upper_bound, size)
        self._group = Group(lower_bound, cell, upper_bound=upper_bound)

        self._h, self._w = size.tolist()
        self._num_pillar = self._h * self._w

        self._mask = torch.logical_not(
            mask_in_range(
                xy_to_rt(
                    reverse_group(
                        torch.stack(
                            torch.meshgrid(
                                torch.arange(self._h),
                                torch.arange(self._w),
                                indexing='ij'
                            ),
                            dim=-1
                        ).reshape(-1, 2) + 0.5,
                        lower_bound, cell
                    )
                ),
                list(TAB.RANGE_RHO) + list(TAB.RANGE_THETA)
            )
        )

        self._mode = mode
        self._color = torch.as_tensor(color)

    def __call__(self, pcd: PointClouds) -> TorchTensor[TorchFloat]:
        TAB.filter_pcd_(pcd)
        match self._mode:
            case self.Mode.CONSTANT:
                return self._constant(pcd)
            case self.Mode.INTENSITY:
                return self._intensity(pcd)
            case self.Mode.Z:
                return self._z(pcd)

    def _constant(self, pcd: PointClouds) -> TorchTensor[TorchFloat]:
        bev = torch.zeros((self._num_pillar, 3))
        bev[self._mask] = 1
        bev = bev.reshape(self._h, self._w, 3)

        groups = self._group(pcd.xy_)
        bev[groups[:, 0], groups[:, 1]] = self._color
        return torch.permute(bev, (1, 0, 2))

    def _intensity(self, pcd: PointClouds) -> TorchTensor[TorchFloat]:
        groups = self._group(pcd.xy_)
        # Assign a pillar ID to each point.
        ids = self._w * groups[:, 0] + groups[:, 1]
        # Sort points by IDs.
        ids, indices = torch.sort(ids)
        # Count the number of points included in a pillar.
        ids, counts = torch.unique_consecutive(
            ids, return_counts=True
        )

        points_z = pcd.z_
        points_i = pcd.intensity_
        mask = torch.zeros((self._num_pillar,), dtype=bool)
        colors = []
        i = 0
        for _id, c in zip(ids.tolist(), torch.cumsum(counts, dim=0).tolist()):
            # For each pillar
            # The indices of the points contained in the pillar.
            inds = indices[i: c]
            colors.append(points_i[inds][torch.argmax(points_z[inds])])
            mask[_id] = True
            i = c
        bev = torch.zeros((self._num_pillar, 3))
        bev[self._mask] = 1
        bev[mask] = pseudo_colors(
            min_max_normalize(
                torch.clip(torch.as_tensor(colors), *TAB.RANGE_INTENSITY),
                *TAB.RANGE_INTENSITY
            )
        )

        # Make the X axis in the LiDAR coordinate system
        # as same as the X axis in the image coordinate system.
        return torch.permute(bev.reshape(self._h, self._w, 3), (1, 0, 2))

    def _z(self, pcd: PointClouds) -> TorchTensor[TorchFloat]:
        groups = self._group(pcd.xy_)
        # Assign a pillar ID to each point.
        ids = self._w * groups[:, 0] + groups[:, 1]
        # Sort points by IDs.
        ids, indices = torch.sort(ids)
        # Count the number of points included in a pillar.
        ids, counts = torch.unique_consecutive(
            ids, return_counts=True
        )

        points_z = pcd.z_
        mask = torch.zeros((self._num_pillar,), dtype=bool)
        colors = []
        i = 0
        for _id, c in zip(ids.tolist(), torch.cumsum(counts, dim=0).tolist()):
            # For each pillar
            # The indices of the points contained in the pillar.
            inds = indices[i: c]
            colors.append(torch.max(points_z[inds]))
            mask[_id] = True
            i = c
        bev = torch.zeros((self._num_pillar, 3))
        bev[self._mask] = 1
        bev[mask] = pseudo_colors(
            min_max_normalize(torch.as_tensor(colors),  *TAB.RANGE_Z)
        )

        # Make the X axis in the LiDAR coordinate system
        # as same as the X axis in the image coordinate system.
        return torch.permute(bev.reshape(self._h, self._w, 3), (1, 0, 2))
