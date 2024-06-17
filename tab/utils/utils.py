import torch

from kaitorch.typing import TorchTensor, TorchFloat


def squared_distances_points_to_linestrip(
    points: TorchTensor[TorchFloat],  # (N, 2)
    linestrip: TorchTensor[TorchFloat]  # (M, 2)
) -> TorchTensor[TorchFloat]:
    r'''Squared distances from points to a linestrip.

    The distance from a point to a linestrip is the minimum distance between
    the point and the linestrip.

    All points should be represented in the 2D rectangular coordinate system.

    ### Args:
        - points: Its shape should be `(N, 2)`.
        - linestrip: Linestrip is comprised of a sequence of points. Its shape
            should be `(M, 2)` and `M >= 2`.

    ### Returns:
        - Squared distances between the points and the linestrip. Its shape is
            `(N,)`.

    '''
    a = linestrip[:-1]  # (M - 1, 2)
    b = linestrip[1:]  # (M - 1, 2)

    num_p = len(points)
    num_seg = len(a)
    distances = torch.zeros(
        (num_p, num_seg), dtype=points.dtype, device=points.device
    )  # (N, M)

    points = points.unsqueeze(1)  # (N, 1, 2)
    a = a.unsqueeze(0)  # (1, M, 2)
    b = b.unsqueeze(0)  # (1, M, 2)

    ab = b - a  # vector AB  (1, M, 2)
    ap = points - a  # vector AP  (N, M, 2)
    pb = b - points  # vector PB  (N, M, 2)

    dot = ab * ap
    dot = dot[..., 0] + dot[..., 1]  # (N, M)
    mask_1 = dot <= 0  # (N, M)
    _ap = ap[mask_1] ** 2
    distances[mask_1] = _ap[..., 0] + _ap[..., 1]

    d2 = ab ** 2
    d2 = d2[..., 0] + d2[..., 1]  # (1, M)
    mask_2 = dot >= d2  # (N, M)
    _pb = pb[mask_2] ** 2
    distances[mask_2] = _pb[..., 0] + _pb[..., 1]

    mask = torch.logical_not(torch.logical_or(mask_1, mask_2))
    a = a.expand(num_p, num_seg, 2)[mask]  # (X, 2)
    c = (dot[mask] / d2.expand_as(mask)[mask]).unsqueeze_(-1) \
        * (b.expand(num_p, num_seg, 2)[mask] - a) \
        + a  # (X, 2)
    cp = (points.expand(num_p, num_seg, 2)[mask] - c) ** 2
    distances[mask] = cp[..., 0] + cp[..., 1]
    return torch.min(distances, dim=1)[0]
