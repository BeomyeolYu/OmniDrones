# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import functools
from typing import Sequence, Union
from contextlib import contextmanager

@contextmanager
def torch_seed(seed: int=0):
    rng_state = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state_all(rng_state_cuda)


def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg
            for arg in args
        )
        kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped


# @manual_batch
def off_diag(a: torch.Tensor) -> torch.Tensor:
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    return (
        a.flatten(0, 1)[1:]
        .unflatten(0, (n - 1, n + 1))[:, :-1]
        .reshape(n, n - 1, *a.shape[2:])
    )


# @manual_batch
def cpos(p1: torch.Tensor, p2: torch.Tensor):
    assert p1.shape[1] == p2.shape[1]
    return p1.unsqueeze(1) - p2.unsqueeze(0)


# @manual_batch
def others(x: torch.Tensor) -> torch.Tensor:
    return off_diag(x.expand(x.shape[0], *x.shape))


# def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
#     """Convert a batch of rotation matrices to quaternions (x, y, z, w)."""
#     assert R.shape[-2:] == (3, 3)
#     m = R
#     t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

#     def quat_from_diag_max(i):
#         qw = torch.sqrt(1.0 + m[..., i, i] - m[..., (i+1)%3, (i+1)%3] - m[..., (i+2)%3, (i+2)%3]) * 0.5
#         q = torch.zeros_like(m[..., 0, 0].unsqueeze(-1).repeat(1, 4))
#         q[..., i] = qw
#         q[..., (i+1)%3] = (m[..., i, (i+1)%3] + m[..., (i+1)%3, i]) / (4.0 * qw)
#         q[..., (i+2)%3] = (m[..., i, (i+2)%3] + m[..., (i+2)%3, i]) / (4.0 * qw)
#         q[..., 3] = (m[..., (i+2)%3, (i+1)%3] - m[..., (i+1)%3, (i+2)%3]) / (4.0 * qw)
#         return q

#     q = torch.where(
#         (t > 0)[..., None],
#         torch.stack([
#             (m[..., 2, 1] - m[..., 1, 2]),
#             (m[..., 0, 2] - m[..., 2, 0]),
#             (m[..., 1, 0] - m[..., 0, 1]),
#             1.0 + t
#         ], dim=-1) * 0.5 / torch.sqrt(1.0 + t[..., None]),
#         quat_from_diag_max(0)  # fall back to numerically stable case
#     )

#     q = q / q.norm(dim=-1, keepdim=True)
#     return q  # [x, y, z, w]

# def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
#     """
#     Convert a batch of rotation matrices to quaternions (w, x, y, z) format.
#     Args:
#         R: (B, 3, 3) torch tensor
#     Returns:
#         Quaternion: (B, 4) torch tensor in (w, x, y, z)
#     """
#     m = R
#     t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

#     qw = 0.5 * torch.sqrt(1.0 + t + 1e-8)
#     qx = (m[..., 2, 1] - m[..., 1, 2]) / (4.0 * qw + 1e-8)
#     qy = (m[..., 0, 2] - m[..., 2, 0]) / (4.0 * qw + 1e-8)
#     qz = (m[..., 1, 0] - m[..., 0, 1]) / (4.0 * qw + 1e-8)

#     quat = torch.stack([qw, qx, qy, qz], dim=-1)
#     return quat / quat.norm(dim=-1, keepdim=True)

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices to quaternions in (w, x, y, z) format.
    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 4) quaternions
    """
    assert R.shape[-2:] == (3, 3), "Input must be of shape (..., 3, 3)"

    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    qw = torch.zeros_like(trace)
    qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace)
    qz = torch.zeros_like(trace)

    cond = trace > 0
    s = torch.sqrt(trace[cond] + 1.0) * 2.0
    qw[cond] = 0.25 * s
    qx[cond] = (R[..., 2, 1][cond] - R[..., 1, 2][cond]) / s
    qy[cond] = (R[..., 0, 2][cond] - R[..., 2, 0][cond]) / s
    qz[cond] = (R[..., 1, 0][cond] - R[..., 0, 1][cond]) / s

    cond1 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~cond
    s1 = torch.sqrt(1.0 + R[..., 0, 0][cond1] - R[..., 1, 1][cond1] - R[..., 2, 2][cond1]) * 2.0
    qw[cond1] = (R[..., 2, 1][cond1] - R[..., 1, 2][cond1]) / s1
    qx[cond1] = 0.25 * s1
    qy[cond1] = (R[..., 0, 1][cond1] + R[..., 1, 0][cond1]) / s1
    qz[cond1] = (R[..., 0, 2][cond1] + R[..., 2, 0][cond1]) / s1

    cond2 = (R[..., 1, 1] > R[..., 2, 2]) & ~cond & ~cond1
    s2 = torch.sqrt(1.0 + R[..., 1, 1][cond2] - R[..., 0, 0][cond2] - R[..., 2, 2][cond2]) * 2.0
    qw[cond2] = (R[..., 0, 2][cond2] - R[..., 2, 0][cond2]) / s2
    qx[cond2] = (R[..., 0, 1][cond2] + R[..., 1, 0][cond2]) / s2
    qy[cond2] = 0.25 * s2
    qz[cond2] = (R[..., 1, 2][cond2] + R[..., 2, 1][cond2]) / s2

    cond3 = ~cond & ~cond1 & ~cond2
    s3 = torch.sqrt(1.0 + R[..., 2, 2][cond3] - R[..., 0, 0][cond3] - R[..., 1, 1][cond3]) * 2.0
    qw[cond3] = (R[..., 1, 0][cond3] - R[..., 0, 1][cond3]) / s3
    qx[cond3] = (R[..., 0, 2][cond3] + R[..., 2, 0][cond3]) / s3
    qy[cond3] = (R[..., 1, 2][cond3] + R[..., 2, 1][cond3]) / s3
    qz[cond3] = 0.25 * s3

    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True)

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

def vector_to_quat(target: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    """
    Compute quaternion to rotate `source` direction to `target` direction.

    Args:
        target: Target direction vectors of shape [B, 3]
        source: Source reference direction vectors of shape [B, 3] or [3]. Defaults to [0, 0, -1]

    Returns:
        Quaternion tensor of shape [B, 4] in (w, x, y, z) format
    """
    B = target.shape[0]
    if source is None:
        source = torch.tensor([0., 0., -1.], device=target.device).expand(B, 3)
    elif source.ndim == 1:
        source = source.unsqueeze(0).expand(B, 3)

    target = F.normalize(target, dim=-1)
    source = F.normalize(source, dim=-1)

    v = torch.cross(source, target, dim=-1)
    c = (source * target).sum(dim=-1, keepdim=True)

    # handle nearly opposite vectors (180° rotation)
    mask = (c < -0.9999).squeeze(-1)
    if mask.any():
        # Pick an orthogonal axis to rotate 180° around
        alt = torch.cross(source[mask], torch.tensor([1., 0., 0.], device=target.device))
        alt = F.normalize(alt, dim=-1)
        q = torch.zeros((B, 4), device=target.device)
        q[mask, 1:] = alt
        # w = 0 for 180°
        return q

    s = torch.sqrt((1. + c) * 2)
    q = torch.cat([s * 0.5, v / s], dim=-1)  # (w, x, y, z)

    return F.normalize(q, dim=-1)



def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    euler = torch.as_tensor(euler)
    r, p, y = torch.unbind(euler, dim=-1)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def make_cells(
    range_min: Union[Sequence[float], torch.Tensor],
    range_max: Union[Sequence[float], torch.Tensor],
    size: Union[float, Sequence[float], torch.Tensor],
):
    """Compute the cell centers of a n-d grid.

    Examples:
        >>> cells = make_cells([0, 0], [1, 1], 0.1)
        >>> cells[:2, :2]
        tensor([[[0.0500, 0.0500],
                 [0.0500, 0.1500]],

                [[0.1500, 0.0500],
                 [0.1500, 0.1500]]])
    """
    range_min = torch.as_tensor(range_min)
    range_max = torch.as_tensor(range_max)
    size = torch.as_tensor(size)
    shape = ((range_max - range_min) / size).round().int()

    cells = torch.meshgrid(*[torch.linspace(l, r, n+1) for l, r, n in zip(range_min, range_max, shape)], indexing="ij")
    cells = torch.stack(cells, dim=-1)
    for dim in range(cells.dim()-1):
        cells = (cells.narrow(dim, 0, cells.size(dim)-1) + cells.narrow(dim, 1, cells.size(dim)-1)) / 2
    return cells


@manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@manual_batch
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@manual_batch
def euler_rotate(rpy: torch.Tensor, v: torch.Tensor):
    shape = rpy.shape
    r, p, y = torch.unbind(rpy, dim=-1)
    cr = torch.cos(r)
    sr = torch.sin(r)
    cp = torch.cos(p)
    sp = torch.sin(p)
    cy = torch.cos(y)
    sy = torch.sin(y)
    R = torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr
    ], dim=-1).view(*shape[:-1], 3, 3)
    return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)


@manual_batch
def quat_axis(q: torch.Tensor, axis: int=0):
    """
    This function is used to rotate a basis vector (e.g., [1, 0, 0] for the X-axis) by the quaternion
    """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def axis_angle_to_quaternion(angle: torch.Tensor, axis: torch.Tensor):
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)


def axis_angle_to_matrix(angle, axis):
    quat = axis_angle_to_quaternion(angle, axis)
    return quaternion_to_rotation_matrix(quat)


def quat_mul(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def symlog(x: torch.Tensor):
    """
    The symlog transformation described in https://arxiv.org/pdf/2301.04104v1.pdf
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

