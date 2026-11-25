# `quat_to_mat` is modified from PyTorch3D (https://github.com/facebookresearch/pytorch3d,
# pytorch3d/transforms/rotation_conversions.py, `quaternion_to_matrix`) for scalar-last quaternions:
#   Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#   Licensed under the BSD 3-Clause License (https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE).

"""Rotation helpers of the init: quaternion to matrix, and the OpenCV-to-ours camera basis change."""

import torch


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternions (..., 4), scalar last (x, y, z, w), to rotation matrices (..., 3, 3)."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def M_change_of_basis(device=None, dtype=None):
    """Proper rotation from OpenCV camera axes (x right, y down, z forward) to ours (x forward, y right, z down)."""
    return torch.tensor([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.]], device=device, dtype=dtype)
