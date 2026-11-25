"""Camera-axis conventions and the Sim(3) fit between two sets of poses."""

import gtsam
import numpy as np

# OpenCV camera axes (the foundation model's) -> ours (x forward, y right, z down): v_ours = PM @ v_opencv
PM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)


def to_ours(T):
    """OpenCV camera-to-world poses [..., 4, 4] -> our axes, world and camera alike."""
    M = np.eye(4)
    M[:3, :3] = PM
    return M @ np.asarray(T, float) @ M.T


def to_opencv(T):
    """Inverse of `to_ours`."""
    M = np.eye(4)
    M[:3, :3] = PM
    return M.T @ np.asarray(T, float) @ M


def sim3(R, t, s):
    return gtsam.Similarity3(gtsam.Rot3(np.asarray(R, float)), np.asarray(t, float), float(s))


def _proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def align(Pa, Pb):
    """Sim(3) Z with Pa ~ Z * Pb (poses 4x4, same frames): rotation from the orientations, scale and translation
    from the centres."""
    R = _proj_so3(sum(a[:3, :3] @ b[:3, :3].T for a, b in zip(Pa, Pb)))
    ca = np.array([a[:3, 3] for a in Pa])
    cb = np.array([b[:3, 3] for b in Pb])
    x, y = cb - cb.mean(0), ca - ca.mean(0)
    s = float(np.sum(y * (x @ R.T)) / max(np.sum(x * x), 1e-12)) if len(Pa) > 1 else 1.0
    s = s if s > 0 else 1.0
    t = ca.mean(0) - s * R @ cb.mean(0)
    return sim3(R, t / s, s)


def se3_to_pose7(T):
    """4x4 SE(3) -> [t(3), q(4) xyzw]."""
    from scipy.spatial.transform import Rotation
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()])
