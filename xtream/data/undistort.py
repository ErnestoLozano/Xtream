"""Undistortion of radtan images, as MASt3R-Fusion reads them (`Intrinsics.from_calib`,
`center_principle_point: True`): alpha 0, centred principal point, linear remap."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def undistorter(seq):
    """(new K, remap function) for `seq`'s distorted images."""
    import cv2
    K, D, (W, H) = np.asarray(seq.K, float), np.asarray(seq.distortion, float), seq.size
    K_opt, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0, (W, H), centerPrincipalPoint=True)
    mx, my = cv2.initUndistortRectifyMap(K, D, None, K_opt, (W, H), cv2.CV_32FC1)
    return K_opt, lambda im: cv2.remap(im, mx, my, cv2.INTER_LINEAR)


def undistorted(seq, out_dir):
    """Every frame of `seq` undistorted, written once to `<out_dir>/<index>.png`; returns the new paths."""
    import cv2
    assert any(seq.distortion), seq.name
    os.makedirs(out_dir, exist_ok=True)
    out = [os.path.join(out_dir, f"{i:06d}.png") for i in range(len(seq.image_paths))]
    _, und = undistorter(seq)

    def one(i):
        if not os.path.exists(out[i]):
            assert cv2.imwrite(out[i], und(cv2.imread(seq.image_paths[i], cv2.IMREAD_UNCHANGED))), out[i]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(len(out))))
    return out
