#!/usr/bin/env python3
# Each preparation reproduces its source's output exactly:
#   EuRoC: MASt3R-Fusion's undistortion (https://github.com/GREAT-WHU/MASt3R-Fusion, mast3r_fusion/dataloader.py,
#     `EurocDataset`, `Intrinsics.from_calib`).
#   Oxford Spires: adapted from CLoSeR (https://github.com/MoyangLi00/CLoSeR, scripts_download/undistort_oxford_spires.py,
#     `--method lingbot`; the repository states no license), itself following the dataset's rectifier
#     (https://github.com/ori-drs/oxford_spires_dataset, scripts/rectify_image_from_distorted_image.py).
#   ScaRF: the steps of FoundationVIO's scripts/bag_to_folder.py and scripts/scarf_undistort.py
#     (https://github.com/ori-drs/FoundationVIO), written anew on rosbags and OpenCV.
"""Prepare downloaded sequences into the images Xtream reads (configs/datasets.yaml, `prepared`).

    python scripts/prepare_data.py euroc --sequences V1_01_easy
    python scripts/prepare_data.py oxford_spires
    python scripts/prepare_data.py scarf --sequences r05

  * EuRoC: cam0 undistorted as MASt3R-Fusion's `EurocDataset` reads it (grey read as BGR; pinhole + radtan from
    `sensor.yaml`, `getOptimalNewCameraMatrix` with alpha 0 and a centred principal point, linear remap), every row
    of `mav0/cam0/data.csv` -> `<prepared>/<seq>/cam0/{data, data.csv, intrinsics.txt}`.
  * Oxford Spires: the front fisheye camera rectified to a pinhole with the fisheye K and the principal point
    moved to the image centre (`calibration/cam0.yaml`, equidistant), linear remap ->
    `<prepared>/<sequence>/raw/{cam0, rgb.txt, cam0_intrinsics.txt}`.
  * ScaRF: the ROS 2 bag's camera (`/insta/cam0/image_raw/compressed`, every frame) and IMU
    (`/insta/imu/data_raw`) to a folder, then the fisheye cam0 rectified onto `config/rectified_intrinsic.yaml`
    -> `<prepared>/<seq>/<seq>_bag_sequence/{images/camera_rgb_undistorted, imu/imu_right/data.csv}`.
KITTI needs no preparation; VBR is undistorted on its first mono-inertial run.
"""
import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xtream.paths import data_dir, dataset_cfg  # noqa: E402


def _map(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fn, items))


def euroc(seqs, workers):
    for s in seqs:
        mav0 = os.path.join(data_dir("euroc"), dataset_cfg("euroc")["sequences"][s], "mav0")
        out = os.path.join(data_dir("euroc", "prepared"), s, "cam0")
        if os.path.exists(os.path.join(out, "intrinsics.txt")):
            continue
        cam = yaml.safe_load(open(os.path.join(mav0, "cam0", "sensor.yaml")))
        W, H = cam["resolution"]
        fx, fy, cx, cy = cam["intrinsics"]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        D = np.array(cam["distortion_coefficients"])
        K_opt, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0, (W, H), centerPrincipalPoint=True)
        mx, my = cv2.initUndistortRectifyMap(K, D, None, K_opt, (W, H), cv2.CV_32FC1)
        rows = np.loadtxt(os.path.join(mav0, "cam0", "data.csv"), delimiter=",", dtype=str, comments="#", ndmin=2)
        os.makedirs(os.path.join(out, "data"), exist_ok=True)

        def one(f):
            dst = os.path.join(out, "data", f)
            if not os.path.isfile(dst):
                img = cv2.cvtColor(cv2.imread(os.path.join(mav0, "cam0", "data", f), cv2.IMREAD_GRAYSCALE),
                                   cv2.COLOR_GRAY2BGR)
                assert cv2.imwrite(dst, cv2.remap(img, mx, my, cv2.INTER_LINEAR)), dst
        _map(one, list(rows[:, 1]), workers)
        shutil.copy(os.path.join(mav0, "cam0", "data.csv"), os.path.join(out, "data.csv"))
        with open(os.path.join(out, "intrinsics.txt"), "w") as fh:
            fh.write("# fx fy cx cy width height (undistorted)\n")
            fh.write(f"{K_opt[0, 0]:.6f} {K_opt[1, 1]:.6f} {K_opt[0, 2]:.6f} {K_opt[1, 2]:.6f} {W} {H}\n")
        print(f"[euroc] {s}: {len(rows)} frames -> {out}")


def oxford_spires(seqs, workers):
    root = data_dir("oxford_spires")
    c = yaml.safe_load(open(os.path.join(root, "calibration", "cam0.yaml")))
    assert c["distortion_model"] == "equidistant", c["distortion_model"]
    K = np.array(c["camera_matrix"]["data"], np.float64).reshape(3, 3)
    D = np.array(c["distortion_coefficients"]["data"], np.float64)
    W, H = int(c["image_width"]), int(c["image_height"])
    new_K = K.copy()
    new_K[0, 2], new_K[1, 2] = W / 2.0, H / 2.0
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (W, H), cv2.CV_32FC1)
    for s in seqs:
        long = dataset_cfg("oxford_spires")["sequences"][s]
        src = os.path.join(root, "sequences", long, "raw", "cam0")
        raw = os.path.join(data_dir("oxford_spires", "prepared"), long, "raw")
        dst = os.path.join(raw, "cam0")
        os.makedirs(dst, exist_ok=True)
        names = sorted(n for n in os.listdir(src) if n.lower().endswith((".jpg", ".jpeg", ".png")))

        def one(n):
            if not os.path.exists(os.path.join(dst, n)):
                img = cv2.imread(os.path.join(src, n))
                tmp = os.path.join(dst, n + ".tmp" + os.path.splitext(n)[1])
                assert cv2.imwrite(tmp, cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT)), n
                os.replace(tmp, os.path.join(dst, n))
        _map(one, names, workers)
        with open(os.path.join(raw, "rgb.txt"), "w") as f:
            for n in names:
                f.write(f"{os.path.splitext(n)[0]} cam0/{n}\n")
        with open(os.path.join(raw, "cam0_intrinsics.txt"), "w") as f:
            f.write("# fx fy cx cy width height (rectified pinhole)\n")
            f.write(f"{new_K[0, 0]:.6f} {new_K[1, 1]:.6f} {new_K[0, 2]:.6f} {new_K[1, 2]:.6f} {W} {H}\n")
        print(f"[oxford_spires] {s}: {len(names)} images -> {dst}")


def _scarf_yaml(path):
    txt = open(path).read()
    return yaml.safe_load(txt.split("\n", 1)[1] if txt.startswith("%YAML") else txt)


def scarf(seqs, workers):
    from rosbags.highlevel import AnyReader
    from rosbags.image import message_to_cvimage
    root = data_dir("scarf")
    cfg = os.path.join(root, "config")
    cam = _scarf_yaml(os.path.join(cfg, "kalibr_imucam_chain.yaml"))["cam0"]
    rect = _scarf_yaml(os.path.join(cfg, "rectified_intrinsic.yaml"))
    pin = lambda fu, fv, pu, pv: np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(pin(*cam["intrinsics"]), np.array(cam["distortion_coeffs"], np.float64),
                                                 np.eye(3), pin(*rect["pinhole_intrinsics"]),
                                                 tuple(rect["pinhole_resolution"]), cv2.CV_16SC2)
    q100 = [cv2.IMWRITE_JPEG_QUALITY, 100]
    for s in seqs:
        out = os.path.join(data_dir("scarf", "prepared"), s, f"{s}_bag_sequence")
        und = os.path.join(out, "images", "camera_rgb_undistorted")
        if os.path.isdir(und) and os.listdir(und):
            continue
        raw_dir, imu_dir = os.path.join(out, "images", "camera_rgb"), os.path.join(out, "imu", "imu_right")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(imu_dir, exist_ok=True)
        bag = Path(root, s, f"{s}_bag")
        n = 0
        with open(os.path.join(imu_dir, "data.csv"), "w", newline="") as f, AnyReader([bag]) as reader:
            w = csv.writer(f)
            w.writerow(["#timestamp [ns]", "omega_x", "omega_y", "omega_z", "acc_x", "acc_y", "acc_z"])
            conns = [c for c in reader.connections if c.topic in ("/insta/cam0/image_raw/compressed",
                                                                  "/insta/imu/data_raw")]
            for conn, _, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                if conn.topic == "/insta/imu/data_raw":
                    g, a = msg.angular_velocity, msg.linear_acceleration
                    w.writerow([t, g.x, g.y, g.z, a.x, a.y, a.z])
                else:
                    rgb = message_to_cvimage(msg, "rgb8")
                    cv2.imwrite(os.path.join(raw_dir, f"image_{t // 1_000_000_000:010d}_{t % 1_000_000_000:09d}.jpg"),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), q100)
                    n += 1
        os.makedirs(und, exist_ok=True)
        names = sorted(os.listdir(raw_dir))
        _map(lambda p: cv2.imwrite(os.path.join(und, p), cv2.remap(cv2.imread(os.path.join(raw_dir, p)), m1, m2,
                                                                   cv2.INTER_LINEAR), q100), names, workers)
        shutil.rmtree(raw_dir)
        print(f"[scarf] {s}: {n} images -> {und}")


DATASETS = {"euroc": euroc, "oxford_spires": oxford_spires, "scarf": scarf}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--sequences", default=None, help="comma-separated (default: every sequence)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    known = list(dataset_cfg(a.dataset)["sequences"])
    seqs = a.sequences.split(",") if a.sequences else known
    assert all(s in known for s in seqs), (seqs, known)
    DATASETS[a.dataset](seqs, a.workers)


if __name__ == "__main__":
    main()
