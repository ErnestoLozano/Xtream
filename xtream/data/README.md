# Dataset layouts

Paths are the keys of `configs/datasets.yaml` (`raw`, `prepared`, ...), under `$XTREAM_DATA` (default `data/`).
`scripts/download_data.py` writes the raw trees, `scripts/prepare_data.py` the prepared ones.

| dataset | reads | ground truth |
|---|---|---|
| EuRoC | `<prepared>/<seq>/cam0/{data/*.png, data.csv, intrinsics.txt}` (undistorted cam0); `<raw>/<room>/<seq>/mav0/{imu0/data.csv, imu0/sensor.yaml, cam0/sensor.yaml}` | `mav0/state_groundtruth_estimate0/data.csv` (body = IMU), by time |
| KITTI | `<raw>/sequences/<seq>/{image_2/*.png, times.txt, calib.txt}` | `<raw>/poses/<seq>.txt` (cam0, moved to cam2), by frame index |
| Oxford Spires | `<prepared>/<sequence>/raw/{cam0/*.jpg, rgb.txt, cam0_intrinsics.txt}` (rectified front camera); `<raw>/sequences/<sequence>/raw/imu.csv`; `<raw>/calibration/imu.yaml`; `oxford_spires_sensor.yaml` | `<raw>/sequences/<sequence>/processed/trajectory/gt-tum.txt` (base frame), by time |
| ScaRF | `<prepared>/<seq>/<seq>_bag_sequence/{images/camera_rgb_undistorted/*.jpg, imu/imu_right/data.csv}`; `<raw>/config/*.yaml` | `<raw>/<seq>/<seq>_gt/poses_gt.txt` (cam0), by time |
| VBR | `<frames>/<seq>_processed_aligned/{rgb/*.png, intrinsics.txt}`; `vbr_frame_maps/<seq>.csv` (stamps); `<raw>/<site>/<seq>/{vbr_calib.yaml, imu.csv}` | `<frames>/processed_gt/<seq>_gt.txt` (cam_l), by frame index |
| custom | `<raw>/<seq>/{images/*.{png,jpg}, times.txt (optional)}`: your own images, monocular | `<raw>/<seq>/gt.txt` (TUM, camera; optional), by time |

The VBR release has no image stamps: `vbr_frame_maps/<seq>.csv` gives each released file's camera stamp [ns],
matched to the official bags' image messages, on the clock of the bags' IMU and ground truth. Mono-inertial runs
undistort VBR's images once into `<undistorted>/<seq>/`.
