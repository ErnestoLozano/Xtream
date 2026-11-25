#!/usr/bin/env python3
"""Download the datasets from their official sources into the layout of configs/datasets.yaml.

    python scripts/download_data.py euroc --sequences V1_01_easy
    python scripts/download_data.py kitti --sequences 00,05          # range requests: only those sequences' images
    python scripts/download_data.py oxford_spires --sequences obs02
    python scripts/download_data.py scarf --sequences r05
    python scripts/download_data.py vbr --sequences campus_train1
    python scripts/download_data.py euroc                            # every sequence

Then `python scripts/prepare_data.py <dataset>` (EuRoC, Oxford Spires, ScaRF). Re-running skips what is there.

  * EuRoC MAV: ETH Research Collection (doi:10.3929/ethz-b-000690084), one archive per room; only the
    per-sequence ASL zips are read from it (HTTP range requests), unpacked to `<raw>/<room>/<seq>/mav0`.
  * KITTI odometry: the official S3 zips; calibration, poses, and the colour images `image_2` of the requested
    sequences only, read from the 65 GB colour zip by range requests.
  * Oxford Spires: Hugging Face `ori-drs/oxford_spires_dataset`: per sequence the camera images (cam0), the IMU
    and the ground-truth trajectory; the calibration.
  * ScaRF: the authors' Google Drive folder (`gdown`): the calibration and the sequences' ROS 2 bags with ground
    truth.
  * VBR: the LoGeR authors' processed release on Hugging Face (`Junyi42/vbr_processed`: images, intrinsics,
    ground truth), and the IMU (`/imu/data`) with `vbr_calib.yaml` from the official FTP bags, ~10 GB each
    (43-177 GB per sequence): each is downloaded, its IMU extracted, and deleted. When the official server does
    not answer, from the archive of the same files extracted from those bags (Hugging Face
    `ernestolozano/vbr-imu-calib`, ~40 MB, CC BY-SA 4.0); `--vbr_imu bags|archive` forces either.

The datasets keep their own licenses and terms of use; see their websites.
"""
import argparse
import os
import shutil
import struct
import sys
import tarfile
import urllib.request
import zipfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xtream.paths import data_dir, dataset_cfg  # noqa: E402

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
_CHUNK = 1 << 22
_EUROC_ARCHIVES = {
    "machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
    "vicon_room1": "02ecda9a-298f-498b-970c-b7c44334d880",
    "vicon_room2": "ea12bc01-3677-4b4c-853d-87c7870b8c44",
}
_EUROC_URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{}/content"
_KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/{}"
_OXFORD_REPO = "ori-drs/oxford_spires_dataset"
_SCARF_IDS = {"config/kalibr_imu_chain.yaml": "1aIFSpVWAEmAHFQLsh5qSkofPfXclwLbW",
              "config/kalibr_imucam_chain.yaml": "1rn8ROYIH7R8Ssqw6FE4DLH03ueJTS0xX",
              "config/rectified_intrinsic.yaml": "1jEE2wDQ_972w6FEAgqukpfVgONgV5vOw",
              "r01.zip": "1j2OurHPiWb-AKLhQmnG1eiZXgFIaEtRE", "r02.zip": "1mu9GFEaF75D_j1lmLhnVqQ7WxAF3eUnw",
              "r03.zip": "1bLVTfWyE0nPSbcOx3LGxxLYn0XzexYl7", "r04.zip": "14Rw2eMe_bvIN8P3U3NR7AO0SCP1xxMNi",
              "r05.zip": "1nOc6ns0wl-Uu_6m0allAPUXac8XYIrhk"}
_VBR_FRAMES_URL = "https://huggingface.co/datasets/Junyi42/vbr_processed/resolve/main/{}.tar.gz"
_VBR_FTP = ("151.100.59.119", "anonymous")          # the official devkit's FTP (vbr_devkit/download)
# The IMU and calibration of the 7 training sequences, extracted from the official bags (`--vbr_imu bags` does the
# same): <site>/<seq>/{vbr_calib.yaml, imu.csv}, CC BY-SA 4.0 as VBR. A URL or local path; $XTREAM_VBR_IMU overrides it.
_VBR_IMU_ARCHIVE = os.environ.get(
    "XTREAM_VBR_IMU", "https://huggingface.co/datasets/ernestolozano/vbr-imu-calib/resolve/main/vbr_imu_calib.tar.gz")
_VBR_IMU_HEADER = ("#timestamp [ns],gyro_x [rad/s],gyro_y [rad/s],gyro_z [rad/s],"
                   "acc_x [m/s^2],acc_y [m/s^2],acc_z [m/s^2]\n")


# ----------------------------------------------------------------------------------------------------------- http
def _request(url, start=None, end=None):
    headers = {"User-Agent": _UA}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def _copy(src, dst_path, total, label, decompress=None):
    """Stream `src` to `dst_path` (via a .part file), optionally inflating it."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    part, done = dst_path + ".part", 0
    with open(part, "wb") as f:
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            done += len(chunk)
            f.write(decompress.decompress(chunk) if decompress else chunk)
            print(f"\r  {label}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        if decompress:
            f.write(decompress.flush())
    print()
    os.replace(part, dst_path)


def download(url, dst_path):
    """Download `url` to `dst_path` unless it is already there."""
    if os.path.exists(dst_path):
        return
    with _request(url) as r:
        _copy(r, dst_path, int(r.headers.get("Content-Length", 0)), os.path.basename(dst_path))


class RemoteZip:
    """Reads members of a zip over HTTP without downloading the whole archive."""

    def __init__(self, url):
        self.url = url
        with _request(url, 0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        tail = self._read(max(0, self.size - (1 << 18)), self.size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        if cd_off == 0xFFFFFFFF:                                     # zip64
            eocd64 = tail.rfind(b"PK\x06\x06")
            cd_size, cd_off = struct.unpack("<QQ", tail[eocd64 + 40:eocd64 + 56])
        self.members = self._parse_central_directory(self._read(cd_off, cd_off + cd_size - 1))

    def _read(self, start, end):
        with _request(self.url, start, end) as r:
            return r.read()

    @staticmethod
    def _parse_central_directory(cd):
        members, p = {}, 0
        while cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
            offset, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + n].decode("utf-8")
            extra, q = cd[p + 46 + n:p + 46 + n + e], 0
            while q + 4 <= len(extra):                                 # zip64 extra field
                tag, ln = struct.unpack("<HH", extra[q:q + 4])
                if tag == 0x0001:
                    vals, k = list(struct.unpack(f"<{ln // 8}Q", extra[q + 4:q + 4 + ln // 8 * 8])), 0
                    if usize == 0xFFFFFFFF:
                        usize, k = vals[k], k + 1
                    if csize == 0xFFFFFFFF:
                        csize, k = vals[k], k + 1
                    if offset == 0xFFFFFFFF:
                        offset = vals[k]
                q += 4 + ln
            members[name] = (offset, csize, usize, method)
            p += 46 + n + e + c
        return members

    def extract(self, name, dst_path):
        """Stream member `name` to `dst_path` unless it is already there."""
        offset, csize, usize, method = self.members[name]
        if os.path.exists(dst_path) and os.path.getsize(dst_path) == usize:
            return
        header = self._read(offset, offset + 29)
        n, e = struct.unpack("<HH", header[26:30])
        start = offset + 30 + n + e
        with _request(self.url, start, start + csize - 1) as r:
            _copy(r, dst_path, csize, os.path.basename(dst_path),
                  zlib.decompressobj(-15) if method == 8 else None)

    def extract_many(self, names, dst_of, block=1 << 27):
        """Many small members (`names`, written to `dst_of(name)`), read in blocks of consecutive members of at
        most `block` bytes per request."""
        todo = sorted((self.members[n][0], n) for n in names
                      if not (os.path.exists(dst_of(n)) and os.path.getsize(dst_of(n)) == self.members[n][2]))
        ends = sorted(o for o, _, _, _ in self.members.values()) + [self.size]
        nxt = {o: ends[i + 1] for i, o in enumerate(ends[:-1])}
        i, done = 0, 0
        while i < len(todo):
            j = i
            while j + 1 < len(todo) and nxt[todo[j][0]] == todo[j + 1][0] and \
                    nxt[todo[j + 1][0]] - todo[i][0] <= block:
                j += 1
            lo, hi = todo[i][0], nxt[todo[j][0]]
            buf = self._read(lo, hi - 1)
            for off, n in todo[i:j + 1]:
                _, csize, _, method = self.members[n]
                k = off - lo
                ln, le = struct.unpack("<HH", buf[k + 26:k + 30])
                data = buf[k + 30 + ln + le: k + 30 + ln + le + csize]
                out = dst_of(n)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out + ".part", "wb") as f:
                    f.write(zlib.decompress(data, -15) if method == 8 else data)
                os.replace(out + ".part", out)
            done += j + 1 - i
            print(f"\r  {done} / {len(todo)} files", end="", flush=True)
            i = j + 1
        print()


def _unzip(zip_path, dst_dir, members=None):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dst_dir, members=members)


# -------------------------------------------------------------------------------------------------------- datasets
def euroc(seqs, keep_zips=False):
    root, by_room = data_dir("euroc"), {}
    for s in seqs:
        by_room.setdefault(dataset_cfg("euroc")["sequences"][s].split("/")[0], []).append(s)
    for room, names in by_room.items():
        print(f"[euroc] {room}: {', '.join(names)}")
        archive = RemoteZip(_EUROC_URL.format(_EUROC_ARCHIVES[room]))
        for s in names:
            d = os.path.join(root, room, s)
            if os.path.isdir(os.path.join(d, "mav0")):
                continue
            z = os.path.join(d, f"{s}.zip")
            archive.extract(f"{room}/{s}/{s}.zip", z)
            _unzip(z, d)
            if not keep_zips:
                os.remove(z)


def kitti(seqs, keep_zips=False):
    root = data_dir("kitti")
    strip = lambda n: os.path.join(root, n.split("/", 1)[1])             # "dataset/..." -> <root>/...
    for name, want in (("data_odometry_calib.zip", lambda n: n.split("/")[2] in seqs),
                       ("data_odometry_poses.zip", lambda n: os.path.basename(n)[:-4] in seqs)):
        print(f"[kitti] {name}")
        z = RemoteZip(_KITTI_URL.format(name))
        z.extract_many([n for n in z.members if not n.endswith("/") and want(n)], strip)
    print(f"[kitti] data_odometry_color.zip: image_2 of {', '.join(seqs)}")
    z = RemoteZip(_KITTI_URL.format("data_odometry_color.zip"))
    for s in seqs:
        names = [n for n in z.members if n.startswith(f"dataset/sequences/{s}/image_2/") and not n.endswith("/")]
        print(f"  {s}: {len(names)} images")
        z.extract_many(names, strip)


def oxford_spires(seqs, keep_zips=False):
    from huggingface_hub import hf_hub_download
    root = data_dir("oxford_spires")
    files = ["calibration/cam0.yaml", "calibration/imu.yaml"]
    for s in seqs:
        long = dataset_cfg("oxford_spires")["sequences"][s]
        files += [f"sequences/{long}/raw/imu.csv", f"sequences/{long}/processed/trajectory/gt-tum.txt"]
        if not os.path.isdir(os.path.join(root, "sequences", long, "raw", "cam0")):
            files.append(f"sequences/{long}/raw/images.zip")
    for f in files:
        print(f"[oxford_spires] {f}")
        hf_hub_download(_OXFORD_REPO, f, repo_type="dataset", local_dir=root)
        if f.endswith("images.zip"):
            z = os.path.join(root, f)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.dirname(z), members=[m for m in zf.namelist() if m.startswith("cam0/")])
            if not keep_zips:
                os.remove(z)


def scarf(seqs, keep_zips=False):
    import gdown
    root = data_dir("scarf")
    for f in [k for k in _SCARF_IDS if k.startswith("config/")] + [f"{s}.zip" for s in seqs]:
        dst = os.path.join(root, f)
        if f.endswith(".zip") and os.path.isdir(os.path.join(root, f[:-4], f"{f[:-4]}_bag")):
            continue
        if not os.path.exists(dst):
            print(f"[scarf] {f}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            gdown.download(id=_SCARF_IDS[f], output=dst, quiet=False)
        if f.endswith(".zip"):
            _unzip(dst, root)
            if not keep_zips:
                os.remove(dst)


def _vbr_imu_from_bags(seq, site_seq, keep_bags=False):
    """`vbr_calib.yaml` and the IMU (`imu.csv`: `/imu/data` header stamp [ns], gyro xyz, acc xyz) from the official
    bags, one bag at a time (downloaded, read, deleted)."""
    import ftplib
    from pathlib import Path

    from rosbags.highlevel import AnyReader
    d = os.path.join(data_dir("vbr"), site_seq)
    imu = os.path.join(d, "imu.csv")
    if os.path.exists(imu + ".done"):
        return
    os.makedirs(d, exist_ok=True)

    def connect():
        ftp = ftplib.FTP(_VBR_FTP[0], timeout=120)
        ftp.login(_VBR_FTP[1], "")
        ftp.cwd(f"/vbr_slam/{site_seq}")
        return ftp
    ftp = connect()
    names = sorted(ftp.nlst())
    with open(os.path.join(d, "vbr_calib.yaml"), "wb") as f:
        ftp.retrbinary("RETR vbr_calib.yaml", f.write)
    done_bags = set(open(imu + ".bags").read().split()) if os.path.exists(imu + ".bags") else set()
    if not done_bags:
        with open(imu, "w") as f:
            f.write(_VBR_IMU_HEADER)
    for bag in [n for n in names if n.endswith(".bag")]:
        if bag in done_bags:
            continue
        dst = os.path.join(d, bag)
        print(f"[vbr] {seq}: {bag} ({ftp.size(bag) / 2 ** 30:.1f} GiB)", flush=True)
        with open(dst, "wb") as f:
            ftp.retrbinary("RETR " + bag, f.write, blocksize=1 << 20)
        with AnyReader([Path(dst)]) as reader, open(imu, "a") as f:
            conns = [c for c in reader.connections if c.topic == "/imu/data"]
            for conn, _, raw in reader.messages(connections=conns):
                m = reader.deserialize(raw, conn.msgtype)
                ts = int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
                a, g = m.linear_acceleration, m.angular_velocity
                f.write(f"{ts},{g.x!r},{g.y!r},{g.z!r},{a.x!r},{a.y!r},{a.z!r}\n")
        with open(imu + ".bags", "a") as f:
            f.write(bag + "\n")
        if not keep_bags:
            os.remove(dst)
        try:
            ftp.voidcmd("NOOP")
        except Exception:
            ftp = connect()
    open(imu + ".done", "w").close()


def _vbr_imu_from_archive(seqs):
    """The IMU and calibration of `seqs` from the archive `_VBR_IMU_ARCHIVE` (URL or path)."""
    root = data_dir("vbr")
    src = _VBR_IMU_ARCHIVE
    if "://" in src:
        dst = os.path.join(root, os.path.basename(src))
        download(src, dst)
        src = dst
    want = [dataset_cfg("vbr")["sequences"][s] + "/" for s in seqs]
    with tarfile.open(src) as t:
        t.extractall(root, members=[m for m in t.getmembers() if any(m.name.startswith(w) for w in want)])
    for s in seqs:
        d = os.path.join(root, dataset_cfg("vbr")["sequences"][s])
        assert os.path.exists(os.path.join(d, "vbr_calib.yaml")) and os.path.exists(
            os.path.join(d, "imu.csv")), f"{s} missing from {src}"
    print(f"[vbr] IMU and calibration of {', '.join(seqs)} from {_VBR_IMU_ARCHIVE}")


def _vbr_server_up(timeout=20):
    """Whether VBR's official FTP server answers."""
    import socket
    try:
        socket.create_connection((_VBR_FTP[0], 21), timeout=timeout).close()
        return True
    except OSError:
        return False


def vbr(seqs, keep_zips=False, imu_source="bags"):
    frames = data_dir("vbr", "frames")
    os.makedirs(os.path.join(frames, "processed_gt"), exist_ok=True)
    for s in seqs:
        dst = os.path.join(frames, f"{s}_processed_aligned")
        if not os.path.exists(os.path.join(dst, ".done")):
            tar = os.path.join(frames, f"{s}.tar.gz")
            print(f"[vbr] {s}: frames")
            download(_VBR_FRAMES_URL.format(s), tar)
            tmp = os.path.join(frames, f"_extract_{s}")
            shutil.rmtree(tmp, ignore_errors=True)
            with tarfile.open(tar) as t:
                t.extractall(tmp)
            shutil.move(os.path.join(tmp, s, f"{s}_gt.txt"), os.path.join(frames, "processed_gt", f"{s}_gt.txt"))
            os.makedirs(dst, exist_ok=True)
            for item in os.listdir(os.path.join(tmp, s)):
                shutil.rmtree(os.path.join(dst, item), ignore_errors=True)
                shutil.move(os.path.join(tmp, s, item), os.path.join(dst, item))
            shutil.rmtree(tmp)
            open(os.path.join(dst, ".done"), "w").close()
            if not keep_zips:
                os.remove(tar)
        if imu_source == "bags":
            _vbr_imu_from_bags(s, dataset_cfg("vbr")["sequences"][s], keep_bags=keep_zips)
    if imu_source == "archive":
        _vbr_imu_from_archive(seqs)


DATASETS = {"euroc": euroc, "kitti": kitti, "oxford_spires": oxford_spires, "scarf": scarf, "vbr": vbr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--sequences", default=None, help="comma-separated (default: every sequence Xtream evaluates)")
    ap.add_argument("--keep_zips", action="store_true", help="keep the downloaded archives (and VBR bags)")
    ap.add_argument("--vbr_imu", choices=["auto", "bags", "archive"], default="auto",
                    help="VBR's IMU: the official bags (FTP), or the archive extracted from them; auto: the bags, the "
                         "archive only when the official server does not answer")
    a = ap.parse_args()
    known = list(dataset_cfg(a.dataset)["sequences"])
    seqs = a.sequences.split(",") if a.sequences else known
    bad = [s for s in seqs if s not in known]
    assert not bad, f"unknown {a.dataset} sequences {bad}; known: {known}"
    if a.dataset == "vbr":
        src = a.vbr_imu
        if src == "auto":
            src = "bags" if _vbr_server_up() else "archive"
            if src == "archive":
                print(f"[vbr] the official server ({_VBR_FTP[0]}) does not answer: IMU and calibration from the "
                      f"archive extracted from its bags")
        assert src == "bags" or _VBR_IMU_ARCHIVE, "no VBR IMU archive: set XTREAM_VBR_IMU or use --vbr_imu bags"
        vbr(seqs, a.keep_zips, src)
    else:
        DATASETS[a.dataset](seqs, a.keep_zips)
    print("done")


if __name__ == "__main__":
    main()
