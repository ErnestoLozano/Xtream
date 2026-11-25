"""Fetch the model weights and the external code Xtream uses.

- CLoSeR's loop detector (SALAD retrieval on DINOv2): the files Xtream imports, from the upstream repository at a
  fixed commit, into third_party/CLoSeR (not redistributed with Xtream: CLoSeR states no license, SALAD is GPL-3.0).
- SALAD (loop and covisibility retrieval): `dino_salad.ckpt` from the SALAD release into ckpt/.
- Depth Anything 3: DA3-GIANT-1.1 (both modes) and DA3METRIC-LARGE (monocular), into the Hugging Face cache
  (they also download on first use).
- GeoCalib's weights download on the first mono-inertial run (torch hub cache).

    python scripts/download_checkpoints.py [--skip_da3]
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xtream.paths import CLOSER_FILES, CLOSER_ROOT, CLOSER_URL, SALAD_CKPT, SALAD_URL  # noqa: E402

DA3_MODELS = ("depth-anything/DA3-GIANT-1.1", "depth-anything/DA3METRIC-LARGE")


def fetch_closer():
    """Extract CLOSER_FILES (files, or folders ending in '/') of CLoSeR's archive at the pinned commit into
    CLOSER_ROOT."""
    print(f"CLoSeR loop detector -> {CLOSER_ROOT}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "closer.tar.gz")
        urllib.request.urlretrieve(CLOSER_URL, arc)
        out = os.path.join(tmp, "out")
        with tarfile.open(arc) as tf:
            n = 0
            for m in tf.getmembers():
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                if not m.isfile() or not any(rel == f or (f.endswith("/") and rel.startswith(f))
                                             for f in CLOSER_FILES):
                    continue
                dst = os.path.join(out, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tf.extractfile(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
                n += 1
        assert os.path.exists(os.path.join(out, "loger", "utils", "loop", "detector.py")), "unexpected archive"
        for d in ("loger", "loger/models", "loger/utils"):      # regular packages: no other `loger` shadows them
            open(os.path.join(out, d, "__init__.py"), "a").close()
        if os.path.exists(CLOSER_ROOT):
            shutil.rmtree(CLOSER_ROOT)
        shutil.move(out, CLOSER_ROOT)
    print(f"  {n} files", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_da3", action="store_true", help="skip the DA3 models (they download on first use)")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(CLOSER_ROOT, "loger", "utils", "loop", "detector.py")):
        fetch_closer()
    if not os.path.exists(SALAD_CKPT):
        os.makedirs(os.path.dirname(SALAD_CKPT), exist_ok=True)
        print(f"SALAD -> {SALAD_CKPT}", flush=True)
        urllib.request.urlretrieve(SALAD_URL, SALAD_CKPT + ".part")
        os.replace(SALAD_CKPT + ".part", SALAD_CKPT)
    if not a.skip_da3:
        from huggingface_hub import snapshot_download
        for m in DA3_MODELS:
            print(f"{m} -> Hugging Face cache", flush=True)
            snapshot_download(m)
    print("done")


if __name__ == "__main__":
    main()
