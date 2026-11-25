"""Repository paths, dataset roots and checkpoints, in one place.

Dataset roots come from `configs/datasets.yaml`, relative to the data root `XTREAM_DATA` (default `data/` in the
repository) unless absolute. An optional `configs/datasets.local.yaml` (not versioned; or the file
`XTREAM_DATASETS_LOCAL` names) overrides any of its keys, for data that already sits elsewhere on disk.
"""

import functools
import os

import yaml

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party")
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
RESULTS_ROOT = os.environ.get("XTREAM_RESULTS", os.path.join(REPO_ROOT, "results"))
DATA_ROOT = os.environ.get("XTREAM_DATA", os.path.join(REPO_ROOT, "data"))

# vendored code (see third_party/*/VENDORED.md)
DA3_SRC_ROOT = os.path.join(THIRD_PARTY, "Depth-Anything-3", "src")
GEOCALIB_ROOT = os.path.join(THIRD_PARTY, "GeoCalib")

# CLoSeR's loop detector (SALAD on DINOv2), not redistributed: scripts/download_checkpoints.py fetches the files
# Xtream imports from the upstream repository at this commit into CLOSER_ROOT
CLOSER_ROOT = os.path.join(THIRD_PARTY, "CLoSeR")
CLOSER_COMMIT = "5230776a606e67593ec58e41185e2eca92c865bd"
CLOSER_URL = f"https://github.com/MoyangLi00/CLoSeR/archive/{CLOSER_COMMIT}.tar.gz"
CLOSER_FILES = ("loger/utils/loop/", "loger/models/dinov2/", "loger/models/layers/attention.py",
                "loger/models/layers/prope.py")

# checkpoints: the DA3 models download from the Hugging Face hub into its cache; SALAD's file is fetched by
# scripts/download_checkpoints.py into CKPT_DIR (or set XTREAM_SALAD_CKPT)
CKPT_DIR = os.path.join(REPO_ROOT, "ckpt")
SALAD_CKPT = os.environ.get("XTREAM_SALAD_CKPT", os.path.join(CKPT_DIR, "dino_salad.ckpt"))
SALAD_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"


@functools.lru_cache(None)
def datasets_cfg():
    """`configs/datasets.yaml`, updated per dataset by `configs/datasets.local.yaml` when it exists."""
    cfg = yaml.safe_load(open(os.path.join(CONFIG_DIR, "datasets.yaml")))
    local = os.environ.get("XTREAM_DATASETS_LOCAL", os.path.join(CONFIG_DIR, "datasets.local.yaml"))
    if os.path.exists(local):
        for name, over in (yaml.safe_load(open(local)) or {}).items():
            cfg[name] = dict(cfg.get(name, {}), **over)
    return cfg


def dataset_cfg(name):
    return datasets_cfg()[name]


def data_dir(name, key="raw"):
    """Absolute path of dataset `name`'s folder `key` (raw, prepared, ...)."""
    return os.path.abspath(os.path.join(DATA_ROOT, dataset_cfg(name)[key]))
