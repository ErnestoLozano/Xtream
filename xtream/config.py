"""The two modes' configurations (`configs/mono.yaml`, `configs/mono_inertial.yaml`)."""

import os

import yaml

from xtream.paths import CONFIG_DIR

MODES = {"mono": "mono.yaml", "mono-inertial": "mono_inertial.yaml"}


def load_config(mode_or_path):
    """A mode's configuration, by name (`mono`, `mono-inertial`) or path to a YAML file."""
    path = os.path.join(CONFIG_DIR, MODES[mode_or_path]) if mode_or_path in MODES else mode_or_path
    cfg = yaml.safe_load(open(path))
    assert cfg["mode"] in MODES, cfg["mode"]
    return cfg
