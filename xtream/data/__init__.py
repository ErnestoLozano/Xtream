"""Dataset loaders: `load(dataset, sequence)` -> `Sequence` (see `base.py`; layouts in `README.md`)."""

import importlib

DATASETS = ("euroc", "kitti", "oxford_spires", "scarf", "vbr", "custom")


def load(dataset, name):
    assert dataset in DATASETS, dataset
    return importlib.import_module(f"xtream.data.{dataset}").load(name)


def sequences(dataset):
    if dataset == "custom":
        from xtream.data.custom import sequence_names
        return sequence_names()
    from xtream.paths import dataset_cfg
    return list(dataset_cfg(dataset)["sequences"])
