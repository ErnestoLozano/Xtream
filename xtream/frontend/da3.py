"""Depth Anything 3 as the front end runs it: one model for every multi-view pass, frames wider than `crop_above`
centre-cropped to `canon_aspect` (DA3's widest training aspect, 16:9), and the next window's images
preprocessed on the CPU while the GPU runs the current pass."""

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from xtream.paths import DA3_SRC_ROOT


def load_da3(name, device="cuda"):
    """A DepthAnything3 model from the Hugging Face hub, in eval mode, without its visualisation copy of the
    images."""
    if DA3_SRC_ROOT not in sys.path:
        sys.path.insert(0, DA3_SRC_ROOT)
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(name).to(device).eval()
    model._add_processed_images = lambda pred, imgs: pred
    return model


def crop_to_aspect(models, aspect):
    """Every input image wider than `aspect` (W/H) is centre-cropped to it (focal kept, W narrowed)."""
    def canon(load):
        def f(img):
            im = load(img)
            w, h = im.size
            if w / h > aspect + 1e-6:
                c = int(round(h * aspect))
                im = im.crop(((w - c) // 2, 0, (w - c) // 2 + c, h))
            return im
        return f
    for ip in {id(m.input_processor): m.input_processor for m in models}.values():
        ip._load_image = canon(ip._load_image)


class Prefetch:
    """The model's unconditioned preprocessing of the next frames, run in a worker thread (outputs unchanged)."""

    def __init__(self, model, paths):
        self.paths, self.pre, self.pool = paths, {}, ThreadPoolExecutor(1)
        self._pre = model._preprocess_inputs

        def prep(image, extrinsics=None, intrinsics=None, process_res=504, process_res_method="upper_bound_resize"):
            fut = (self.pre.pop((tuple(image), process_res, process_res_method), None)
                   if extrinsics is None and intrinsics is None else None)
            return fut.result() if fut is not None else self._pre(image, extrinsics, intrinsics, process_res,
                                                                   process_res_method)
        model._preprocess_inputs = prep

    def __call__(self, frames):
        im = [self.paths[f] for f in frames]
        self.pre[(tuple(im), 504, "upper_bound_resize")] = self.pool.submit(self._pre, im)


def camera_to_world(model, paths, frames):
    """One unconditioned pass over `frames`: (camera-to-world poses (N, 4, 4), OpenCV axes, the pass's unit;
    the prediction)."""
    import torch
    with torch.no_grad():
        pred = model.inference([paths[f] for f in frames])
    w2c = np.tile(np.eye(4), (len(frames), 1, 1))
    w2c[:, :3, :] = pred.extrinsics[:, :3, :]
    return np.linalg.inv(w2c), pred
