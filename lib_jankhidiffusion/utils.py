"""Time-window, block-list and rescaling helpers for the RAUNet patches.

Ported from `py/utils.py` of blepping's ComfyUI node pack, with the ComfyUI imports
swapped for their Forge Neo equivalents:

| ComfyUI                            | Forge Neo                                     |
|------------------------------------|-----------------------------------------------|
| `comfy.utils.bislerp`              | `backend.misc.image_resize.bislerp`           |
| `model_sampling.percent_to_sigma`  | `unet.model.predictor.percent_to_sigma`       |
| `model_sampling.timestep`          | `unet.model.predictor.timestep`               |

There is no object called `model_sampling` in Neo.  The thing carrying the noise
schedule is the *predictor* (`backend/modules/k_prediction.py`), reachable as
`unet.model.predictor` or `unet.get_model_object("predictor")`, and it exposes the same
`percent_to_sigma` / `timestep` pair the ComfyUI code calls.  Everything here takes that
object as `predictor` and never touches a model directly, so the module imports cleanly
under the offline harness in `tests/`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

import torch
import torch.nn.functional as torchf

try:  # pragma: no cover - exercised only inside the webui
    from backend.misc.image_resize import bislerp
except ImportError:  # pragma: no cover - the offline harness has no backend
    bislerp = None


UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "nearest", "area")
DOWNSCALE_METHODS = ("adaptive_avg_pool2d", "avg_pool2d", *UPSCALE_METHODS)
TWO_STAGE_METHODS = ("disabled", *UPSCALE_METHODS)


class TimeMode(str, Enum):
    PERCENT = "percent"
    TIMESTEP = "timestep"
    SIGMA = "sigma"

    def __str__(self) -> str:
        return str(self.value)


#   `percent_to_sigma(0.0)` answers 999999999.9 on the epsilon predictors but 1.0 on the
#   flow-matching ones, which is a live sigma — a window starting at 0% would drop its
#   first step there.  "From the very beginning" is not a schedule lookup, so say it
#   outright rather than routing it through the model.
NO_GATE_START = float("inf")


def parse_blocks(name: str, val: str | Sequence[int]) -> set[tuple[str, int]]:
    """`"3, 6"` (or `[3, 6]` out of YAML) -> `{("input", 3), ("input", 6)}`."""

    if isinstance(val, (tuple, list, set)):
        if not all(isinstance(item, int) and item >= 0 for item in val):
            raise ValueError("blocks must be a comma-separated string or a sequence of non-negative ints")
        return {(name, int(item)) for item in val}
    if isinstance(val, int):
        return {(name, val)}
    if val is None:
        return set()
    return {(name, int(part)) for part in (raw.strip() for raw in str(val).split(",")) if part}


def convert_time(predictor, time_mode: TimeMode | str, start_time: float, end_time: float) -> tuple[float, float]:
    """A (start, end) pair in the chosen unit -> the (start_sigma, end_sigma) window."""

    time_mode = TimeMode(str(time_mode))

    if time_mode == TimeMode.SIGMA:
        return (float(start_time), float(end_time))

    if time_mode == TimeMode.TIMESTEP:
        start_time = 1.0 - (start_time / 999.0)
        end_time = 1.0 - (end_time / 999.0)
    else:
        if not (0.0 <= start_time <= 1.0):
            raise ValueError("invalid value for start percent")
        if not (0.0 <= end_time <= 1.0):
            raise ValueError("invalid value for end percent")

    return (percent_to_sigma(predictor, start_time), percent_to_sigma(predictor, end_time))


def percent_to_sigma(predictor, percent: float) -> float:
    if percent <= 0.0:
        return NO_GATE_START
    if predictor is None:
        return NO_GATE_START if percent <= 0.0 else 0.0
    return round(float(predictor.percent_to_sigma(float(percent))), 4)


def get_sigma(options, key: str = "sigmas") -> float | None:
    """The largest sigma in a `transformer_options` dict, or `None` when there is none.

    Forge Neo writes the current timestep into `transformer_options["sigmas"]` in
    `backend/sampling/sampling_function.py:254`, the same key ComfyUI uses, so the
    upstream reader works unchanged.
    """

    if isinstance(options, (int, float)):
        return float(options)
    if not isinstance(options, dict):
        return None
    sigmas = options.get(key)
    if sigmas is None:
        return None
    if isinstance(sigmas, (int, float)):
        return float(sigmas)
    return float(sigmas.detach().cpu().max().item())


def check_time(time_arg, start_sigma: float, end_sigma: float) -> bool:
    sigma = get_sigma(time_arg)
    if sigma is None:
        return False
    return start_sigma >= sigma >= end_sigma


def sigma_to_pct(predictor, sigma) -> float:
    """Inverse of `percent_to_sigma`, used to place a sigma inside the fadeout ramp.

    The two sentinel sigmas are short-circuited rather than looked up: `NO_GATE_START` is
    not a point on any schedule, and `predictor.timestep()` would take its logarithm.
    """

    if predictor is None:
        return 0.0
    if not torch.is_tensor(sigma):
        sigma = float(sigma)
        if sigma == float("inf"):
            return 0.0
        if sigma <= 0.0:
            return 1.0
        sigma = torch.tensor(sigma, dtype=torch.float32)
    return (1.0 - (predictor.timestep(sigma).detach().cpu().float() / 999.0)).clamp(0.0, 1.0).max().item()


def fade_scale(pct: float, start_pct: float = 0.0, end_pct: float = 1.0, fade_start: float = 1.0, fade_cap: float = 0.0) -> float:
    """1.0 inside the window, ramping down to `fade_cap` between `fade_start` and `end_pct`."""

    if start_pct > end_pct or not (start_pct <= pct <= end_pct):
        return 0.0
    if pct < fade_start:
        return 1.0
    if end_pct <= fade_start:
        return max(fade_cap, 0.0)
    return max(fade_cap, 1.0 - ((pct - fade_start) / (end_pct - fade_start)))


def scale_samples(samples: torch.Tensor, width: int, height: int, mode: str = "bicubic") -> torch.Tensor:
    if mode == "bislerp":
        if bislerp is None:  # pragma: no cover - harness fallback
            mode = "bicubic"
        else:
            return bislerp(samples, width, height)
    if mode in ("bicubic", "bilinear"):
        return torchf.interpolate(samples, size=(height, width), mode=mode, align_corners=False)
    return torchf.interpolate(samples, size=(height, width), mode=mode)


def latent_to_megapixels(shape: Sequence[int], vae_scale: int = 8) -> float:
    """Latent shape -> the megapixel count of the image it decodes to.

    Keyed off the trailing two dims so a 5D `(B, C, T, H, W)` latent measures the frame
    rather than the frame count — RAUNet only ever runs on 4D UNet latents today, but the
    rule costs nothing and is the one that survives a Wan-format model later.
    """

    if shape is None or len(shape) < 2:
        return 0.0
    return (shape[-1] * vae_scale) * (shape[-2] * vae_scale) / 1_000_000.0


def rescale_size(width: int, height: int, target_res: int, *, tolerance: int = 1) -> tuple[int, int]:
    """Naive factorisation of `target_res` into a width/height near the given aspect."""

    tolerance = min(target_res, tolerance)

    def neighbours(num: float):
        if num < 1:
            return ()
        numi = int(num)
        span = range(-min(numi - 1, tolerance), tolerance + 1 + math.ceil(num - numi))
        return tuple(numi + adj for adj in sorted(span, key=abs))

    scale = math.sqrt(height * width / target_res)
    for h, w in zip(neighbours(height / scale), neighbours(width / scale), strict=False):
        h_adj = target_res / w
        if h_adj % 1 == 0:
            return (w, int(h_adj))
        w_adj = target_res / h
        if w_adj % 1 == 0:
            return (int(w_adj), h)
    raise ValueError(f"can't rescale {width} and {height} to fit {target_res}")


__all__ = (
    "DOWNSCALE_METHODS",
    "NO_GATE_START",
    "TWO_STAGE_METHODS",
    "UPSCALE_METHODS",
    "TimeMode",
    "check_time",
    "convert_time",
    "fade_scale",
    "get_sigma",
    "latent_to_megapixels",
    "parse_blocks",
    "percent_to_sigma",
    "rescale_size",
    "scale_samples",
    "sigma_to_pct",
)
