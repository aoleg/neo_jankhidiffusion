"""RAUNet (from HiDiffusion) for Forge Neo's UNet.

RAUNet has two halves, applied over separate sigma windows:

**The Upsample/Downsample rewrite** ("main" window).  One `Downsample` block on the way
down has its stride-2, dilation-1, padding-1 convolution run as stride-4, dilation-2,
padding-2 instead, halving the feature-map resolution a second time; the mirrored
`Upsample` block on the way up compensates with a 4x interpolation before its conv.
Between them the deep layers of the UNet see feature maps the size they were trained on,
which is what stops the duplicated subjects and mush you get from generating well above
a model's native resolution.

**The cross-attention rescale** ("CA" window).  A separate, shallower pair of blocks has
its hidden state pooled down before attention and scaled back up afterwards, via the
`input_block_patch` / `output_block_patch` hooks.

### Porting notes (ComfyUI -> Forge Neo)

Neo's UNet is its own implementation (`backend/nn/unet.py`,
`IntegratedUNet2DConditionModel`) rather than ComfyUI's `openaimodel`, but it kept the
same extension surface, so the port is close to mechanical:

| ComfyUI                                       | Forge Neo                                |
|-----------------------------------------------|------------------------------------------|
| `transformer_patches["input_block_patch"]`     | same (`backend/nn/unet.py:678`)          |
| `transformer_patches["output_block_patch"]`    | same (`:698`)                            |
| `transformer_options["sigmas"]`                | same (`sampling_function.py:254`)        |
| `ModelPatcher.add_object_patch`                | same (`backend/patcher/base.py:342`)     |
| `openaimodel.ops.conv_nd(...)` temp conv       | **replaced**, see `_strided_conv` below  |

The one real divergence is the downsample convolution.  Upstream builds a fresh
`comfy.ops` conv with the wider stride and copies the original layer's weight handles
onto it.  Neo has no equivalent `ops` handle on the block, and its conv classes carry
non-trivial state (`parameters_manual_cast`, GGUF/bnb/fp8 dequant paths, weight/bias
functions).  Rebuilding that faithfully would mean re-implementing five operation
classes.  Instead we borrow the layer itself and swap only `stride`/`padding`/`dilation`
for the duration of the call: `Conv2d.forward` resolves its weights and then hands them
to `self._conv_forward`, which reads exactly those three attributes
(`backend/operations.py:216-228`), so every quantisation path keeps working for free.

### State

There is no module-level state.  A `Config` is built per generation and captured by the
closures and `HDForward` objects installed on one cloned `UnetPatcher`; Forge rebuilds
`forge_objects` from `forge_objects_after_applying_lora` before every sampling pass
(`modules/processing.py:1376`), so a disabled generation is simply one that never
installs anything.  This is the main structural fix over the older reForge port, which
kept a global `HDCONFIG` singleton and monkey-patched `openaimodel.Upsample` /
`Downsample` at import time for the whole process.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .utils import (
    NO_GATE_START,
    TimeMode,
    check_time,
    convert_time,
    fade_scale,
    get_sigma,
    latent_to_megapixels,
    parse_blocks,
    scale_samples,
    sigma_to_pct,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """One generation's worth of RAUNet settings, plus the live per-forward state."""

    #   main (Upsample/Downsample) window
    start_sigma: float = NO_GATE_START
    end_sigma: float = 0.0
    use_blocks: set = field(default_factory=set)
    upscale_mode: str = "bicubic"
    two_stage_upscale_mode: str = "disabled"

    #   cross-attention window
    ca_start_sigma: float = NO_GATE_START
    ca_end_sigma: float = 0.0
    ca_use_blocks: set = field(default_factory=set)
    ca_upscale_mode: str = "bicubic"
    ca_downscale_mode: str = "adaptive_avg_pool2d"
    ca_downscale_factor: float = 2.0
    ca_downscale_factor_w: float | None = None
    ca_input_after_skip_mode: bool = False
    ca_avg_pool2d_ceil_mode: bool = True
    ca_latent_pixel_increment: int = 8

    #   fadeout: taper the CA downscale to `ca_fadeout_cap` instead of cutting it off
    ca_fadeout_start_sigma: float | None = None
    ca_fadeout_cap: float = 0.0
    ca_start_pct: float = 0.0
    ca_end_pct: float = 1.0
    ca_fadeout_start_pct: float = 1.0

    #   multipliers on the tensors going in and out of each rescale (advanced/YAML only)
    pre_upscale_multiplier: float = 1.0
    post_upscale_multiplier: float = 1.0
    pre_downscale_multiplier: float = 1.0
    post_downscale_multiplier: float = 1.0
    ca_pre_upscale_multiplier: float = 1.0
    ca_post_upscale_multiplier: float = 1.0
    ca_pre_downscale_multiplier: float = 1.0
    ca_post_downscale_multiplier: float = 1.0

    #   Forge-specific: below this image size the whole effect is skipped.  See
    #   `_note_shape` for why this is a runtime check and not just a UI-level one.
    min_megapixels: float = 0.0

    #   sigma is read from the input_block_patch and stashed here because Neo's
    #   Upsample/Downsample forwards take no transformer_options - same trick upstream uses
    curr_sigma: float | None = None
    curr_megapixels: float = 0.0
    gated_out: bool = False

    verbose: bool = False
    predictor: object = None

    #   diagnostics, read by the script for its one-shot log line
    hits_main: int = 0
    hits_ca: int = 0
    skips_resolution: int = 0

    @classmethod
    def build(
        cls,
        predictor,
        *,
        input_blocks="",
        output_blocks="",
        ca_input_blocks="",
        ca_output_blocks="",
        time_mode: str | TimeMode = TimeMode.PERCENT,
        start_time: float = 0.0,
        end_time: float = 0.45,
        ca_start_time: float = 0.0,
        ca_end_time: float = 0.3,
        ca_fadeout_start_time: float | None = None,
        **kwargs,
    ) -> Config:
        start_sigma, end_sigma = convert_time(predictor, time_mode, start_time, end_time)
        ca_start_sigma, ca_end_sigma = convert_time(predictor, time_mode, ca_start_time, ca_end_time)

        blocks = itertools.starmap(parse_blocks, (("input", input_blocks), ("output", output_blocks)))
        ca_blocks = itertools.starmap(parse_blocks, (("input", ca_input_blocks), ("output", ca_output_blocks)))

        config = cls(
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            ca_start_sigma=ca_start_sigma,
            ca_end_sigma=ca_end_sigma,
            use_blocks=set().union(*blocks),
            ca_use_blocks=set().union(*ca_blocks),
            predictor=predictor,
            **kwargs,
        )

        if ca_fadeout_start_time is not None and config.ca_use_blocks:
            config.ca_fadeout_start_sigma = convert_time(predictor, time_mode, ca_fadeout_start_time, ca_fadeout_start_time)[0]
            config.ca_start_pct = sigma_to_pct(predictor, config.ca_start_sigma)
            config.ca_end_pct = sigma_to_pct(predictor, config.ca_end_sigma)
            config.ca_fadeout_start_pct = sigma_to_pct(predictor, config.ca_fadeout_start_sigma)

        if config.ca_downscale_mode == "avg_pool2d":
            factors = (config.ca_downscale_factor, config.ca_downscale_factor_w)
            if any(f is not None and not float(f).is_integer() for f in factors):
                raise ValueError("the avg_pool2d downscale mode only accepts whole-number downscale factors; use adaptive_avg_pool2d for fractional ones")

        return config

    # ---------------------------------------------------------------- live state ----

    def note_shape(self, extra_options) -> None:
        """Record sigma and latent size for this forward pass.

        Called from the input_block_patch on block 0, which Neo runs for every forward
        (`backend/nn/unet.py:678`) whether or not block 0 is a CA target — the patch is
        registered unconditionally for exactly this reason.
        """

        self.curr_sigma = get_sigma(extra_options)
        self.curr_megapixels = latent_to_megapixels(extra_options.get("original_shape"))
        self.gated_out = 0.0 < self.curr_megapixels < self.min_megapixels
        if self.gated_out:
            self.skips_resolution += 1

    def check(self, topts, *, ca: bool = False) -> bool:
        if self.gated_out:
            return False
        start_sigma, end_sigma, use_blocks = (self.ca_start_sigma, self.ca_end_sigma, self.ca_use_blocks) if ca else (self.start_sigma, self.end_sigma, self.use_blocks)
        if not use_blocks:
            return False
        if not isinstance(topts, dict) or topts.get("block") not in use_blocks:
            return False
        return check_time(topts, start_sigma, end_sigma)

    def current_downscale_factors(self, extra_options) -> tuple[float, float]:
        """The CA downscale factors for this step, after the fadeout ramp."""

        factor_h = self.ca_downscale_factor
        factor_w = self.ca_downscale_factor if self.ca_downscale_factor_w is None else self.ca_downscale_factor_w

        if self.ca_fadeout_start_sigma is None:
            return factor_h, factor_w

        pct = sigma_to_pct(self.predictor, extra_options.get("sigmas", self.curr_sigma))
        scale = fade_scale(pct, self.ca_start_pct, self.ca_end_pct, self.ca_fadeout_start_pct, self.ca_fadeout_cap)
        if scale >= 1.0:
            return factor_h, factor_w
        if scale <= 0.0:
            return 1.0, 1.0
        #   ramp each factor towards 1.0 (= no rescale) rather than towards 0
        return (
            factor_h - (factor_h - 1.0) * (1.0 - scale),
            factor_w - (factor_w - 1.0) * (1.0 - scale),
        )

    @staticmethod
    def maybe_multiply(t: torch.Tensor, multiplier: float = 1.0, *, post: bool = False) -> torch.Tensor:
        if multiplier == 1.0:
            return t
        return t.mul_(multiplier) if post else t * multiplier


class HDForward:
    """Replacement `forward` for one `Upsample` or `Downsample` block.

    Installed with `add_object_patch("....forward", self)` rather than by subclassing, so
    nothing is monkey-patched globally and Forge's own unpatch path
    (`ModelPatcher.unpatch_model`) removes it when the next generation loads a patcher
    that does not ask for it.  `nn.Module.__call__` looks `forward` up on the instance, so
    a plain callable object slots straight in.
    """

    def __init__(self, orig_block, config: Config, block_index: int, is_up: bool):
        self.orig_block = orig_block
        #   Re-patching a model whose previous patcher is still alive can hand us a
        #   HDForward as the "original"; unwrap to the real bound method.
        orig_forward = orig_block.forward
        while isinstance(orig_forward, HDForward):
            orig_forward = orig_forward.orig_forward
        self.orig_forward = orig_forward
        self.config = config
        self.block_index = block_index
        self.is_up = is_up

    def __call__(self, *args, **kwargs):
        return self.forward_upsample(*args, **kwargs) if self.is_up else self.forward_downsample(*args, **kwargs)

    def _active(self, block_type: str) -> bool:
        block = self.orig_block
        if getattr(block, "dims", 2) == 3 or not getattr(block, "use_conv", False):
            return False
        return self.config.check({"sigmas": self.config.curr_sigma, "block": (block_type, self.block_index)})

    def forward_upsample(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        config = self.config
        if not self._active("output"):
            return self.orig_forward(x, output_shape=output_shape)

        #   4x, not the block's usual 2x: this undoes the extra halving the paired
        #   Downsample did on the way in.
        shape = output_shape[2:4] if output_shape is not None else (x.shape[-2] * 4, x.shape[-1] * 4)

        x = config.maybe_multiply(x, config.pre_upscale_multiplier)
        if config.two_stage_upscale_mode != "disabled":
            x = scale_samples(x, shape[1] // 2, shape[0] // 2, mode=config.two_stage_upscale_mode)
        x = scale_samples(x, shape[1], shape[0], mode=config.upscale_mode)
        config.hits_main += 1
        return config.maybe_multiply(self.orig_block.conv(x), config.post_upscale_multiplier, post=True)

    def forward_downsample(self, x: torch.Tensor) -> torch.Tensor:
        config = self.config
        if not self._active("input"):
            return self.orig_forward(x)

        x = config.maybe_multiply(x, config.pre_downscale_multiplier)
        config.hits_main += 1
        out = _strided_conv(self.orig_block.op, x, stride=(4, 4), padding=(2, 2), dilation=(2, 2))
        return config.maybe_multiply(out, config.post_downscale_multiplier, post=True)


def _strided_conv(op, x: torch.Tensor, *, stride, padding, dilation) -> torch.Tensor:
    """Run an existing conv layer with a different stride/padding/dilation.

    `torch.nn.Conv2d._conv_forward` reads `self.stride`, `self.padding`, `self.dilation`
    and `self.groups` at call time, and every one of Neo's conv variants
    (`ForgeOperations`, `ForgeOperationsGGUF`, `ForgeOperationsFP8`) resolves its weights
    and then delegates to it.  Swapping the three attributes around the call therefore
    reuses the layer's entire weight pipeline — manual cast, dequantisation, LoRA weight
    functions — instead of reconstructing it.

    The kernel is untouched (3x3 either way), so the weights are used as-is; only the
    sampling grid widens.  Restoring in a `finally` matters because sampling can be
    interrupted mid-forward.
    """

    saved = (op.stride, op.padding, op.dilation)
    op.stride, op.padding, op.dilation = stride, padding, dilation
    try:
        return op(x)
    finally:
        op.stride, op.padding, op.dilation = saved


def make_input_block_patch(config: Config):
    """`input_block_patch`: stash per-forward state, then pool down the CA blocks."""

    def input_block_patch(h: torch.Tensor, extra_options: dict) -> torch.Tensor:
        block_type, block_index = extra_options.get("block", ("unknown", -1))
        if block_type == "input" and block_index == 0:
            config.note_shape(extra_options)

        if not config.check(extra_options, ca=True):
            return h

        factor_h, factor_w = config.current_downscale_factors(extra_options)
        if factor_h == 1.0 and factor_w == 1.0:
            return h

        height, width = h.shape[-2:]
        increment = max(1, config.ca_latent_pixel_increment)
        target_h = int(max(increment, ((height / increment) // factor_h) * increment))
        target_w = int(max(increment, ((width / increment) // factor_w) * increment))
        #   don't overshoot when downscaling, don't undershoot when upscaling
        target_h = min(height, target_h) if factor_h >= 1 else max(height, target_h)
        target_w = min(width, target_w) if factor_w >= 1 else max(width, target_w)
        if (target_h, target_w) == (height, width):
            return h

        config.hits_ca += 1
        h = config.maybe_multiply(h, config.ca_pre_downscale_multiplier)

        if config.ca_downscale_mode == "avg_pool2d":
            result = F.avg_pool2d(
                h,
                kernel_size=(max(1, int(height // target_h)), max(1, int(width // target_w))),
                ceil_mode=config.ca_avg_pool2d_ceil_mode,
            )
        elif config.ca_downscale_mode == "adaptive_avg_pool2d":
            result = F.adaptive_avg_pool2d(h, (target_h, target_w))
        else:
            result = scale_samples(h, target_w, target_h, mode=config.ca_downscale_mode)

        return config.maybe_multiply(result, config.ca_post_downscale_multiplier, post=True)

    return input_block_patch


def make_output_block_patch(config: Config):
    """`output_block_patch`: scale the hidden state back up to meet its skip connection."""

    def output_block_patch(h: torch.Tensor, hsp: torch.Tensor, extra_options: dict):
        if not config.check(extra_options, ca=True) or h.shape[-2:] == hsp.shape[-2:]:
            return h, hsp
        config.hits_ca += 1
        h = config.maybe_multiply(h, config.ca_pre_upscale_multiplier)
        h = scale_samples(h, hsp.shape[-1], hsp.shape[-2], mode=config.ca_upscale_mode)
        return config.maybe_multiply(h, config.ca_post_upscale_multiplier, post=True), hsp

    return output_block_patch


def apply_raunet(unet, config: Config, unet_map=None):
    """Install RAUNet on a **cloned** `UnetPatcher`. Returns the same patcher.

    The caller owns the clone (`p.sd_model.forge_objects.unet.clone()`), matching how
    every other Forge extension composes: patch the clone, assign it back.
    """

    if unet_map is not None:
        problems = unet_map.validate(config.use_blocks)
        if problems:
            raise ValueError("; ".join(problems))

    #   Registered even with no CA blocks configured: this is also what feeds
    #   `config.curr_sigma` and the resolution gate to the Upsample/Downsample forwards.
    if config.ca_input_after_skip_mode:
        unet.set_model_input_block_patch_after_skip(make_input_block_patch(config))
    else:
        unet.set_model_input_block_patch(make_input_block_patch(config))

    if any(block_type == "output" for block_type, _ in config.ca_use_blocks):
        unet.set_model_output_block_patch(make_output_block_patch(config))

    for block_type, block_index in sorted(config.use_blocks):
        container_name = f"diffusion_model.{block_type}_blocks.{block_index}"
        container = unet.get_model_object(container_name)
        block_name = f"{container_name}.{len(container) - 1}"
        block = unet.get_model_object(block_name)

        expected = "Downsample" if block_type == "input" else "Upsample"
        if type(block).__name__ != expected:
            raise ValueError(f"{block_type} block {block_index} must end in an {expected} layer, but this model has {type(block).__name__} there")

        unet.add_object_patch(f"{block_name}.forward", HDForward(block, config, block_index, is_up=block_type != "input"))

    if config.verbose:
        logger.info(f"RAUNet: {config}")

    return unet


__all__ = ("Config", "HDForward", "apply_raunet", "make_input_block_patch", "make_output_block_patch")
