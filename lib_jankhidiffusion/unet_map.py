"""Which blocks of *this* model RAUNet is allowed to touch, discovered at runtime.

RAUNet is not a generic effect: it rewrites the strided convolution inside a UNet
`Downsample` and the interpolation inside the matching `Upsample`, so it only exists for
architectures that *have* those.  Rather than hard-coding "3/8 for SD1.5, 3/5 for SDXL"
and hoping the loaded checkpoint agrees, this module walks
`unet.model.diffusion_model` and reports what is actually there.

That buys three things:

  * the UI can show the valid block numbers for the checkpoint that is loaded;
  * a wrong block number fails with a readable message instead of an `AttributeError`
    ten frames deep;
  * **a model with no UNet at all is detected and refused cleanly.**  This is the path
    every non-UNet checkpoint takes today — Flux, Qwen, Chroma, and notably **Krea 2**,
    whose `backend/nn/krea.py` is a `SingleStreamDiT`: a flat stack of single-stream
    transformer blocks with no resolution pyramid, no `Downsample`, and no
    `input_block_patch` hook.  RAUNet as an algorithm has nothing to attach to there.

### Extending to another family later

Adding a family is meant to be additive:

  1. teach `inspect_unet` how to enumerate that architecture's scaling blocks
     (or add a sibling inspector and pick between them in `describe`);
  2. add a `Preset` row to `PRESETS` for its native resolution and block pairing;
  3. add its name to `ModelFamily`.

`raunet.py` never names a family — it consumes a `UNetMap` and a `Preset` — so an
architecture that genuinely has a down/up pyramid should need no changes there.  A
DiT-shaped model like Krea 2 would need a different algorithm rather than a new preset,
and `supported=False` is the honest answer until someone writes one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#   Blocks are recognised by class *name* rather than by importing
#   `backend.nn.unet.Downsample`.  It keeps this module importable with no webui present
#   (the harness in `tests/` relies on that) and it does not care which of Neo's
#   operation classes the checkpoint's conv layers ended up being built from.


class ModelFamily(str, Enum):
    SD15 = "SD15"
    SDXL = "SDXL"

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class UNetMap:
    """What `inspect_unet` found. `supported=False` means "do not patch this model"."""

    supported: bool
    reason: str = ""
    n_input_blocks: int = 0
    n_output_blocks: int = 0
    #   input blocks whose last layer is a Downsample / output blocks ending in an Upsample
    downsample_blocks: tuple[int, ...] = ()
    upsample_blocks: tuple[int, ...] = ()
    #   blocks holding a SpatialTransformer, i.e. the ones the cross-attention
    #   (input_block_patch / output_block_patch) half of RAUNet can usefully target
    ca_input_blocks: tuple[int, ...] = ()
    ca_output_blocks: tuple[int, ...] = ()

    def paired_output(self, input_block: int) -> int | None:
        """The Upsample block that mirrors a given Downsample block.

        The UNet skip stack pops in reverse, so block *i* on the way down is undone by
        block `n_output - 1 - i` on the way up: SD1.5 3/8, 6/5, 9/2 and SDXL 3/5, 6/2 all
        fall out of that one rule instead of being three more constants to keep in sync.
        """

        candidate = self.n_output_blocks - 1 - input_block
        return candidate if candidate in self.upsample_blocks else None

    def validate(self, use_blocks, ca_use_blocks=(), *, after_skip: bool = False) -> list[str]:
        """Fatal complaints: configurations that would crash or silently do nothing.

        Everything here is a refusal, not a warning. A broken pairing does not degrade the
        image, it raises a `torch.cat` size error several frames inside someone else's
        forward pass — so it has to stop the run before sampling starts, with a message
        that names the value to change. Advisory-only notes live in `advise_ca`.
        """

        problems = []
        for block_type, index in sorted(use_blocks):
            valid = self.downsample_blocks if block_type == "input" else self.upsample_blocks
            if index not in valid:
                pretty = ", ".join(str(v) for v in valid) or "none"
                problems.append(f"{block_type} block {index} is not a scaling block on this model (valid: {pretty})")

        inputs = {index for kind, index in use_blocks if kind == "input"}
        outputs = {index for kind, index in use_blocks if kind == "output"}
        for index in sorted(inputs):
            partner = self.paired_output(index)
            if partner is not None and partner not in outputs:
                problems.append(f"input block {index} must be paired with output block {partner}, or the extra downscale is never undone")
        for index in sorted(outputs):
            #   the pairing is its own inverse, since a UNet has as many output blocks
            #   as input blocks
            partner = self.n_output_blocks - 1 - index
            if partner not in inputs:
                problems.append(f"output block {index} has no matching input block ({partner}) selected")

        ca_inputs = {index for kind, index in ca_use_blocks if kind == "input"}
        ca_outputs = {index for kind, index in ca_use_blocks if kind == "output"}
        for index in sorted(ca_inputs):
            expected = self.ca_paired_output(index, after_skip=after_skip)
            if not (0 <= expected < self.n_output_blocks) or expected in ca_outputs:
                continue
            got = ", ".join(str(o) for o in sorted(ca_outputs)) or "none"
            mode = " (after-skip mode shifts this pairing in by one)" if after_skip else ""
            problems.append(
                f'CA input block {index} requires CA output block {expected} on this model{mode} - set "CA output blocks" to {expected} (currently: {got}). '
                f"Without it the downscaled hidden state never returns to the size of its skip connection and the UNet fails at torch.cat"
            )
        #   The reverse - a CA output with no CA input - is harmless: the output patch
        #   returns early when h and hsp already match.
        return problems

    def ca_paired_output(self, ca_input_block: int, *, after_skip: bool = False) -> int:
        """The output block that has to undo a cross-attention downscale.

        The input patch runs *before* the hidden state is pushed onto the skip stack, so
        the entry for its own block is already downscaled and matches on the way up; the
        first skip that does *not* match is the one from the block before it, popped one
        step later — hence `n_output - index` rather than `n_output - 1 - index`.
        Switching to after-skip mode moves the patch to the other side of that push, so
        the skip for its own block stays at full resolution and the pairing shifts in by
        one.  Getting this wrong does not degrade the image, it raises a `torch.cat` size
        error deep in the UNet, so it is worth saying out loud.
        """

        return self.n_output_blocks - ca_input_block - (1 if after_skip else 0)

    def advise_ca(self, ca_use_blocks, *, after_skip: bool = False) -> list[str]:  # noqa: ARG002
        """Advisory notes about the cross-attention blocks. Nothing here is fatal.

        A block with no cross-attention still *works* - the patch rescales the hidden
        state on its way through - it is just not what the effect was designed around, and
        the shallow blocks in particular are far more destructive than the deep ones
        (block 2 rescales the latent at full resolution; block 4 rescales it after two
        downsamples). Worth flagging, not worth refusing.
        """

        notes = []
        for block_type, index in sorted(ca_use_blocks):
            limit = self.n_input_blocks if block_type == "input" else self.n_output_blocks
            valid = self.ca_input_blocks if block_type == "input" else self.ca_output_blocks
            if index >= limit:
                notes.append(f"CA {block_type} block {index} does not exist on this model (0-{limit - 1})")
            elif index not in valid:
                pretty = ", ".join(str(v) for v in valid) or "none"
                notes.append(f"CA {block_type} block {index} holds no cross-attention on this model (blocks that do: {pretty}); it will still rescale, but shallow blocks change the image far more than deep ones")
        return notes


@dataclass(frozen=True)
class Preset:
    """Per-family defaults. Times are percentages of the step progression."""

    family: ModelFamily
    native_megapixels: float
    input_blocks: str
    output_blocks: str
    ca_input_blocks: str
    ca_output_blocks: str
    #   res_mode -> (start, end, ca_start, ca_end); an empty tuple means "RAUNet off".
    #   Within a window, `start >= end` means that half is off — that is how upstream
    #   spells "disabled" (its `Preset` defaults every time field to 1.0), and both halves
    #   are switched independently.
    res_modes: dict = field(default_factory=dict)


#   Windows below are blepping's `SIMPLE_PRESETS` (`py/raunet.py`) verbatim, *not* the
#   older reForge script's.  For SDXL the two are close to inverted and it matters:
#
#     reForge  SDXL high -> scaling blocks 0.0-0.5, cross-attention off
#     ComfyUI  SDXL high -> scaling blocks off,     cross-attention 0.0-0.5
#
#   i.e. upstream concluded the Downsample/Upsample rewrite is the part that hurts SDXL
#   at 1536-2048 and left only the gentler cross-attention rescale on, while the fork
#   enables precisely the half upstream turned off.  The maintained pack wins.
#
#   SDXL `low` is empty on purpose: SDXL generates 1024x1024 natively, so there is
#   nothing to correct and the honest preset is "do nothing".
PRESETS: dict[ModelFamily, Preset] = {
    ModelFamily.SD15: Preset(
        family=ModelFamily.SD15,
        native_megapixels=0.26,  # 512x512
        input_blocks="3",
        output_blocks="8",
        ca_input_blocks="1",
        ca_output_blocks="11",
        res_modes={
            "low": (0.0, 0.4, 1.0, 1.0),
            "high": (0.0, 0.5, 0.0, 0.35),
            "ultra": (0.0, 0.6, 0.0, 0.45),
        },
    ),
    ModelFamily.SDXL: Preset(
        family=ModelFamily.SDXL,
        native_megapixels=1.05,  # 1024x1024
        input_blocks="3",
        output_blocks="5",
        ca_input_blocks="4",
        ca_output_blocks="5",
        res_modes={
            "low": (),
            "high": (1.0, 1.0, 0.0, 0.5),
            "ultra": (0.0, 0.45, 0.0, 0.6),
        },
    ),
}

RES_MODES = ("low (1024 or lower)", "high (1536-2048)", "ultra (over 2048)")


def res_mode_key(res_mode: str) -> str:
    return str(res_mode).split(" ", 1)[0]


def _last_layer(block):
    try:
        return block[len(block) - 1]
    except (TypeError, IndexError, KeyError):
        return None


def _holds_transformer(block) -> bool:
    try:
        children = list(block)
    except TypeError:
        return False
    return any(type(layer).__name__ == "SpatialTransformer" for layer in children)


def inspect_unet(diffusion_model) -> UNetMap:
    """Walk a diffusion model and report the RAUNet-addressable blocks it has."""

    if diffusion_model is None:
        return UNetMap(False, "no diffusion model on this patcher")

    input_blocks = getattr(diffusion_model, "input_blocks", None)
    output_blocks = getattr(diffusion_model, "output_blocks", None)
    if input_blocks is None or output_blocks is None:
        arch = type(diffusion_model).__name__
        return UNetMap(False, f"{arch} is not a UNet (no input_blocks/output_blocks); RAUNet needs a down/up resolution pyramid")

    downs, ups, ca_in, ca_out = [], [], [], []

    for index, block in enumerate(input_blocks):
        last = _last_layer(block)
        if last is not None and type(last).__name__ == "Downsample" and getattr(last, "use_conv", False) and getattr(last, "dims", 2) == 2:
            downs.append(index)
        if _holds_transformer(block):
            ca_in.append(index)

    for index, block in enumerate(output_blocks):
        last = _last_layer(block)
        if last is not None and type(last).__name__ == "Upsample" and getattr(last, "use_conv", False) and getattr(last, "dims", 2) == 2:
            ups.append(index)
        if _holds_transformer(block):
            ca_out.append(index)

    if not downs or not ups:
        return UNetMap(False, "this UNet has no 2D conv Downsample/Upsample blocks to rewrite")

    return UNetMap(
        supported=True,
        n_input_blocks=len(input_blocks),
        n_output_blocks=len(output_blocks),
        downsample_blocks=tuple(downs),
        upsample_blocks=tuple(ups),
        ca_input_blocks=tuple(ca_in),
        ca_output_blocks=tuple(ca_out),
    )


def detect_family(sd_model=None, unet_map: UNetMap | None = None) -> ModelFamily | None:
    """Best guess at the preset family for the loaded checkpoint.

    The webui's own flags come first because they are authoritative
    (`backend/diffusion_engine/base.py` sets `is_sd1` / `is_sdxl`); block counts are the
    fallback for the case where this is called without a `sd_model` — e.g. from a test.
    """

    if sd_model is not None:
        if getattr(sd_model, "is_sdxl", False):
            return ModelFamily.SDXL
        if getattr(sd_model, "is_sd1", False):
            return ModelFamily.SD15

    if unet_map is not None and unet_map.supported:
        #   SD1.x/2.x: 12 input blocks and three downsamples. SDXL: 9 and two.
        if unet_map.n_input_blocks >= 12 and len(unet_map.downsample_blocks) >= 3:
            return ModelFamily.SD15
        if unet_map.n_input_blocks >= 8:
            return ModelFamily.SDXL

    return None


def describe(unet, sd_model=None) -> tuple[UNetMap, ModelFamily | None]:
    """Convenience entry point: `(map, family)` for a Forge `UnetPatcher`."""

    diffusion_model = getattr(getattr(unet, "model", None), "diffusion_model", None)
    unet_map = inspect_unet(diffusion_model)
    return unet_map, detect_family(sd_model, unet_map)


__all__ = (
    "PRESETS",
    "RES_MODES",
    "ModelFamily",
    "Preset",
    "UNetMap",
    "describe",
    "detect_family",
    "inspect_unet",
    "res_mode_key",
)
