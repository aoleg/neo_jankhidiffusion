"""Offline harness for the Neo RAUNet extension.

No GPU, no checkpoint, no running webui — it needs only torch, so the quickest way to run
it is with the webui's own interpreter:

    <forge>/venv/Scripts/python.exe tests/harness.py

  1. lib_jankhidiffusion.utils     - windows, block lists, rescaling, the MP gate
  2. lib_jankhidiffusion.unet_map  - block discovery, pairing rules, presets
  3. lib_jankhidiffusion.raunet    - the patches, end to end through a miniature UNet
                                     built to mirror SDXL's block layout
  4. scripts/neo_raunet.py         - arg plumbing and preset resolution, under stubs

Tier 3 is the one that earns its keep: it runs a real forward pass through a real
`torch.nn.Conv2d` with the stride/padding/dilation swap in place, so the 4x upsample and
the stride-4 downsample have to actually agree on shapes or the run fails.
"""

import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        print(f"  FAIL  {message}")


def section(title):
    print(f"\n=== {title} ===")


class FakePredictor:
    """Stands in for `backend/modules/k_prediction.py::Prediction`.

    sigma(t) = (t / 999) * 14.6 over a linear timestep grid, which is close enough to an
    SD schedule for the window maths and exactly invertible for the fadeout maths.
    """

    MAX_SIGMA = 14.6

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        return (1.0 - percent) * self.MAX_SIGMA

    def timestep(self, sigma):
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(float(sigma))
        return (sigma / self.MAX_SIGMA).clamp(0.0, 1.0) * 999.0


# =======================================================================================
section("tier 1 - lib_jankhidiffusion.utils")
# =======================================================================================

from lib_jankhidiffusion import utils

check(utils.parse_blocks("input", "3, 6") == {("input", 3), ("input", 6)}, "parse_blocks splits a string")
check(utils.parse_blocks("output", "") == set(), "parse_blocks accepts an empty string")
check(utils.parse_blocks("output", None) == set(), "parse_blocks accepts None")
check(utils.parse_blocks("input", [3, 6]) == {("input", 3), ("input", 6)}, "parse_blocks accepts a YAML list")
check(utils.parse_blocks("input", 3) == {("input", 3)}, "parse_blocks accepts a bare int")
try:
    utils.parse_blocks("input", [-1])
    check(False, "parse_blocks rejects negative block numbers")
except ValueError:
    check(True, "parse_blocks rejects negative block numbers")

pred = FakePredictor()

start, end = utils.convert_time(pred, "percent", 0.0, 0.45)
check(start == utils.NO_GATE_START, "0% start becomes the no-gate sentinel, not a live sigma")
check(abs(end - 0.55 * 14.6) < 0.01, "45% end converts through percent_to_sigma")
check(utils.check_time({"sigmas": 10.0}, start, end), "a sigma inside the window passes")
check(not utils.check_time({"sigmas": 1.0}, start, end), "a sigma past the end of the window fails")
check(not utils.check_time({}, start, end), "a dict with no sigmas fails closed")

check(utils.convert_time(pred, "sigma", 12.0, 3.0) == (12.0, 3.0), "sigma mode passes values straight through")
ts_start, ts_end = utils.convert_time(pred, "timestep", 0.0, 999.0)
check(ts_start < 0.0001 and ts_end == utils.NO_GATE_START, "timestep mode is inverted relative to percent")
try:
    utils.convert_time(pred, "percent", 0.0, 2.0)
    check(False, "percent mode rejects out-of-range values")
except ValueError:
    check(True, "percent mode rejects out-of-range values")

check(utils.sigma_to_pct(pred, float("inf")) == 0.0, "the no-gate sentinel maps back to 0%")
check(utils.sigma_to_pct(pred, 0.0) == 1.0, "sigma 0 maps back to 100%")
check(abs(utils.sigma_to_pct(pred, 7.3) - 0.5) < 0.01, "sigma_to_pct inverts percent_to_sigma")

check(utils.fade_scale(0.1, 0.0, 0.5, 0.3, 0.0) == 1.0, "fade_scale is 1.0 before the fade starts")
check(abs(utils.fade_scale(0.4, 0.0, 0.5, 0.3, 0.0) - 0.5) < 1e-6, "fade_scale ramps linearly")
check(utils.fade_scale(0.5, 0.0, 0.5, 0.3, 0.0) == 0.0, "fade_scale reaches 0 at the end of the window")
check(utils.fade_scale(0.5, 0.0, 0.5, 0.3, 0.25) == 0.25, "fade_scale respects its floor")
check(utils.fade_scale(0.9, 0.0, 0.5, 0.3, 0.0) == 0.0, "fade_scale is 0 outside the window")
check(utils.fade_scale(0.4, 0.0, 0.5, 0.5, 0.0) == 1.0, "a fade starting at the window end is a no-op")

check(abs(utils.latent_to_megapixels([1, 4, 128, 128]) - 1.048576) < 1e-6, "128x128 latent is 1.05MP")
check(abs(utils.latent_to_megapixels([1, 16, 1, 128, 128]) - 1.048576) < 1e-6, "a 5D latent measures its frame, not its frame count")
check(utils.latent_to_megapixels(None) == 0.0, "a missing shape measures 0")

sample = torch.randn(1, 4, 16, 16)
for mode in ("bicubic", "bilinear", "nearest-exact", "nearest", "area"):
    check(utils.scale_samples(sample, 32, 24, mode=mode).shape[-2:] == (24, 32), f"scale_samples honours (width, height) order for {mode}")

# =======================================================================================
section("tier 2 - lib_jankhidiffusion.unet_map")
# =======================================================================================

from lib_jankhidiffusion import unet_map as maps


class Downsample(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = self.out_channels = channels
        self.use_conv = True
        self.dims = 2
        self.op = torch.nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = self.out_channels = channels
        self.use_conv = True
        self.dims = 2
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, output_shape=None):
        shape = list(output_shape[2:4]) if output_shape is not None else [x.shape[-2] * 2, x.shape[-1] * 2]
        return self.conv(torch.nn.functional.interpolate(x, size=shape, mode="nearest"))


class SpatialTransformer(torch.nn.Module):
    """Only the class name is load-bearing - `inspect_unet` matches on it."""

    def forward(self, x, context=None, transformer_options=None):
        return x


def make_unet(channels=8, n_levels=3):
    """A miniature UNet with SDXL's block layout: 9 in, 9 out, downsamples at 3 and 6."""

    def conv():
        return torch.nn.Conv2d(channels, channels, 3, padding=1)

    def merge():
        return torch.nn.Conv2d(channels * 2, channels, 1)

    model = torch.nn.Module()
    model.in_conv = torch.nn.Conv2d(4, channels, 3, padding=1)
    model.out_conv = torch.nn.Conv2d(channels, 4, 3, padding=1)
    model.input_blocks = torch.nn.ModuleList(
        [
            torch.nn.ModuleList([conv()]),  # 0
            torch.nn.ModuleList([conv()]),  # 1
            torch.nn.ModuleList([conv()]),  # 2
            torch.nn.ModuleList([Downsample(channels)]),  # 3
            torch.nn.ModuleList([conv(), SpatialTransformer()]),  # 4
            torch.nn.ModuleList([conv(), SpatialTransformer()]),  # 5
            torch.nn.ModuleList([Downsample(channels)]),  # 6
            torch.nn.ModuleList([conv(), SpatialTransformer()]),  # 7
            torch.nn.ModuleList([conv(), SpatialTransformer()]),  # 8
        ]
    )
    model.middle_block = torch.nn.ModuleList([conv()])
    model.output_blocks = torch.nn.ModuleList(
        [
            torch.nn.ModuleList([merge(), SpatialTransformer()]),  # 0 <- input 8
            torch.nn.ModuleList([merge(), SpatialTransformer()]),  # 1 <- input 7
            torch.nn.ModuleList([merge(), SpatialTransformer(), Upsample(channels)]),  # 2 <- input 6
            torch.nn.ModuleList([merge(), SpatialTransformer()]),  # 3 <- input 5
            torch.nn.ModuleList([merge(), SpatialTransformer()]),  # 4 <- input 4
            torch.nn.ModuleList([merge(), SpatialTransformer(), Upsample(channels)]),  # 5 <- input 3
            torch.nn.ModuleList([merge()]),  # 6
            torch.nn.ModuleList([merge()]),  # 7
            torch.nn.ModuleList([merge()]),  # 8
        ]
    )
    return model


unet_model = make_unet()
umap = maps.inspect_unet(unet_model)

check(umap.supported, "a UNet-shaped model is supported")
check(umap.downsample_blocks == (3, 6), f"downsamples found at 3 and 6, got {umap.downsample_blocks}")
check(umap.upsample_blocks == (2, 5), f"upsamples found at 2 and 5, got {umap.upsample_blocks}")
check(umap.ca_input_blocks == (4, 5, 7, 8), f"cross-attention input blocks, got {umap.ca_input_blocks}")
check(umap.ca_output_blocks == (0, 1, 2, 3, 4, 5), f"cross-attention output blocks, got {umap.ca_output_blocks}")
check(umap.paired_output(3) == 5, "input 3 pairs with output 5 on an SDXL-shaped UNet")
check(umap.paired_output(6) == 2, "input 6 pairs with output 2")
check(umap.paired_output(9) is None, "a nonexistent input block has no pair")

check(umap.validate({("input", 3), ("output", 5)}) == [], "the canonical SDXL pairing validates")
check(umap.validate({("input", 3)}), "an unpaired input block is rejected")
check(umap.validate({("output", 5)}), "an unpaired output block is rejected")
check(umap.validate({("input", 4), ("output", 4)}), "a non-scaling block is rejected")
check(umap.advise_ca({("input", 4), ("output", 5)}) == [], "the canonical SDXL CA blocks pass without comment")
check(umap.advise_ca({("output", 8)}), "a CA block with no attention is flagged")
check(umap.advise_ca({("output", 42)}), "a CA block that does not exist is flagged")

dit = torch.nn.Module()
dit.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
dit.__class__.__name__ = "SingleStreamDiT"
dit_map = maps.inspect_unet(dit)
check(not dit_map.supported, "a DiT-shaped model is refused")
check("not a UNet" in dit_map.reason, f"and says why: {dit_map.reason!r}")
check(not maps.inspect_unet(None).supported, "a missing model is refused")

check(maps.detect_family(types.SimpleNamespace(is_sdxl=True, is_sd1=False)) is maps.ModelFamily.SDXL, "the webui flag identifies SDXL")
check(maps.detect_family(types.SimpleNamespace(is_sdxl=False, is_sd1=True)) is maps.ModelFamily.SD15, "the webui flag identifies SD15")
check(maps.detect_family(None, umap) is maps.ModelFamily.SDXL, "9 input blocks falls back to SDXL")
check(maps.detect_family(None, None) is None, "with nothing to go on, no guess")

for family, preset in maps.PRESETS.items():
    for key, window in preset.res_modes.items():
        check(window == () or len(window) == 4, f"{family} {key} window is a 4-tuple or empty")
        check(all(0.0 <= v <= 1.0 for v in window), f"{family} {key} window is in percent")
check(maps.PRESETS[maps.ModelFamily.SDXL].res_modes["high"] == (1.0, 1.0, 0.0, 0.5), "SDXL 'high' leaves the scaling blocks alone (upstream, not reForge)")
check(maps.res_mode_key("high (1536-2048)") == "high", "res_mode_key strips the parenthetical")

# =======================================================================================
section("tier 3 - lib_jankhidiffusion.raunet, end to end")
# =======================================================================================

from lib_jankhidiffusion.raunet import Config, apply_raunet


class FakePatcher:
    """The slice of `ModelPatcher` that `apply_raunet` touches, applied eagerly."""

    def __init__(self, model):
        self.model = types.SimpleNamespace(diffusion_model=model, predictor=FakePredictor())
        self.patches = {}
        self.object_patches = {}

    def get_model_object(self, name):
        obj = self.model
        for part in name.split("."):
            obj = obj[int(part)] if part.isdigit() and not hasattr(obj, part) else getattr(obj, part)
        return obj

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj
        target = self.get_model_object(name.rsplit(".", 1)[0])
        setattr(target, name.rsplit(".", 1)[1], obj)

    def _add(self, key, patch):
        self.patches.setdefault(key, []).append(patch)

    def set_model_input_block_patch(self, patch):
        self._add("input_block_patch", patch)

    def set_model_input_block_patch_after_skip(self, patch):
        self._add("input_block_patch_after_skip", patch)

    def set_model_output_block_patch(self, patch):
        self._add("output_block_patch", patch)


def run_unet(model, patcher, x, sigma):
    """Mirrors `IntegratedUNet2DConditionModel.forward` (`backend/nn/unet.py:657`)."""

    options = {"original_shape": list(x.shape), "sigmas": torch.tensor([sigma])}
    patches = patcher.patches
    hs = []
    h = model.in_conv(x)

    for index, block in enumerate(model.input_blocks):
        options["block"] = ("input", index)
        for layer in block:
            h = layer(h)
        for patch in patches.get("input_block_patch", ()):
            h = patch(h, options)
        hs.append(h)
        for patch in patches.get("input_block_patch_after_skip", ()):
            h = patch(h, options)

    options["block"] = ("middle", 0)
    for layer in model.middle_block:
        h = layer(h)

    for index, block in enumerate(model.output_blocks):
        options["block"] = ("output", index)
        hsp = hs.pop()
        for patch in patches.get("output_block_patch", ()):
            h, hsp = patch(h, hsp, options)
        h = torch.cat([h, hsp], dim=1)
        output_shape = hs[-1].shape if hs else None
        for layer in block:
            h = layer(h, output_shape=output_shape) if type(layer).__name__ == "Upsample" else layer(h)

    return model.out_conv(h)


BASE_SETTINGS = dict(
    input_blocks="3",
    output_blocks="5",
    ca_input_blocks="4",
    ca_output_blocks="5",
    time_mode="percent",
    start_time=0.0,
    end_time=0.45,
    ca_start_time=0.0,
    ca_end_time=0.6,
)

#   reference: the unpatched model, so every later comparison has a baseline
plain_model = make_unet()
plain_patcher = FakePatcher(plain_model)
latent_2048 = torch.randn(1, 4, 256, 256)  # 2048x2048 -> 4.19MP
with torch.no_grad():
    reference = run_unet(plain_model, plain_patcher, latent_2048, sigma=13.0)
check(reference.shape == latent_2048.shape, "the reference UNet round-trips its latent shape")

model = make_unet()
patcher = FakePatcher(model)
config = Config.build(FakePredictor(), **BASE_SETTINGS)
apply_raunet(patcher, config, maps.inspect_unet(model))

check("diffusion_model.input_blocks.3.0.forward" in patcher.object_patches, "the Downsample forward is patched")
check("diffusion_model.output_blocks.5.2.forward" in patcher.object_patches, "the Upsample forward is patched")
check(len(patcher.patches.get("input_block_patch", ())) == 1, "the input block patch is registered")
check(len(patcher.patches.get("output_block_patch", ())) == 1, "the output block patch is registered")

with torch.no_grad():
    out = run_unet(model, patcher, latent_2048, sigma=13.0)
check(out.shape == latent_2048.shape, f"RAUNet preserves the latent shape, got {tuple(out.shape)}")
check(config.hits_main >= 2, f"both scaling blocks fired, got {config.hits_main}")
check(config.hits_ca >= 2, f"both cross-attention hooks fired, got {config.hits_ca}")
check(not torch.allclose(out, reference), "and the result actually differs from the unpatched model")

#   past the end of both windows, nothing should fire and the output should match exactly
model_off = make_unet()
model_off.load_state_dict(plain_model.state_dict())
patcher_off = FakePatcher(model_off)
config_off = Config.build(FakePredictor(), **BASE_SETTINGS)
apply_raunet(patcher_off, config_off, maps.inspect_unet(model_off))
with torch.no_grad():
    late = run_unet(model_off, patcher_off, latent_2048, sigma=1.0)
check(config_off.hits_main == 0 and config_off.hits_ca == 0, "outside the sigma windows nothing fires")
check(torch.allclose(late, run_unet(plain_model, plain_patcher, latent_2048, sigma=1.0), atol=1e-5), "and the output is bit-for-bit the unpatched path")

#   the resolution gate
model_gate = make_unet()
model_gate.load_state_dict(plain_model.state_dict())
patcher_gate = FakePatcher(model_gate)
config_gate = Config.build(FakePredictor(), min_megapixels=1.155, **BASE_SETTINGS)
apply_raunet(patcher_gate, config_gate, maps.inspect_unet(model_gate))
latent_1024 = torch.randn(1, 4, 128, 128)  # 1024x1024 -> 1.05MP, below the gate
with torch.no_grad():
    gated = run_unet(model_gate, patcher_gate, latent_1024, sigma=13.0)
    ungated = run_unet(plain_model, plain_patcher, latent_1024, sigma=13.0)
check(config_gate.hits_main == 0 and config_gate.hits_ca == 0, "the gate suppresses the effect at native resolution")
check(config_gate.skips_resolution == 1, f"and counts the skip, got {config_gate.skips_resolution}")
check(torch.allclose(gated, ungated, atol=1e-5), "a gated pass is identical to an unpatched one")
with torch.no_grad():
    run_unet(model_gate, patcher_gate, latent_2048, sigma=13.0)
check(config_gate.hits_main >= 2, "the same config still fires above the gate")

#   odd latent sizes: 1536x1024 is 96x64 latent, and 4x scaling has to survive it
model_odd = make_unet()
patcher_odd = FakePatcher(model_odd)
config_odd = Config.build(FakePredictor(), **BASE_SETTINGS)
apply_raunet(patcher_odd, config_odd, maps.inspect_unet(model_odd))
latent_odd = torch.randn(1, 4, 64, 96)
with torch.no_grad():
    out_odd = run_unet(model_odd, patcher_odd, latent_odd, sigma=13.0)
check(out_odd.shape == latent_odd.shape, f"a non-square latent round-trips, got {tuple(out_odd.shape)}")

#   the conv layer must be handed back exactly as it was found
down_op = model.input_blocks[3][0].op
check((tuple(down_op.stride), tuple(down_op.padding), tuple(down_op.dilation)) == ((2, 2), (1, 1), (1, 1)), "the borrowed conv is restored after the call")

#   downscale modes and factors
for mode in ("adaptive_avg_pool2d", "avg_pool2d", "bicubic", "area"):
    m = make_unet()
    pt = FakePatcher(m)
    cfg = Config.build(FakePredictor(), ca_downscale_mode=mode, **BASE_SETTINGS)
    apply_raunet(pt, cfg, maps.inspect_unet(m))
    with torch.no_grad():
        r = run_unet(m, pt, latent_2048, sigma=13.0)
    check(r.shape == latent_2048.shape, f"ca_downscale_mode {mode} round-trips")
    check(cfg.hits_ca >= 2, f"ca_downscale_mode {mode} actually rescaled")

try:
    Config.build(FakePredictor(), ca_downscale_mode="avg_pool2d", ca_downscale_factor=1.5, **BASE_SETTINGS)
    check(False, "avg_pool2d rejects a fractional downscale factor")
except ValueError:
    check(True, "avg_pool2d rejects a fractional downscale factor")

m = make_unet()
pt = FakePatcher(m)
cfg = Config.build(FakePredictor(), ca_downscale_mode="adaptive_avg_pool2d", ca_downscale_factor=1.5, **BASE_SETTINGS)
apply_raunet(pt, cfg, maps.inspect_unet(m))
with torch.no_grad():
    r = run_unet(m, pt, latent_2048, sigma=13.0)
check(r.shape == latent_2048.shape, "a fractional downscale factor round-trips under adaptive_avg_pool2d")

#   the fadeout ramp
fade = Config.build(FakePredictor(), ca_fadeout_start_time=0.3, ca_fadeout_cap=0.0, **BASE_SETTINGS)
early = fade.current_downscale_factors({"sigmas": torch.tensor([13.0])})
mid = fade.current_downscale_factors({"sigmas": torch.tensor([FakePredictor().percent_to_sigma(0.45)])})
late_f = fade.current_downscale_factors({"sigmas": torch.tensor([FakePredictor().percent_to_sigma(0.6)])})
check(abs(early[0] - 2.0) < 1e-6, f"before the fade the factor is untouched, got {early[0]}")
check(1.0 < mid[0] < 2.0, f"inside the fade the factor is on its way to 1.0, got {mid[0]}")
check(abs(late_f[0] - 1.0) < 1e-3, f"at the end of the window the factor is 1.0 (no rescale), got {late_f[0]}")
no_fade = Config.build(FakePredictor(), **BASE_SETTINGS)
check(no_fade.current_downscale_factors({"sigmas": torch.tensor([5.0])})[0] == 2.0, "with no fade configured the factor never moves")

#   two-stage upscale, and after-skip mode with its shifted CA pairing
for extra in ({"two_stage_upscale_mode": "nearest-exact"}, {"ca_input_after_skip_mode": True, "ca_output_blocks": "4"}):
    m = make_unet()
    pt = FakePatcher(m)
    cfg = Config.build(FakePredictor(), **{**BASE_SETTINGS, **extra})
    apply_raunet(pt, cfg, maps.inspect_unet(m))
    with torch.no_grad():
        r = run_unet(m, pt, latent_2048, sigma=13.0)
    check(r.shape == latent_2048.shape, f"{list(extra)[0]} round-trips")
    check(cfg.hits_ca >= 2, f"{list(extra)[0]} still rescales cross-attention")

#   the CA pairing rule, which is what makes the case above work
check(umap.ca_paired_output(4) == 5, "CA input 4 pairs with CA output 5 in normal mode")
check(umap.ca_paired_output(4, after_skip=True) == 4, "after-skip mode shifts the CA pairing in by one")
check(maps.inspect_unet(make_unet()).advise_ca({("input", 4), ("output", 5)}, after_skip=True), "the wrong CA pairing is flagged before torch.cat can fail on it")
check(maps.inspect_unet(make_unet()).advise_ca({("input", 4), ("output", 4)}, after_skip=True) == [], "and the right one passes quietly")

#   cross-attention only, which is what the SDXL 'high' preset asks for
m = make_unet()
pt = FakePatcher(m)
cfg = Config.build(FakePredictor(), **{**BASE_SETTINGS, "input_blocks": "", "output_blocks": "", "ca_end_time": 0.5})
apply_raunet(pt, cfg, maps.inspect_unet(m))
with torch.no_grad():
    r = run_unet(m, pt, latent_2048, sigma=13.0)
check(r.shape == latent_2048.shape, "a cross-attention-only config round-trips")
check(cfg.hits_main == 0 and cfg.hits_ca >= 2, "and only the cross-attention half fires")

#   a bad pairing is refused before anything is patched
try:
    m = make_unet()
    apply_raunet(FakePatcher(m), Config.build(FakePredictor(), **{**BASE_SETTINGS, "output_blocks": "2"}), maps.inspect_unet(m))
    check(False, "a mismatched input/output pairing is refused")
except ValueError:
    check(True, "a mismatched input/output pairing is refused")

# =======================================================================================
section("tier 4 - scripts/neo_raunet.py under stubs")
# =======================================================================================


def install_stubs():
    gradio = types.ModuleType("gradio")

    class Component:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def change(self, *args, **kwargs):
            return None

    for name in ("Radio", "Dropdown", "Textbox", "Slider", "Checkbox", "Markdown", "HTML", "Row", "Group", "Accordion", "Number"):
        setattr(gradio, name, type(name, (Component,), {}))
    gradio.update = lambda **kwargs: kwargs
    sys.modules["gradio"] = gradio

    modules = types.ModuleType("modules")
    modules.__path__ = []

    scripts_mod = types.ModuleType("modules.scripts")

    class Script:
        AlwaysVisible = object()

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = Script.AlwaysVisible
    scripts_mod.scripts_data = []
    modules.scripts = scripts_mod

    processing = types.ModuleType("modules.processing")
    logged = []

    class Logger:
        def _log(self, level, msg):
            logged.append((level, str(msg)))

        info = lambda self, m: self._log("info", m)
        warning = lambda self, m: self._log("warning", m)
        error = lambda self, m: self._log("error", m)
        debug = lambda self, m: self._log("debug", m)

    processing.logger = Logger()
    modules.processing = processing

    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = type("InputAccordion", (Component,), {})
    modules.ui_components = ui_components

    sys.modules.update(
        {
            "modules": modules,
            "modules.scripts": scripts_mod,
            "modules.processing": processing,
            "modules.ui_components": ui_components,
        }
    )
    return logged


log_records = install_stubs()

import importlib.util

spec = importlib.util.spec_from_file_location("neo_raunet", REPO / "scripts" / "neo_raunet.py")
neo_raunet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neo_raunet)
NeoRAUNet = neo_raunet.NeoRAUNet

DEFAULT_UI = {
    "mode": "Simple",
    "model_type": "auto",
    "res_mode": maps.RES_MODES[1],
    "simple_upscale_mode": "default",
    "simple_ca_upscale_mode": "default",
    "input_blocks": "3",
    "output_blocks": "5",
    "time_mode": "percent",
    "start_time": 0.0,
    "end_time": 0.45,
    "upscale_mode": "bicubic",
    "two_stage_upscale_mode": "disabled",
    "ca_input_blocks": "4",
    "ca_output_blocks": "5",
    "ca_start_time": 0.0,
    "ca_end_time": 0.6,
    "ca_downscale_factor": 2.0,
    "ca_downscale_mode": "adaptive_avg_pool2d",
    "ca_ca_upscale_mode": "bicubic",
    "ca_fadeout_start_time": 0.0,
    "ca_fadeout_cap": 0.0,
    "ca_input_after_skip_mode": False,
    "yaml_parameters": "",
    "auto_gate": True,
    "min_megapixels": 0.0,
    "apply_to_hr": True,
}

#   DEFAULT_UI is written in the same order as the component list, so this catches a key
#   added to one and not the other
round_tripped = NeoRAUNet._named_args(tuple(DEFAULT_UI.values()))
check(round_tripped == DEFAULT_UI, "the positional arg list and the key list stay in step")

for family in (maps.ModelFamily.SD15, maps.ModelFamily.SDXL):
    for res in maps.RES_MODES:
        ui = dict(DEFAULT_UI, mode="Simple", model_type=str(family), res_mode=res)
        settings = NeoRAUNet._settings(ui, family)
        if settings is None:
            check(family is maps.ModelFamily.SDXL and maps.res_mode_key(res) == "low", f"only SDXL 'low' resolves to nothing, not {family} {res}")
            continue
        cfg = Config.build(FakePredictor(), **settings, min_megapixels=0.0)
        check(cfg.use_blocks or cfg.ca_use_blocks, f"{family} {res} selects at least one half")
        check(maps.inspect_unet(make_unet()).validate(cfg.use_blocks) == [] or family is maps.ModelFamily.SD15, f"{family} {res} block pairing is valid on an SDXL-shaped UNet")

sdxl_high = NeoRAUNet._settings(dict(DEFAULT_UI, model_type="SDXL", res_mode=maps.RES_MODES[1]), maps.ModelFamily.SDXL)
check(sdxl_high["input_blocks"] == "" and sdxl_high["ca_input_blocks"] == "4", "SDXL 'high' is cross-attention only")
sdxl_ultra = NeoRAUNet._settings(dict(DEFAULT_UI, model_type="SDXL", res_mode=maps.RES_MODES[2]), maps.ModelFamily.SDXL)
check(sdxl_ultra["input_blocks"] == "3" and sdxl_ultra["output_blocks"] == "5", "SDXL 'ultra' turns the scaling blocks back on")

advanced = NeoRAUNet._settings(dict(DEFAULT_UI, mode="Advanced", ca_fadeout_start_time=0.0), None)
check(advanced["ca_fadeout_start_time"] is None, "a fadeout slider at 0 disables the fade rather than fading from step 0")
advanced = NeoRAUNet._settings(dict(DEFAULT_UI, mode="Advanced", ca_fadeout_start_time=0.3), None)
check(advanced["ca_fadeout_start_time"] == 0.3, "a fadeout slider above 0 is passed through")

check(abs(NeoRAUNet._gate_threshold(dict(DEFAULT_UI, auto_gate=True), maps.ModelFamily.SDXL) - 1.155) < 1e-3, "the auto gate sits just above SDXL's native 1.05MP")
check(abs(NeoRAUNet._gate_threshold(dict(DEFAULT_UI, auto_gate=True), maps.ModelFamily.SD15) - 0.286) < 1e-3, "the auto gate follows SD15's native 0.26MP")
check(NeoRAUNet._gate_threshold(dict(DEFAULT_UI, auto_gate=False, min_megapixels=2.0), maps.ModelFamily.SDXL) == 2.0, "the manual gate wins when auto is off")
check(NeoRAUNet._gate_threshold(dict(DEFAULT_UI, auto_gate=True), None) == 0.0, "an unknown family gets no gate rather than a wrong one")

merged = NeoRAUNet._merge_yaml({"start_time": 0.0}, "start_time: 0.2\nca_post_upscale_multiplier: 1.05")
check(merged["start_time"] == 0.2 and merged["ca_post_upscale_multiplier"] == 1.05, "YAML overrides and extends the settings dict")
check(NeoRAUNet._merge_yaml({"a": 1}, "   ") == {"a": 1}, "blank YAML is a no-op")
try:
    NeoRAUNet._merge_yaml({}, "- 1\n- 2")
    check(False, "a non-mapping YAML document is rejected")
except ValueError:
    check(True, "a non-mapping YAML document is rejected")

NeoRAUNet.xyz_cache.update({"enable": "False", "end_time": 0.7})
ui = dict(DEFAULT_UI)
still_enabled = NeoRAUNet._apply_xyz(True, ui)
check(still_enabled is False, "an XYZ 'Enable=False' cell switches the effect off")
check(ui["end_time"] == 0.7, "an XYZ value overrides the UI value")
check(NeoRAUNet.xyz_cache == {}, "and the cache is cleared afterwards")

check(neo_raunet._is_dy_sampler("Euler Dy CFG++"), "Euler Dy CFG++ is recognised")
check(neo_raunet._is_dy_sampler("Euler SMEA Dy"), "Euler SMEA Dy is recognised")
check(not neo_raunet._is_dy_sampler("DPM++ 2M SDE"), "an ordinary sampler is not")
check(not neo_raunet._is_dy_sampler("Dynamic Thresholding"), "and 'dy' inside a word does not count")
check(not neo_raunet._is_dy_sampler(""), "a missing sampler name is handled")


#   the full hook sequence, so an XYZ override has to survive from `process` to
#   `process_before_every_sampling` and the patches have to land on the cloned unet
class FakeP:
    def __init__(self, model, sampler_name="Euler a", is_hr_pass=False):
        patcher = FakePatcher(model)
        patcher.clone = lambda: patcher
        self.sd_model = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(unet=patcher),
            is_sdxl=True,
            is_sd1=False,
        )
        self.extra_generation_params = {}
        self.sampler_name = sampler_name
        self.is_hr_pass = is_hr_pass


script = NeoRAUNet()
ordered_args = tuple(dict(DEFAULT_UI, mode="Advanced").values())

NeoRAUNet.xyz_cache.update({"ca_downscale_factor": 1.5})
fake_p = FakeP(make_unet())
script.process(fake_p, True, *ordered_args)
check(NeoRAUNet.active, "process arms the extension")
check(NeoRAUNet.resolved["ca_downscale_factor"] == 1.5, "the XYZ override is carried past process")
check("RAUNet mode" in fake_p.extra_generation_params, "infotext is written")

script.process_before_every_sampling(fake_p, True, *ordered_args)
patched = fake_p.sd_model.forge_objects.unet
check("diffusion_model.input_blocks.3.0.forward" in patched.object_patches, "the sampling hook patched the unet")
check(NeoRAUNet.configs and NeoRAUNet.configs[0].ca_downscale_factor == 1.5, "and it used the XYZ value, not the UI one")
check(abs(NeoRAUNet.configs[0].min_megapixels - 1.155) < 1e-3, "the auto gate resolved against the detected SDXL family")
check(any("not one of the Dy" in msg for level, msg in log_records if level == "info"), "and the sampler note fired for a non-Dy sampler")

log_records.clear()
NeoRAUNet.configs = []
script.process_before_every_sampling(FakeP(make_unet(), sampler_name="Euler Dy CFG++"), True, *ordered_args)
check(not any("not one of the Dy" in msg for level, msg in log_records), "the sampler note stays quiet for a Dy sampler")

script.postprocess(fake_p, None, True, *ordered_args)
check(not NeoRAUNet.active and NeoRAUNet.resolved == {} and NeoRAUNet.configs == [], "postprocess clears every scrap of state")

#   a run that never patched anything must not leave RAUNet parameters in the infotext
stranded = FakeP(make_unet())
script.process(stranded, True, *ordered_args)
check(any(k.startswith("RAUNet ") for k in stranded.extra_generation_params), "process writes the parameters up front")
NeoRAUNet.configs = []
script.postprocess(stranded, None, True, *ordered_args)
check(not any(k.startswith("RAUNet ") for k in stranded.extra_generation_params), "and postprocess takes them back if nothing was patched")

#   a DiT-shaped model is declined rather than crashed into
log_records.clear()
NeoRAUNet.active = True
NeoRAUNet.resolved = dict(DEFAULT_UI, mode="Advanced")
dit_p = FakeP(make_unet())
dit_p.sd_model.forge_objects.unet.model.diffusion_model = dit
script.process_before_every_sampling(dit_p, True, *ordered_args)
check(any("not a UNet" in msg for level, msg in log_records if level == "warning"), "a DiT model is declined with an explanation")
NeoRAUNet.active = False
NeoRAUNet.resolved = {}

# =======================================================================================
print(f"\n{'=' * 72}")
if failures:
    print(f"{len(failures)} of {checks} checks FAILED:")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print(f"all {checks} checks passed")
