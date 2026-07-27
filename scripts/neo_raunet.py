"""RAUNet for WebUI Forge Neo — a port of blepping's `comfyui_jankhidiffusion`.

Base choice: the **ComfyUI** node, not the older reForge script.  Neo's UNet keeps
ComfyUI's extension surface almost verbatim (`input_block_patch`, `output_block_patch`,
`add_object_patch`, `transformer_options["sigmas"]`), so the maintained upstream ports
across with less adaptation than the reForge fork needed — and it brings several years of
fixes the fork predates: the cross-attention fadeout, fractional downscale factors,
`adaptive_avg_pool2d`, per-clone state instead of a global singleton, and SDXL presets
that agree with themselves.  The reForge script contributed the UI shape only.

Three things here are Forge-specific rather than ported:

* **The resolution gate.**  In ComfyUI you wire RAUNet into one KSampler.  In a webui the
  patched UNet is reused by every pass of the run, so with Hires. fix on, an unguarded
  RAUNet also mangles the low-resolution first pass — the pass it is supposed to leave
  alone.  The gate skips the effect whenever the latent being denoised is at or below the
  model's native resolution, which fixes that and, as a side effect, keeps RAUNet out of
  the half-resolution sub-steps that Euler Dy and friends take.
* **The sampler note.**  Measured: RAUNet is much better behaved under CFG++ samplers,
  which steer along the unconditional prediction and so do not multiply its resampling
  artifacts by the guidance scale.  If the settings are aggressive on a sampler that is
  neither CFG++ nor Dy, the log says so once.  See the README.
* **XYZ axes and infotext**, as usual for a webui script.
"""

import re

import gradio as gr
from modules import scripts
from modules.processing import logger
from modules.ui_components import InputAccordion

from lib_jankhidiffusion import unet_map as maps
from lib_jankhidiffusion import xyz
from lib_jankhidiffusion.controlnet import install as install_controlnet_shim
from lib_jankhidiffusion.raunet import Config, apply_raunet
from lib_jankhidiffusion.utils import (
    DOWNSCALE_METHODS,
    TWO_STAGE_METHODS,
    UPSCALE_METHODS,
)

SIMPLE = "Simple"
ADVANCED = "Advanced"
MODES = [SIMPLE, ADVANCED]

TIME_MODES = ["percent", "timestep", "sigma"]
#   SDXL first and default: it is what this port targets, and SD1.x is marginal now.
#   `auto` still works and still detects correctly - it is just not the default, so a
#   fresh install never starts on another architecture's numbers.
FAMILIES = [str(maps.ModelFamily.SDXL), str(maps.ModelFamily.SD15)]
MODEL_TYPE_CHOICES = [*FAMILIES, "auto"]
DEFAULT_MODEL_TYPE = str(maps.ModelFamily.SDXL)

#   Samplers that absorb the end of the RAUNet window without visible damage. Measured on
#   SDXL at 1792px with the SDXL 'high' preset (cross-attention 0-50%):
#
#     Euler                deformed faces and limbs
#     Euler CFG++          clean
#     Euler Dy CFG++       clean, and near-identical to Euler CFG++
#     Euler Dy             clean, but heavily re-noised - see README
#
#   So it is the **CFG++ combination** that carries this, not the Dy sub-step: at 40 steps
#   `dy_sampling_step` fires twice (`if i // 2 == 1`), which is why the two CFG++ variants
#   land in the same place. Dy stays on the list because it does help on its own.
#
#   Matched as whole words, so a sampler with "dy" buried in its name is not mistaken for
#   one; "+" is kept in the token so "cfg++" survives the split.
FORGIVING_MARKERS = frozenset({"cfg++", "cfgpp", "dy", "smea"})


def _is_forgiving_sampler(name: str) -> bool:
    return bool(FORGIVING_MARKERS & set(re.split(r"[^a-z0-9+]+", (name or "").lower())))


#   Native resolution, times a little headroom, is the gate threshold. 1024x1024 SDXL is
#   1.05MP, so 1.1x keeps "exactly native" on the skip side without excluding 1152x896.
AUTO_GATE_HEADROOM = 1.1

#   Upstream's SDXL presets enable the scaling-block rewrite only in the `ultra` band,
#   "over 2048"; the 1536-2048 band gets cross-attention alone. Measured at 1792px, that
#   division is real - scaling blocks 3/5 at 0-45% blotch on Euler, Euler CFG++ and Euler
#   Dy CFG++ alike. The Advanced tab cannot know your band, so it says so instead.
SCALING_BLOCK_MIN_SIDE = 2048

#   The four time sliders mean different things per time mode, and 0-1 is only right for
#   one of them: timesteps run 0-999 and sigma tops out around 14.6 on an SD schedule.
TIME_RANGES = {"percent": (1.0, 0.01), "timestep": (999.0, 1.0), "sigma": (20.0, 0.05)}


class NeoRAUNet(scripts.Script):
    sorting_priority = 16.05

    #   class attributes: `process_before_every_sampling` returns long before the patches
    #   run, and `postprocess` is not guaranteed to see the same instance
    xyz_cache: dict = {}
    active: bool = False
    configs: list = []
    #   the UI arguments *after* X/Y/Z overrides.  `process` is where the axis cache can
    #   be read and cleared (it runs once per grid cell, before any sampling), so the
    #   resolved values have to be carried forward rather than re-derived per pass.
    resolved: dict = {}

    def title(self):
        return "RAUNet (Neo)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ------------------------------------------------------------------------ UI ----

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enable:
            gr.Markdown("Generate above a model's native resolution with fewer duplicated subjects and less mush. SD1.x / SD2.x / SDXL only — this rewrites UNet scaling blocks, which DiT models (Flux, Qwen, Krea 2) do not have.")

            #   Model family sits outside the Simple group on purpose: it refills the
            #   Advanced block numbers, so hiding it in Advanced mode is how you end up
            #   hand-editing the scaling blocks while the CA blocks keep another
            #   architecture's values.
            with gr.Row():
                mode = gr.Radio(value=SIMPLE, choices=MODES, label="Mode", info="Simple picks a preset for the model and resolution; Advanced exposes every knob")
                model_type = gr.Dropdown(value=DEFAULT_MODEL_TYPE, choices=list(MODEL_TYPE_CHOICES), label="Model family", info="decides the timing only - block numbers always come from the loaded checkpoint. Choosing one here also refills the Advanced block numbers; 'auto' reads it from the checkpoint")

            with gr.Group() as g_simple:
                with gr.Row():
                    res_mode = gr.Dropdown(value=maps.RES_MODES[1], choices=list(maps.RES_MODES), label="Resolution mode", info="a preset band, not a match against your actual size")
                with gr.Row():
                    simple_upscale_mode = gr.Dropdown(value="default", choices=["default", *UPSCALE_METHODS], label="Upscale mode")
                    simple_ca_upscale_mode = gr.Dropdown(value="default", choices=["default", *UPSCALE_METHODS], label="CA upscale mode")

            with gr.Group(visible=False) as g_advanced:
                with gr.Row():
                    input_blocks = gr.Textbox(value="3", label="Scaling input blocks", info="comma-separated Downsample blocks")
                    output_blocks = gr.Textbox(value="5", label="Scaling output blocks", info="the matching Upsample blocks")
                gr.Markdown(
                    "**Scaling block pairings** — SDXL: 3 with 5, 6 with 2. SD1.5: 3 with 8, 6 with 5, 9 with 2. "
                    "HiDiffusion's own SDXL setting is 6/2; 3/5 is gentler. "
                    "Defaults on this tab are SDXL — set **Model family** above to SD15 to refill them."
                )

                time_mode = gr.Dropdown(value="percent", choices=TIME_MODES, label="Time mode", info="how the start/end values below are read; use percent unless you know otherwise")
                with gr.Row():
                    start_time = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Start")
                    end_time = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.45, label="End", info="past this point the model runs unmodified")
                with gr.Row():
                    upscale_mode = gr.Dropdown(value="bicubic", choices=list(UPSCALE_METHODS), label="Upscale mode", info="bicubic or bislerp")
                    two_stage_upscale_mode = gr.Dropdown(value="disabled", choices=list(TWO_STAGE_METHODS), label="Two-stage upscale", info="do half the upscale with this mode first; different, not necessarily better")

                with gr.Accordion(open=False, label="Cross-attention (off by default)"):
                    #   Off by default and behind a closed accordion because it is the
                    #   destructive half: it rescales a hidden state the model is about to
                    #   attend over, and the shallower the block the more violent that is.
                    #   Measured on SDXL at 1792px: scaling blocks 3/5 alone behave, adding
                    #   the CA rescale at blocks 2/7 turns the image blotchy.
                    ca_enabled = gr.Checkbox(value=False, label="Enable the cross-attention rescale", info="a second, independent effect; try the scaling blocks on their own first")
                    gr.Markdown("**CA pairings** — SDXL: 4 with 5, 5 with 4. SD1.5: 1 with 11, 2 with 10. The rule is `output = block count - input`, and blocks nearer the middle of the UNet (SDXL 4/5) change the image far less than shallow ones.")
                    with gr.Row():
                        ca_input_blocks = gr.Textbox(value="4", label="CA input blocks")
                        ca_output_blocks = gr.Textbox(value="5", label="CA output block")
                    with gr.Row():
                        ca_start_time = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="CA start")
                        ca_end_time = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.3, label="CA end")
                    with gr.Row():
                        ca_downscale_factor = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, value=2.0, label="CA downscale factor", info="2.0 = half size; fractional values need adaptive_avg_pool2d")
                        ca_downscale_mode = gr.Dropdown(value="adaptive_avg_pool2d", choices=list(DOWNSCALE_METHODS), label="CA downscale mode", info="avg_pool2d is stock HiDiffusion; adaptive_avg_pool2d matches it and allows fractional factors")
                        ca_ca_upscale_mode = gr.Dropdown(value="bicubic", choices=list(UPSCALE_METHODS), label="CA upscale mode")
                    with gr.Row():
                        ca_fadeout_start_time = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="CA fadeout start", info="taper the downscale from here to CA end instead of cutting it off; 0 disables. The single most useful setting if the image falls apart when the effect ends")
                        ca_fadeout_cap = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="CA fadeout floor", info="how much of the effect survives the taper")
                    ca_input_after_skip_mode = gr.Checkbox(value=False, label="Patch input blocks after the skip connection", info="the skip connection keeps the original resolution; changes the effect noticeably")

                yaml_parameters = gr.Textbox(value="", lines=3, label="Extra parameters (YAML)", info="overrides any Config field by name — pre/post scale multipliers, ca_downscale_factor_w, ca_avg_pool2d_ceil_mode, verbose")

            with gr.Row():
                auto_gate = gr.Checkbox(value=True, label="Skip at or below native resolution", info="leaves the Hires. fix first pass and the rescaled sub-steps of Dy samplers alone")
                min_megapixels = gr.Slider(minimum=0.0, maximum=8.0, step=0.05, value=0.0, label="…or skip below (MP)", info="used when the checkbox above is off; 0 = always apply")
            apply_to_hr = gr.Checkbox(value=True, label="Apply to the Hires. fix pass")

        mode.change(
            fn=lambda chosen: [gr.update(visible=chosen == SIMPLE), gr.update(visible=chosen == ADVANCED)],
            inputs=[mode],
            outputs=[g_simple, g_advanced],
            show_progress=False,
        )

        #   Choosing a model type refills the Advanced *block numbers*, which are structural
        #   facts about that architecture. It deliberately leaves the time windows alone:
        #   those are the preset's opinion, and overwriting a hand-tuned schedule because
        #   someone touched a dropdown is worse than the inconsistency it would prevent.
        model_type.change(
            fn=self._blocks_for,
            inputs=[model_type],
            outputs=[input_blocks, output_blocks, ca_input_blocks, ca_output_blocks],
            show_progress=False,
        )

        time_sliders = [start_time, end_time, ca_start_time, ca_end_time, ca_fadeout_start_time]
        time_mode.change(fn=self._time_ranges, inputs=[time_mode], outputs=time_sliders, show_progress=False)

        self.infotext_fields = [
            (enable, lambda d: "RAUNet mode" in d),
            (mode, "RAUNet mode"),
            (model_type, "RAUNet model type"),
            (res_mode, "RAUNet res mode"),
            (simple_upscale_mode, "RAUNet simple upscale"),
            (simple_ca_upscale_mode, "RAUNet simple CA upscale"),
            (input_blocks, "RAUNet input blocks"),
            (output_blocks, "RAUNet output blocks"),
            (time_mode, "RAUNet time mode"),
            (start_time, "RAUNet start"),
            (end_time, "RAUNet end"),
            (upscale_mode, "RAUNet upscale"),
            (two_stage_upscale_mode, "RAUNet two-stage upscale"),
            (ca_enabled, "RAUNet CA"),
            (ca_input_blocks, "RAUNet CA input blocks"),
            (ca_output_blocks, "RAUNet CA output blocks"),
            (ca_start_time, "RAUNet CA start"),
            (ca_end_time, "RAUNet CA end"),
            (ca_downscale_factor, "RAUNet CA downscale factor"),
            (ca_downscale_mode, "RAUNet CA downscale"),
            (ca_ca_upscale_mode, "RAUNet CA upscale"),
            (ca_fadeout_start_time, "RAUNet CA fadeout start"),
            (ca_fadeout_cap, "RAUNet CA fadeout floor"),
            (ca_input_after_skip_mode, "RAUNet CA after skip"),
            (auto_gate, "RAUNet auto gate"),
            (min_megapixels, "RAUNet min MP"),
        ]

        components = [
            enable,
            mode,
            model_type,
            res_mode,
            simple_upscale_mode,
            simple_ca_upscale_mode,
            input_blocks,
            output_blocks,
            time_mode,
            start_time,
            end_time,
            upscale_mode,
            two_stage_upscale_mode,
            ca_enabled,
            ca_input_blocks,
            ca_output_blocks,
            ca_start_time,
            ca_end_time,
            ca_downscale_factor,
            ca_downscale_mode,
            ca_ca_upscale_mode,
            ca_fadeout_start_time,
            ca_fadeout_cap,
            ca_input_after_skip_mode,
            yaml_parameters,
            auto_gate,
            min_megapixels,
            apply_to_hr,
        ]

        xyz.register(NeoRAUNet.xyz_cache, UPSCALE_METHODS, DOWNSCALE_METHODS, maps.RES_MODES, FAMILIES)
        return components

    @staticmethod
    def _time_ranges(time_mode):
        """Widen the time sliders when they stop meaning percentages."""

        maximum, step = TIME_RANGES.get(str(time_mode), TIME_RANGES["percent"])
        return [gr.update(maximum=maximum, step=step) for _ in range(5)]

    @staticmethod
    def _blocks_for(model_type):
        """The four Advanced block fields for a model family. SDXL when unspecified."""

        family = maps.ModelFamily.SDXL if model_type == "auto" else maps.ModelFamily(model_type)
        preset = maps.PRESETS[family]
        return [
            gr.update(value=preset.input_blocks),
            gr.update(value=preset.output_blocks),
            gr.update(value=preset.ca_input_blocks),
            gr.update(value=preset.ca_output_blocks),
        ]

    # ------------------------------------------------------------- arg plumbing ----

    @staticmethod
    def _named_args(args) -> dict:
        keys = (
            "mode",
            "model_type",
            "res_mode",
            "simple_upscale_mode",
            "simple_ca_upscale_mode",
            "input_blocks",
            "output_blocks",
            "time_mode",
            "start_time",
            "end_time",
            "upscale_mode",
            "two_stage_upscale_mode",
            "ca_enabled",
            "ca_input_blocks",
            "ca_output_blocks",
            "ca_start_time",
            "ca_end_time",
            "ca_downscale_factor",
            "ca_downscale_mode",
            "ca_ca_upscale_mode",
            "ca_fadeout_start_time",
            "ca_fadeout_cap",
            "ca_input_after_skip_mode",
            "yaml_parameters",
            "auto_gate",
            "min_megapixels",
            "apply_to_hr",
        )
        return dict(zip(keys, args, strict=True))

    @classmethod
    def _apply_xyz(cls, enable: bool, ui: dict) -> bool:
        for field, value in cls.xyz_cache.items():
            if field == "enable":
                enable = str(value).lower() not in ("false", "0", "none", "")
            elif field == "mode":
                ui["mode"] = value
            elif field in ui:
                ui[field] = value
        cls.xyz_cache.clear()
        return enable

    @staticmethod
    def _settings(ui: dict, family: maps.ModelFamily | None, unet_map=None) -> dict | None:
        """UI arguments -> the keyword arguments `Config.build` takes.

        Simple mode is a preset lookup that then goes through the exact same path as
        Advanced, so there is only one code path to be wrong about.
        """

        if ui["mode"] == SIMPLE:
            if family is None:
                logger.warning("RAUNet: could not identify the model family; pick one explicitly in Model family")
                return None

            preset = maps.PRESETS[family]
            key = maps.res_mode_key(ui["res_mode"])
            window = preset.res_modes.get(key)
            if window is None:
                logger.warning(f"RAUNet: no preset for {family} / {key}")
                return None
            if not window:
                logger.info(f"RAUNet: the {family} '{key}' preset is a no-op; that resolution is native for this model")
                return None

            #   Block numbers come from the loaded model, not from the family's table.  The
            #   family only decides the *timing*: it is a guess (a dropdown, or a flag on
            #   the checkpoint) and a wrong guess should cost a suboptimal schedule, not a
            #   config full of another architecture's block numbers.
            blocks = unet_map.default_blocks() if unet_map is not None else None
            if blocks is None:
                logger.warning("RAUNet: could not work out this model's scaling blocks")
                return None

            #   `start >= end` is upstream's way of saying "this half is off"; blank the
            #   block lists rather than relying on an empty sigma window to do it, so the
            #   log line and the infotext both read the way the preset means
            start, end, ca_start, ca_end = window
            main_enabled = start < end
            ca_on = ca_start < ca_end and blocks["ca_input_blocks"]
            return {
                "input_blocks": blocks["input_blocks"] if main_enabled else "",
                "output_blocks": blocks["output_blocks"] if main_enabled else "",
                "ca_input_blocks": blocks["ca_input_blocks"] if ca_on else "",
                "ca_output_blocks": blocks["ca_output_blocks"] if ca_on else "",
                "time_mode": "percent",
                "start_time": start,
                "end_time": end,
                "ca_start_time": ca_start,
                "ca_end_time": ca_end,
                "upscale_mode": "bicubic" if ui["simple_upscale_mode"] == "default" else ui["simple_upscale_mode"],
                "ca_upscale_mode": "bicubic" if ui["simple_ca_upscale_mode"] == "default" else ui["simple_ca_upscale_mode"],
            }

        fadeout = float(ui["ca_fadeout_start_time"])
        return {
            "input_blocks": ui["input_blocks"],
            "output_blocks": ui["output_blocks"],
            "ca_input_blocks": ui["ca_input_blocks"] if ui["ca_enabled"] else "",
            "ca_output_blocks": ui["ca_output_blocks"] if ui["ca_enabled"] else "",
            "time_mode": ui["time_mode"],
            "start_time": float(ui["start_time"]),
            "end_time": float(ui["end_time"]),
            "ca_start_time": float(ui["ca_start_time"]),
            "ca_end_time": float(ui["ca_end_time"]),
            "ca_fadeout_start_time": fadeout if fadeout > 0.0 else None,
            "ca_fadeout_cap": float(ui["ca_fadeout_cap"]),
            "upscale_mode": ui["upscale_mode"],
            "two_stage_upscale_mode": ui["two_stage_upscale_mode"],
            "ca_upscale_mode": ui["ca_ca_upscale_mode"],
            "ca_downscale_mode": ui["ca_downscale_mode"],
            "ca_downscale_factor": float(ui["ca_downscale_factor"]),
            "ca_input_after_skip_mode": bool(ui["ca_input_after_skip_mode"]),
        }

    @staticmethod
    def _merge_yaml(settings: dict, raw: str) -> dict:
        if not raw or not raw.strip():
            return settings
        import yaml

        extra = yaml.safe_load(raw)
        if extra is None:
            return settings
        if not isinstance(extra, dict):
            raise ValueError("Extra parameters must be a YAML mapping (key: value), or empty")
        return {**settings, **extra}

    # ----------------------------------------------------------------- hooks ----

    def process(self, p, enable, *args):
        cls = NeoRAUNet
        cls.active = False
        cls.configs = []
        cls.resolved = {}

        ui = self._named_args(args)
        enable = cls._apply_xyz(bool(enable), ui)
        if not enable:
            return
        cls.resolved = ui

        params = {"RAUNet mode": ui["mode"]}
        if ui["mode"] == SIMPLE:
            params.update(
                {
                    "RAUNet model type": ui["model_type"],
                    "RAUNet res mode": ui["res_mode"],
                    "RAUNet simple upscale": ui["simple_upscale_mode"],
                    "RAUNet simple CA upscale": ui["simple_ca_upscale_mode"],
                }
            )
        else:
            params.update(
                {
                    "RAUNet input blocks": ui["input_blocks"],
                    "RAUNet output blocks": ui["output_blocks"],
                    "RAUNet time mode": ui["time_mode"],
                    "RAUNet start": ui["start_time"],
                    "RAUNet end": ui["end_time"],
                    "RAUNet upscale": ui["upscale_mode"],
                    "RAUNet CA input blocks": ui["ca_input_blocks"],
                    "RAUNet CA output blocks": ui["ca_output_blocks"],
                    "RAUNet CA start": ui["ca_start_time"],
                    "RAUNet CA end": ui["ca_end_time"],
                    "RAUNet CA downscale factor": ui["ca_downscale_factor"],
                    "RAUNet CA downscale": ui["ca_downscale_mode"],
                    "RAUNet CA upscale": ui["ca_ca_upscale_mode"],
                }
            )
            if ui["two_stage_upscale_mode"] != "disabled":
                params["RAUNet two-stage upscale"] = ui["two_stage_upscale_mode"]
            if float(ui["ca_fadeout_start_time"]) > 0.0:
                params["RAUNet CA fadeout start"] = ui["ca_fadeout_start_time"]
                params["RAUNet CA fadeout floor"] = ui["ca_fadeout_cap"]
            if ui["ca_input_after_skip_mode"]:
                params["RAUNet CA after skip"] = True

        params["RAUNet auto gate"] = bool(ui["auto_gate"])
        if not ui["auto_gate"] and float(ui["min_megapixels"]) > 0.0:
            params["RAUNet min MP"] = ui["min_megapixels"]

        p.extra_generation_params.update(params)
        NeoRAUNet.active = True

    def process_before_every_sampling(self, p, enable, *args, **kwargs):
        cls = NeoRAUNet
        if not cls.active or not cls.resolved:
            return

        ui = cls.resolved
        is_hr = bool(getattr(p, "is_hr_pass", False))
        if is_hr and not ui["apply_to_hr"]:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        unet_map, family = maps.describe(unet, getattr(p, "sd_model", None))
        if not unet_map.supported:
            logger.warning(f"RAUNet: not applied - {unet_map.reason}")
            return

        detected, source = family, "checkpoint"
        if ui["model_type"] != "auto":
            family, source = maps.ModelFamily(ui["model_type"]), "Model family dropdown"
            if detected is not None and family is not detected:
                #   The dropdown persists in ui-config.json, so a value chosen during one
                #   experiment silently governs every later run. Say so rather than quietly
                #   applying the other architecture's schedule.
                logger.warning(f"RAUNet: Model family is set to {family}, but this checkpoint looks like {detected}. Set it to 'auto' unless you mean it")

        settings = self._settings(ui, family, unet_map)
        if settings is None:
            return

        try:
            settings = self._merge_yaml(settings, ui["yaml_parameters"])
            settings["min_megapixels"] = self._gate_threshold(ui, family)
            config = Config.build(unet.get_model_object("predictor"), **settings)
            if not config.use_blocks and not config.ca_use_blocks:
                logger.info("RAUNet: no blocks selected, nothing to do")
                return
            apply_raunet(unet, config, unet_map)
        except (ValueError, KeyError, TypeError) as exc:
            logger.error(f"RAUNet: not applied - {exc}")
            return

        install_controlnet_shim()
        p.sd_model.forge_objects.unet = unet
        cls.configs.append(config)

        self._report(p, settings, config, f'{family} (from {source})', unet_map, is_hr)

    @staticmethod
    def _gate_threshold(ui: dict, family) -> float:
        if not ui["auto_gate"]:
            return float(ui["min_megapixels"])
        preset = maps.PRESETS.get(family)
        if preset is None:
            return 0.0
        return round(preset.native_megapixels * AUTO_GATE_HEADROOM, 4)

    @staticmethod
    def _report(p, settings: dict, config: Config, family, unet_map, is_hr: bool) -> None:
        pass_name = "hires pass" if is_hr else "first pass"
        blocks = ", ".join(f"{t}{i}" for t, i in sorted(config.use_blocks)) or "none"
        ca_blocks = ", ".join(f"{t}{i}" for t, i in sorted(config.ca_use_blocks)) or "none"
        logger.info(
            f"RAUNet [{pass_name}]: {family}, blocks [{blocks}] sigma {config.start_sigma:.4g}->{config.end_sigma:.4g}"
            f" | CA [{ca_blocks}] sigma {config.ca_start_sigma:.4g}->{config.ca_end_sigma:.4g} x{config.ca_downscale_factor:g} {config.ca_downscale_mode}"
            f" | skip below {config.min_megapixels:g}MP"
        )

        for note in unet_map.advise_ca(config.ca_use_blocks, after_skip=config.ca_input_after_skip_mode):
            logger.warning(f"RAUNet: {note}")

        #   The band check. This is the single most likely reason an Advanced config
        #   blotches: the scaling-block rewrite is a much bigger intervention than the
        #   cross-attention rescale, and upstream only turns it on above 2048.
        if config.use_blocks:
            width = (getattr(p, "hr_upscale_to_x", 0) if is_hr else 0) or getattr(p, "width", 0) or 0
            height = (getattr(p, "hr_upscale_to_y", 0) if is_hr else 0) or getattr(p, "height", 0) or 0
            longest = max(width, height)
            if 0 < longest <= SCALING_BLOCK_MIN_SIDE:
                logger.warning(
                    f"RAUNet: the scaling blocks are enabled at {width}x{height}, but upstream only enables them above {SCALING_BLOCK_MIN_SIDE}px; "
                    f"the 1536-{SCALING_BLOCK_MIN_SIDE} band uses cross-attention alone (that is what Simple / high does). "
                    "Blotching at this size is expected - clear Input/Output blocks and enable the cross-attention rescale instead."
                )

        sampler = (getattr(p, "hr_sampler_name", None) if is_hr else None) or getattr(p, "sampler_name", "") or ""
        if _is_forgiving_sampler(sampler):
            return
        if str(settings.get("time_mode", "percent")) != "percent":
            return

        #   Both halves of the effect stop dead at their end percent, and everything after
        #   that runs on a latent built under different geometry.  The later the cutoff and
        #   the harder the transition, the more that shows — which is the mechanism behind
        #   "it falls apart with normal samplers".  Only nag when the settings are in that
        #   territory, and say what to change.
        cutoffs = []
        if config.use_blocks:
            cutoffs.append(float(settings.get("end_time", 0.0)))
        if config.ca_use_blocks and not settings.get("ca_fadeout_start_time"):
            cutoffs.append(float(settings.get("ca_end_time", 0.0)))
        if cutoffs and max(cutoffs) >= 0.4:
            logger.info(f"RAUNet: '{sampler}' is neither a CFG++ nor a Dy sampler. Those absorb the end of the RAUNet window much better - measured on SDXL, plain Euler deforms faces and limbs where Euler CFG++ does not. If the image degrades, switch to a CFG++ variant, lower End / CA end, or set a CA fadeout start.")

    def postprocess(self, p, processed, *args):
        cls = NeoRAUNet

        #   `process` writes the infotext before anything is known about the model, so a
        #   run that turned out to be unpatchable (DiT checkpoint, invalid blocks, a
        #   no-op preset) would otherwise ship parameters claiming an effect that never
        #   ran.  Take them back out rather than lie in the PNG metadata.
        if cls.active and not cls.configs:
            for key in [k for k in p.extra_generation_params if k.startswith("RAUNet ")]:
                p.extra_generation_params.pop(key, None)

        for config in cls.configs:
            if config.hits_main == 0 and config.hits_ca == 0:
                reason = "the start/end windows excluded every step"
                if config.skips_resolution:
                    reason = f"the resolution gate skipped all {config.skips_resolution} forward passes (image is at or below native resolution)"
                logger.warning(f"RAUNet was enabled but never modified a step: {reason}")
            elif config.skips_resolution:
                logger.debug(f"RAUNet: {config.hits_main} scaling-block hits, {config.hits_ca} cross-attention hits, {config.skips_resolution} passes skipped by the resolution gate")
        cls.configs = []
        cls.active = False
        cls.resolved = {}
