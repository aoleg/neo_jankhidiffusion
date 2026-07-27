"""X/Y/Z Plot axes for RAUNet.

Axis callbacks fire before `process_before_every_sampling`, so they write into a dict the
script reads there.  Registration is best-effort: X/Y/Z Plot is a built-in script but a
user can disable it, and losing the grid axes must not take the extension down with it.
"""

from __future__ import annotations

from modules import scripts

_registered = False


def _grid_module():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("scripts.xyz_grid", "xyz_grid.py") and hasattr(data, "module"):
            return data.module
    return None


def register(cache: dict, upscale_methods, downscale_methods, res_modes, families) -> None:
    global _registered
    if _registered:
        return

    xyz_grid = _grid_module()
    if xyz_grid is None:
        return

    def apply_field(field):
        def _(p, x, xs):
            cache[field] = x

        return _

    xyz_grid.axis_options.extend(
        [
            xyz_grid.AxisOption("[RAUNet] Enable", str, apply_field("enable"), choices=xyz_grid.boolean_choice(reverse=True)),
            xyz_grid.AxisOption("[RAUNet] Mode", str, apply_field("mode"), choices=lambda: ["Simple", "Advanced"]),
            xyz_grid.AxisOption("[RAUNet] Model family", str, apply_field("model_type"), choices=lambda: [*families, "auto"]),
            xyz_grid.AxisOption("[RAUNet] Resolution mode", str, apply_field("res_mode"), choices=lambda: list(res_modes)),
            xyz_grid.AxisOption("[RAUNet] Input blocks", str, apply_field("input_blocks")),
            xyz_grid.AxisOption("[RAUNet] Output blocks", str, apply_field("output_blocks")),
            xyz_grid.AxisOption("[RAUNet] Start", float, apply_field("start_time")),
            xyz_grid.AxisOption("[RAUNet] End", float, apply_field("end_time")),
            xyz_grid.AxisOption("[RAUNet] Upscale mode", str, apply_field("upscale_mode"), choices=lambda: list(upscale_methods)),
            xyz_grid.AxisOption("[RAUNet] Two-stage upscale", str, apply_field("two_stage_upscale_mode"), choices=lambda: ["disabled", *upscale_methods]),
            xyz_grid.AxisOption("[RAUNet] CA input blocks", str, apply_field("ca_input_blocks")),
            xyz_grid.AxisOption("[RAUNet] CA output blocks", str, apply_field("ca_output_blocks")),
            xyz_grid.AxisOption("[RAUNet] CA start", float, apply_field("ca_start_time")),
            xyz_grid.AxisOption("[RAUNet] CA end", float, apply_field("ca_end_time")),
            xyz_grid.AxisOption("[RAUNet] CA fadeout start", float, apply_field("ca_fadeout_start_time")),
            xyz_grid.AxisOption("[RAUNet] CA fadeout cap", float, apply_field("ca_fadeout_cap")),
            xyz_grid.AxisOption("[RAUNet] CA downscale factor", float, apply_field("ca_downscale_factor")),
            xyz_grid.AxisOption("[RAUNet] CA downscale mode", str, apply_field("ca_downscale_mode"), choices=lambda: list(downscale_methods)),
            xyz_grid.AxisOption("[RAUNet] CA upscale mode", str, apply_field("ca_upscale_mode"), choices=lambda: list(upscale_methods)),
            xyz_grid.AxisOption("[RAUNet] Min megapixels", float, apply_field("min_megapixels")),
        ]
    )

    _registered = True


__all__ = ("register",)
