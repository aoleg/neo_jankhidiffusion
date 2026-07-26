"""ControlNet compatibility shim.

ControlNet hints are computed against the *unmodified* UNet geometry, so once RAUNet has
changed the resolution of a feature map the two no longer line up.  Neo's stock
`apply_control` (`backend/nn/unet.py:46`) catches the resulting broadcast error and
prints a warning, which means the hint is silently dropped for every block RAUNet
touched.  Upstream's answer, kept here, is to resize the hint to the tensor it is being
added to.

The replacement is behaviour-preserving when RAUNet is off: identical shapes take the
same `h += ctrl` path they always did.  It is installed on first use and left in place —
`apply_control` is resolved from module globals at call time, so patching the module
attribute is enough, but there is no safe moment to *remove* it (another generation may
be mid-flight), and leaving a strictly-more-tolerant version installed costs nothing.

Set `JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND=1` in the environment to opt out.
"""

from __future__ import annotations

import logging
import os

import torch.nn.functional as F

logger = logging.getLogger(__name__)

SCALE_ARGS = {"mode": "bilinear", "align_corners": False}

_state = {"patched": False, "original": None, "warned": False}


def _hd_apply_control(h, control, name):
    ctrls = control.get(name) if control is not None else None
    if not ctrls:
        return h
    ctrl = ctrls.pop()
    if ctrl is None:
        return h
    if ctrl.shape[-2:] != h.shape[-2:]:
        if not _state["warned"]:
            _state["warned"] = True
            logger.info(f"RAUNet: rescaling ControlNet conditioning {tuple(ctrl.shape[-2:])} -> {tuple(h.shape[-2:])}")
        ctrl = F.interpolate(ctrl, size=h.shape[-2:], **SCALE_ARGS)
    h += ctrl
    return h


def install() -> bool:
    """Patch `backend.nn.unet.apply_control`. Returns True if it is now active."""

    if os.environ.get("JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND"):
        return False
    if _state["patched"]:
        return True

    try:
        from backend.nn import unet as unet_module
    except ImportError:  # pragma: no cover - offline harness
        return False

    if unet_module.apply_control is _hd_apply_control:
        _state["patched"] = True
        return True

    _state["original"] = unet_module.apply_control
    unet_module.apply_control = _hd_apply_control
    _state["patched"] = True
    logger.info("RAUNet: patched backend.nn.unet.apply_control to rescale mismatched ControlNet hints")
    return True


def uninstall() -> None:
    """Restore the stock `apply_control`. Only used by the tests."""

    if not _state["patched"]:
        return
    try:
        from backend.nn import unet as unet_module
    except ImportError:  # pragma: no cover
        return
    if _state["original"] is not None:
        unet_module.apply_control = _state["original"]
    _state.update({"patched": False, "original": None, "warned": False})


__all__ = ("install", "uninstall")
