# RAUNet for WebUI Forge Neo

A port of [blepping's `comfyui_jankhidiffusion`](https://github.com/blepping/comfyui_jankhidiffusion)
(RAUNet, from [HiDiffusion](https://hidiffusion.github.io/)) to
[sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) (Neo branch).

RAUNet lets a UNet model generate well above the resolution it was trained for with far
fewer duplicated subjects, repeated limbs and general mush. It does this by temporarily
changing where the UNet's resolution pyramid sits, for the first part of sampling only.

**Supports SD 1.x, SD 2.x and SDXL.** It does not support DiT models (Flux, Qwen, Chroma,
Krea 2) — see [Other model families](#other-model-families).

Human here. **Current status:** finally usable. Best results with https://github.com/aoleg/Neo_ExtraSchedulers
Euler Dy CFG++/Euler SMEA Dy CFG++ samplers, but other samplers work fine in both
Simple and Advanced modes. For non-Dy samplers, the Advanced mode has the "churn" setting.
Adjust it accordingly: bump to increase detail, reduce if the image starts looking blotched or
faces/limbs collapse. In a sense, this now perfectly matches the reForge implementation.

The reForge implementation was working there because of two bugs in two different places;
Claude will tell you exactly what they were. When implemented properly, RAUNet is all but
unusable. However, the two bugs helped me find the right direction and make this extension
usable with literally any sampler/scheduler combo. Clean hi-resolution images in one go.

Now back to Claude.

## Install

Clone into `extensions/`:

```bash
git clone https://github.com/<you>/neo_jankhidiffusion extensions/neo_jankhidiffusion
```

Restart the webui. The panel appears as **RAUNet (Neo)** in txt2img and img2img.

## What it actually does

Two independent effects, each with its own start/end window:

**Scaling blocks.** One `Downsample` block on the way down runs its 3×3 convolution with
stride 4 / dilation 2 / padding 2 instead of stride 2, halving the feature map a second
time; the mirrored `Upsample` block compensates with a 4× interpolation. In between, the
deep layers of the UNet see feature maps the size they were trained on. This is the big,
structural half of the effect.

**Cross-attention.** A shallower pair of blocks has its hidden state pooled down before
attention and scaled back up afterwards. Gentler, and often the only half you want.

Both stop at their end percentage, after which the model runs completely unmodified.

## Quick start

Leave **Mode** on *Simple*, set **Resolution mode** to the band you are generating in, and
generate.

**Model family** defaults to SDXL (set it to `auto` to read it from the loaded
checkpoint) and only chooses the *timing*. The block numbers always
come from the model that is loaded — read off its structure at runtime, not looked up in a
table — so a wrong family costs you a suboptimal schedule, never a broken configuration.
`auto` is still available and still detects correctly. When the choice disagrees with the
checkpoint the log says so, and every run reports which source the family came from:

```
RAUNet [first pass]: SDXL (from checkpoint), blocks [input3, output5] …
```

> **Upgrading from an earlier version?** Forge pins every UI default into
> `ui-config.json` the first time a script loads, and the saved value then wins over the
> code. Fields whose defaults changed have been renamed (`Input blocks` →
> `Scaling input blocks`, and so on) so the new defaults take effect. If any field still
> shows a stale value, delete its `customscript/neo_raunet.py/...` lines from
> `ui-config.json` and restart.

| Model | Resolution mode | What the preset does |
|---|---|---|
| SD1.5 | low | scaling blocks 0–40% |
| SD1.5 | high | scaling blocks 0–50%, cross-attention 0–35% |
| SD1.5 | ultra | scaling blocks 0–60%, cross-attention 0–45% |
| SDXL | low | **nothing** — 1024×1024 is native, there is nothing to fix |
| SDXL | high | cross-attention 0–50% only |
| SDXL | ultra | scaling blocks 0–45%, cross-attention 0–60% |

Resolution mode is a preset band, not a check against your actual size — picking `ultra`
at 1536² is a legitimate thing to do if you want a stronger effect.

### The two halves, and which one to reach for on SDXL

The scaling-block rewrite and the cross-attention rescale are independent, and on SDXL
**which one you want depends on the resolution band.** Upstream's presets encode this and
they turn out to be right:

| Band | scaling blocks | cross-attention |
|---|---|---|
| ≤ 1024 | off | off — nothing to correct |
| 1536–2048 (`high`) | **off** | 0–50%, blocks 4/5 |
| over 2048 (`ultra`) | 0–45%, blocks 3/5 | 0–60% |

Measured on SDXL at 1792px, 40 steps, Beta, same seed:

| Setting | Result |
|---|---|
| cross-attention only, blocks 4/5, 0–50% (`Simple / high`) | clean on Euler CFG++ |
| scaling blocks 3/5, 0–45%, CA off | blotched — "noise" objects over the picture |
| scaling blocks 3/5 plus CA | the same blotches, plus deformities |

The scaling-block rewrite is by far the bigger intervention — it changes the resolution
the whole lower half of the UNet operates at — and at 1792px that is more correction than
the image needs. **If an Advanced config blotches, clearing the scaling blocks is the
first thing to try.** The extension now warns when they are enabled at or below 2048px.

The `sd-forge-extra-samplers` Euler Dy and Euler SMEA Dy hide this, which is worth knowing
because it can send you chasing the wrong variable: both re-noise the latent every step
(see [Samplers](#samplers)), so blotches get smoothed away along with fine detail. A
config that only looks clean under those two is not clean.

Within the cross-attention half, *where* matters as much as whether: block 2 rescales the
latent at full resolution, block 4 rescales it after two downsamples. Upstream's SDXL
choice is 4/5, the deep one; the reForge fork's SDXL default is 2/7, the shallow one, and
that is blotchy in its own right. The Advanced tab ships with cross-attention **off**
behind a closed accordion — turn it on deliberately, with the deep blocks.

> **An earlier version of this file said the opposite** — that upstream was wrong to
> disable the scaling blocks for SDXL `high`. That was based on a single generation with
> Euler SMEA Dy, whose re-noising was masking exactly the blotching described above.
> Upstream's preset was right.

#### Why reForge's Advanced tab looked better

The reForge port defaults to scaling blocks 0–0.45 with cross-attention off, and that
combination is reported to work well there — while the same settings blotch here on every
sampler except Euler Dy and Euler SMEA Dy. The scaling-block code is not the difference:
the downsample rewrite in this port is bit-identical to reForge's and ComfyUI's (same
weights, stride 4 / dilation 2 / padding 2, verified with `torch.equal`), and the upsample
path is the same interpolate-then-conv.

The difference is in the **samplers**. reForge's `sample_euler_dy_cfg_pp`
(`ldm_patched/k_diffusion/sampling.py`) computes

```python
gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
if s_dy_pow >= 0:                       # default -1.0, so this never runs
    gamma = gamma * (1.0 - (i / (len(sigmas) - 2)) ** s_dy_pow)
```

With its shipped defaults (`s_churn = 0`, `s_dy_pow = -1`) that pins gamma at **0.414 on
every step** — the sampler re-noises the latent to `sigma * 1.414` each step and denoises
it again. That is the "severely noisy image resolved in the last steps" you see in the
live preview, and it is what anneals the scaling-block artifacts away.

reForge's Advanced tab therefore never looked good *because RAUNet behaved better there*.
It looked good because its Dy samplers churn hard by default, and reForge blotches on
non-Dy samplers for the same reason ours does.

**So the scaling-block half needs a churning sampler, in either webui.** There are two
ways to get one:

* **`Euler Dy CFG++ (reForge)` / `Euler SMEA Dy CFG++ (reForge)`** in
  [Neo_ExtraSchedulers](https://github.com/aoleg/Neo_ExtraSchedulers) — the reForge
  behaviour reproduced as separate samplers, gamma pinned at 0.4142 with no configuration.
  The plain `Euler Dy CFG++` / `Euler SMEA Dy CFG++` keep `min` and stay sharper.
* **The `Sampler churn (s_churn)` slider** in this extension's Advanced tab, which works
  with any sampler that accepts the parameter. See below.

### Sampler churn (s_churn)

Advanced → **Sampler churn (s_churn)**. `0` (the default) leaves your sampler completely
alone; above 0, RAUNet sets `p.s_churn` for the run, so the sampler re-noises the latent to
`sigma × (gamma + 1)` before each step and denoises it again. That is the same mechanism
the reForge Dy samplers use, made available to any sampler that takes it.

Gamma is `min(s_churn / (steps - 1), 0.4142)`, so at 40 steps you need `s_churn ≈ 16.6` to
reach reForge's 0.4142. Start lower — a little churn goes a long way, and it trades fine
detail for smoothness.

Three things to know:

* **It only reaches samplers that declare `s_churn`.** Euler, Heun, DPM2 and the CFG++/Dy
  samplers do; DPM++ 2M and the ancestral solvers generally do not. If yours does not, the
  log says so and nothing is changed.
* **The global setting wins.** Forge's Settings → *sigma churn* is applied after this and
  overrides it whenever it is non-zero (`sd_samplers_common.py:477-485`). Leave it at 0 —
  the log warns if it is not.
* **It composes with the (reForge) samplers rather than conflicting.** Those compute
  `max(s_churn / (steps - 1), 0.4142)`, so the slider can only raise gamma above their
  floor; setting it to 0 leaves them exactly as they are.

## Samplers

**Use a CFG++ sampler.** That is the short version, and it is measured rather than
theorised — SDXL, 1792px, 40 steps, Beta schedule, `Simple / SDXL / high`
(cross-attention 0–50%), same seed throughout:

| Sampler | Result |
|---|---|
| Euler | deformed faces and limbs |
| Euler CFG++ | clean |
| Euler Dy CFG++ | clean, and near-identical to Euler CFG++ |
| Euler Dy | clean, but see the caveat below |

CFG++ samplers steer along the *unconditional* prediction instead of the CFG-amplified
one, so the resampling artifacts RAUNet introduces do not get multiplied by the guidance
scale. That is what carries the result here.

The Dy sub-step turns out to contribute very little. `dy_sampling_step` fires on
`if i // 2 == 1`, which at 40 steps means steps 2 and 3 — twice, out of forty. That is why
Euler CFG++ and Euler Dy CFG++ land in the same place, and it is why an earlier version of
this README was wrong to credit the half-resolution sub-step with the improvement.

**Caveat on plain Euler Dy.** It looks clean, but `sd-forge-extra-samplers` computes
`gamma = max(s_churn / (len(sigmas) - 1), 2**0.5 - 1)` — `max`, where every other sampler
uses `min`. With the default `s_churn = 0` that pins gamma at 0.414 instead of 0, so it
re-noises the latent at *every* step. The visible signature is exactly what you would
expect: clean output, noticeably less fine detail, and a composition that diverges from
every other sampler at the same seed. It is smoothing artifacts away rather than avoiding
them. Prefer **Euler Dy CFG++**, which uses `min` and is unaffected.

Rather than build sampler logic into this extension, the other fixes live where the
problem is:

* **CA fadeout start** (Advanced → Cross-attention). Tapers the downscale factor smoothly
  from that point to CA end instead of cutting it off. This is the single most effective
  setting if the image degrades as the effect ends, and it is the main thing the older
  reForge port did not have. Try a fadeout start around 60–70% of the way through the CA
  window.
* **Lower End / CA end.** A cutoff at 45% is much easier to absorb than one at 60%.
* **The resolution gate** (below) keeps RAUNet out of the Dy samplers' rescaled sub-steps,
  where it would otherwise be correcting a latent that is already at native resolution.

If you enable RAUNet with a late cutoff on a sampler that is neither CFG++ nor Dy, the log
says so once, with these suggestions. It never changes your settings.

## The resolution gate

`Skip at or below native resolution` is checked by default and has no upstream equivalent
— it exists because a webui is not ComfyUI.

In ComfyUI you wire RAUNet into one KSampler and it affects that sampler only. In Forge the
patched UNet is used by every pass of the run, so with **Hires. fix** on, an unguarded
RAUNet also runs during the low-resolution *first* pass — the pass that does not need it
and is actively harmed by it. The gate measures the latent at the start of each forward and
skips the whole effect when the image is at or below the model's native resolution
(SDXL 1.05 MP × 1.1 headroom; SD1.5 0.26 MP × 1.1).

Consequences worth knowing:

* With Hires. fix, RAUNet effectively becomes "hires pass only" without you configuring
  anything, which is almost always what you want.
* During a Dy sampler's half-resolution sub-step, the gate suppresses RAUNet. This is
  correct — that sub-step is already at native scale — but it is a behaviour change versus
  the reForge port. Uncheck the box to get the old behaviour.
* If the gate suppresses *every* pass, the log says so at the end of the generation rather
  than leaving you wondering why nothing changed.

Uncheck it to use the manual **…or skip below (MP)** slider instead; 0 there means "always
apply".

## Advanced settings

**Input blocks / Output blocks** — the scaling-block pair, comma separated. They must be
paired or the UNet's skip connections will not line up; the extension refuses obviously
wrong combinations with an explanation rather than letting torch fail deep inside a
forward pass.

| Model | valid pairs |
|---|---|
| SD1.5 / SD2.x | 3↔8, 6↔5, 9↔2 |
| SDXL | 3↔5, 6↔2 |

HiDiffusion's own SDXL setting is 6/2; 3/5 is the gentler one and is what the presets use.

**Time mode** — `percent` (of the step progression), `timestep` (0–999, inverted) or
`sigma` (raw). The sliders re-range themselves when you switch. Use percent unless you have
a reason not to.

**Upscale mode** — how the Upsample block interpolates. `bicubic` or `bislerp`.

**Two-stage upscale** — do half the upscale with a second method first. Different, not
necessarily better; upstream defaults it off and so does this.

**Enable the cross-attention rescale** — off by default; see
[the two halves](#the-two-halves-and-which-one-to-reach-for-on-sdxl) for why.

**CA input / output blocks** — the cross-attention pair, and a different rule from the
scaling blocks: `output = block count - input`.

| Model | valid CA pairs |
|---|---|
| SD1.5 / SD2.x (12 blocks) | 1↔11, 2↔10, 4↔8, 5↔7 |
| SDXL (9 blocks) | 4↔5, 5↔4, 7↔2, 8↔1 |

Deeper is gentler: SDXL 4/5 acts after two downsamples, SDXL 2/7 acts on the latent at
full resolution. The pairing shifts in by one in after-skip mode. Getting it wrong does
not degrade the image — it raises a `torch.cat` size error several frames inside the
UNet's forward pass — so the extension refuses the run before sampling starts and names
the value to change. Note that upstream's *default* CA pair, 4/8, is an **SD1.5** pairing;
on SDXL the partner of 4 is 5.

**CA downscale factor / mode** — 2.0 means half size. `avg_pool2d` is stock HiDiffusion
and only accepts whole numbers; `adaptive_avg_pool2d` matches it for whole numbers and also
allows fractional factors, which is how you get an effect gentler than "half".

**CA fadeout start / floor** — see [Samplers](#samplers). 0 disables the fade.

**Patch input blocks after the skip connection** — moves the cross-attention downscale to
the far side of the skip push, so the skip keeps full resolution. Changes the effect
noticeably, and shifts the CA block pairing by one.

**Extra parameters (YAML)** — a mapping that overrides any field of the internal `Config`
by name. Useful ones: `pre_upscale_multiplier`, `post_upscale_multiplier`,
`pre_downscale_multiplier`, `post_downscale_multiplier` and their `ca_` counterparts,
`ca_downscale_factor_w` (a separate horizontal factor), `ca_avg_pool2d_ceil_mode`,
`ca_latent_pixel_increment`, `verbose: true`.

```yaml
ca_downscale_factor_w: 1.5
ca_post_upscale_multiplier: 1.02
verbose: true
```

There is very little error checking on this path — it is the escape hatch, by design.

## ControlNet

ControlNet hints are computed against unmodified UNet geometry, so once RAUNet has changed
a feature map's resolution they no longer line up. Neo's stock `apply_control` catches the
resulting error, prints a warning and drops the hint. On first use this extension replaces
it with a version that resizes the hint instead. The replacement is behaviour-identical
when shapes already match, and is left installed afterwards. Set
`JANKHIDIFFUSION_NO_CONTROLNET_WORKAROUND=1` in the environment to opt out.

## X/Y/Z Plot

Twenty-one axes are registered under `[RAUNet] …` — Enable, Mode, Model family, Resolution mode,
the block lists, all the time windows, the fadeout, the downscale factor and mode, both
upscale modes, and Min megapixels.

## Other model families

`inspect_unet` walks the loaded model rather than trusting a hard-coded table, so an
unsupported checkpoint gets a clear log line instead of a crash:

```
RAUNet: not applied - SingleStreamDiT is not a UNet (no input_blocks/output_blocks);
RAUNet needs a down/up resolution pyramid
```

**Krea 2 specifically** is a `SingleStreamDiT` (`backend/nn/krea.py`) — a flat stack of
single-stream transformer blocks with no resolution pyramid, no `Downsample`, and no
`input_block_patch` hook. RAUNet as an algorithm has nothing to attach to there; it would
need a different method rather than a new preset, and pretending otherwise would be worse
than declining.

The code is laid out so that a family that *does* have a down/up pyramid is additive work:

1. teach `lib_jankhidiffusion/unet_map.py::inspect_unet` to enumerate that architecture's
   scaling blocks;
2. add a `Preset` row to `PRESETS` with its native resolution and block pairing;
3. add its name to `ModelFamily`.

`raunet.py` never names a family — it consumes a `UNetMap` and a settings dict — so it
should not need changes.

## Tests

An offline harness covers the maths, the block discovery, the patches end-to-end through a
miniature SDXL-shaped UNet, and the script's argument plumbing under stubbed `modules` /
`gradio`. No GPU, no checkpoint, no running webui:

```bash
venv/Scripts/python.exe tests/harness.py
```

## Layout

```
lib_jankhidiffusion/
  utils.py       time windows, block lists, rescaling
  unet_map.py    runtime block discovery, pairing rules, per-family presets
  raunet.py      the patches
  controlnet.py  ControlNet hint-rescaling shim
  xyz.py         X/Y/Z Plot axes
scripts/
  neo_raunet.py  the Forge script
tests/
  harness.py
```

## Credits

* [blepping](https://github.com/blepping) — `comfyui_jankhidiffusion`, the implementation
  this is ported from.
* The HiDiffusion authors — the [original method](https://hidiffusion.github.io/).
* `reforge_jankhidiffusion` — the earlier webui port, whose panel layout this borrows.
