# Do we need SD1.5 (or any pretrained diffusion model)?

Short answer: **no. CENT trains from scratch.** The only pretrained networks in the project sit in
*evaluation*, not in generation. This is a design consequence, not a budget compromise.

## 1. Why a pretrained pixel prior does not transfer

CENT's denoiser does not generate pixels. It generates a **token sequence**: 415 items x 8 scalars
(the control points of cubic Béziers, plus layout scalars). The image is produced afterwards by an
analytic differentiable renderer. So:

- **SD1.5's UNet** expects 4-channel 64x64 image latents, produced by a conv encoder over pixels,
  and is a 2-D convolutional network. Our denoiser is a 1-D sequence model over ~415 tokens with
  kind and character embeddings. There is no tensor on which to load SD's weights: the shapes,
  the dimensionality and the space itself differ.
- **SD1.5's VAE** exists to compress pixels into latents and decode them back. CENT never encodes a
  line image into a latent and never decodes latents into an image — the renderer is analytic. A
  VAE would have nothing to do.

The generative target is also far smaller: **3,320 numbers versus 65,536 pixels**, about 20x less
information than the image itself. This is a structured-prediction problem, closer to trajectory
modelling than to image synthesis.

## 2. What the baselines actually do (checked in the local clones)

| Baseline | Pretrained? | Evidence |
|---|---|---|
| DiffBrush | **VAE from SD1.5**; its own UNet trained from scratch | `generate.py:116` `AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")`; `:109` `unet.load_state_dict(torch.load(pretrained_model))` — the UNet is *their* checkpoint, `models/unet.py` is their own class |
| TGC-Diff | **VAE from SD3.5**, frozen; its own diffusion model | `test_ch.py:50` `AutoencoderKL.from_pretrained(vae_path, subfolder="vae")`, `:51` `requires_grad_(False)`, `:63` `model.load_state_dict(...)` |
| DiffInk | **trains its own VAE** (`train_vae_one_epoch`) then its own DiT | `train_vae_ddp.py` |
| OLHWG | **nothing pretrained** — a 1-D U-Net over trajectory points, from scratch | no `from_pretrained`/`timm`/`hub` in its model files |

So the pattern in this literature is: image-space methods reuse a pretrained **VAE** because they
need a pixel codec; the generator is theirs. The only method without an image codec (OLHWG) also
trains from scratch. CENT joins OLHWG in having no image codec at all, so there is nothing to
reuse.

## 3. Is from-scratch feasible at our scale?

Yes, and the numbers are measured, not hopeful:

- **5.69 M parameters**, 38,800 paired training lines, 242,600 steps at batch 32.
- That is ~10 M line-samples of gradient signal for a 5.7 M model — a reasonable ratio, and the
  same order as the baselines above (OLHWG trains a comparable 1-D U-Net on 52k lines).
- 10.7 min/epoch on an RTX 3080; ~17 h for 200 epochs.

The risk is not "too little data for a big model" — our model is deliberately small. The risk is
that the **style encoder** is under-trained and style collapses. That is mitigated structurally:
writer-classification auxiliary loss, k reference lines, and a validation style check on the
contact sheet.

## 4. Where pretrained weights *are* needed

| Purpose | Model | Why it must be pretrained |
|---|---|---|
| Content metric (AR/CR, D_CER/D_WER) | `Teklia/pylaia-casia-hwdb2` | A generator cannot grade its own legibility; the reader must be fixed and calibrated |
| Style metric (HWD) | VGG16 trained on handwriting text | Substituting an ImageNet net makes the number incomparable to published HWD |
| Optional perceptual loss | either of the above, frozen | only if the pixel L1 proves too weak (see §5) |

None of these enter the generator's parameters.

## 5. The one optional middle path, with a decision rule

If, after ~20 epochs, the sheets show **blurry strokes while `legib` is falling and the geometry is
right**, the pixel L1 image term is the weak link. Then add a frozen perceptual term:

```
loss += lambda_perc * || phi(pred) - phi(target) ||^2      # phi frozen, no gradient to phi
```

with `phi` = the HWD handwriting VGG16 (one model, two uses: loss and metric) or, if unavailable,
an ImageNet VGG16. Sweep `lambda_perc ∈ {0.01, 0.1}`. This changes no architecture and no
pretraining story: `phi` is a fixed feature extractor, not an initialisation.

**Decision rule:** do not add it speculatively. Add it only on the measured symptom above, and
record the run comparing against the L1-only baseline. If it is never needed, the paper is simpler
and the claim cleaner.

## 6. Consequences for the run

- **No SD1.5 / SD3.5 download for CENT.** The 319 MB SD VAE in `docs/WEIGHTS_AND_DATA.md` is for
  the *baselines* only. CENT's stack is torch + numpy; no `diffusers`, `transformers` or `timm`.
- **No `--stable_dif_path`-style configuration**, no VAE checkpoints in the ledger, no latent-space
  mismatch between training and evaluation.
- **One sentence for the paper**: "No pretrained generator is used; the decoder is analytic, and
  the only pretrained networks are the fixed recogniser and style backbone used for evaluation."
  That sentence also pre-empts the reviewer question "why not initialise from a text-to-image
  model?", whose answer is that the generative object is geometry, not pixels.

## 7. What would change this decision

If the target changed to **pixel-level line synthesis** (i.e. CENT stopped predicting geometry and
started predicting image latents), a pretrained VAE would become useful for the codec and a
pretrained UNet a plausible initialisation. That is a different paper. As long as the output is a
stroke graph with an analytic renderer, pretraining has nothing to attach to.
