# paper/

ICLR-style manuscript for the prompt-routed multimodal co-generation method.

| File | Purpose |
|---|---|
| `iclr2026_cogen.tex` | Main paper and appendices |
| `reference.bib` | Bibliography |
| `iclr.sty`, `iclr.bst` | Supplied ICLR style files |
| `math.tex` | Supplied notation macros |
| `iclr_v1.md` | Earlier scratch outline; not built |

## Build

With a standard TeX distribution:

```bash
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_cogen.tex
bibtex iclr2026_cogen
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_cogen.tex
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_cogen.tex
```

Tectonic can also build the complete document in one command:

```bash
tectonic --keep-logs --keep-intermediates iclr2026_cogen.tex
```

The current source has been compiled with Tectonic 0.15.0, not only checked statically. It builds
to a 10-page PDF with resolved citations/references and no overfull or underfull boxes. Tectonic
may print its own non-fatal *internal consistency* rerun warning for the supplied BibTeX style;
the generated BBL is stable across builds and a standard PDFLaTeX/BibTeX build is unaffected.

## Scientific scope that must survive later edits

- `arm6` is the principal and only experimental arm reported. Its nominal template mixture is
  `video+action@0.4,video+depth@0.2,video+segmentation@0.2,video+normal@0.2`, with
  `scene_roles` and perception mask preset `M2`.
- The current arm6 result is checkpoint 4,500 of a planned 10,000, seed 42. The route-grid
  measurements are one held-in episode. The four-episode depth check is also held-in.
- The active data split is 788 variation-0 training episodes and 240 held-out variation-1/2
  episodes (130 + 110). The held-out sweep and arm6 closed-loop evaluation are not complete.
- Depth and action are trained in separate templates. Nothing with `depth+action` or
  `video+depth+action` has been trained or evaluated.
- `f0f0` is not a training condition and is deliberately excluded. Because every target segment
  has a clean first-frame anchor during training, present evidence supports the full prompt-routed
  template interface, not a claim that text alone causes modality selection.
- Results from arm0/arm1/arm4/arm5 are not evidence for arm6 and do not appear in this paper.

## Number provenance

| Manuscript content | Stored source |
|---|---|
| Step-4,500 action/depth/segmentation/normal diagnostic | `reports/modality_mode_grid/arm6_4500/metrics.json` |
| Three additional depth episodes | `reports/modality_mode_grid/arm6_4500_ep_{push_buttons,stack_wine,turn_tap}/metrics.json` |
| Depth frozen-VAE round-trip | `reports/vae_roundtrip_depth.txt` (`scripts/vae_roundtrip_depth.py`) |
| Normal frozen-VAE round-trip | `reports/vae_roundtrip_normal.txt` |
| Depth codec-only round-trip | `python -m training.percep.depth_codec --input-glob '...' --max-files 6` |
| Scene-role frozen-VAE round-trip | `reports/vae_roundtrip_seg.txt` |
| Scene-role vs. referring-mask density | `reports/seg_pixel_stats.csv` |
| Nominal and realized template/mask probabilities | `training/templates.py`, `training/dataset/rlbench_selfgen.py` |
| Actual runtime hyperparameters | `wandb/run-20260823_204147-23qbte9z/files/config.yaml` |
| Arm6 launcher and planned step count | `scripts/train_arm.sh` |

## Reproduce arm6

From the repository root:

```bash
GPUS=1,2 bash scripts/train_arm.sh arm6
python scripts/modality_mode_grid.py --help
```

The logged arm6 run used gradient accumulation 2 through a runtime override, even though the
launcher's current inline default is 1. The manuscript reports the logged value.

## Notation conventions (`math.tex`)

`iclr2026_cogen_before_rewrite.tex` §3 defines every symbol **inline, at the equation that first
uses it** — deliberately not as a lookup table — and follows the supplied dlbook macros. Anything
reusing that section must keep these conventions, each of which removes a collision that was
present before:

| Convention | Why |
|---|---|
| $n$ = video frame index, $t$ = flow-matching time | both were `t`; they are different axes and appear in the same equations |
| $\sigma$ = Gaussian blob width only | the flow-matching noise level is written as $t$ directly (it *is* the interpolation coefficient in the scheduler), so $\sigma$ is free |
| $\mathbb{G}$ given / $\mathbb{P}$ predicted positions | was $\mathbb{C}$ / $\bar{\mathbb{C}}$: $\mathbb{C}$ reads as the complex numbers, sat next to the camera tensor $\mathsf{C}$, and the overline on $\bar{\mathbb{C}}$ is nearly invisible at 10pt |
| $\phi_{m,v}$ = plan flag | was $g_{m,v}$, colliding with gripper openness $g_n$ |
| tensors `\tZ \tA \tU \tC`, elements `\etA`, vectors `\vp \vq \vx`, sets `\sG \sP \sM` | the dlbook table reproduced in `iclr_v1.md` |

**Font requirement.** `math.tex` defines tensors via
`\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}`, and `times.sty` points
`\sfdefault` at Helvetica and `\ttdefault` at Courier. On a TeX installation without those
metrics, every tensor symbol fails with `\textfont 9 is undefined (character Z)` and the build
also errors on `phvb` / `pcrr7t`. Install `psnfss helvetic courier` (TeX Live `scheme-small` does
not include them). This is an environment failure, not a source error.

## Build status of `iclr2026_cogen_before_rewrite.tex`

Compiled with pdfTeX 3.141592653 (TeX Live 2026) + BibTeX, four passes: **0 errors, 0 undefined
references or citations, 0 overfull/underfull boxes**, 13 pages total. Output kept alongside as
`iclr2026_cogen_before_rewrite.pdf`.

Two caveats:

- **Main text runs to ~10.3 pages**, over the ICLR 9-page cap (Conclusion on page 10, references
  start on page 11). Cut in this order: fold §4.5 to its three bold lead-ins; drop the `FIII`
  column from the depth table; move the codec paragraphs of §4.1 wholly into Appendix A.
- `reference.bib` was missing `barron2019general`, cited by the depth codec; it has been restored.
