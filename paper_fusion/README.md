# paper_fusion/

ICLR-style manuscript for the **full-modality canvas** direction: all five modalities for both
views in ONE latent sequence (ten segments, 110 latent frames), plus a conditioning role that
lets a modality be withheld entirely.

Separate directory from `../paper/` on purpose. That manuscript reports `arm6`, whose thesis is
"one checkpoint serves four perception streams selected by prompt tag", and whose templates are
four segments with one visual modality each. This one argues that the prompt is *not* what
selects the stream, and changes both the canvas and the mask space. The two cannot share a
results section without one of them being misread.

| File | Purpose |
|---|---|
| `iclr2026_fusion.tex` | Main paper and appendices |
| `NUMBERS.md` | **Every quantity in the .tex, with provenance and MEASURED/PENDING status** |
| `reference.bib` | Bibliography (copied from `../paper/`) |
| `iclr.sty`, `iclr.bst`, `math.tex` | Supplied ICLR style files (copied) |

## Build

```bash
pdflatex -interaction=nonstopmode -halt-on-error iclr2026_fusion.tex
bibtex iclr2026_fusion
pdflatex iclr2026_fusion.tex && pdflatex iclr2026_fusion.tex
```

## The one rule for this directory

**No number enters the `.tex` without a MEASURED row in `NUMBERS.md`.** Anything unmeasured is
`\TODO{...}`, which renders in red and is impossible to mistake for a result while editing. The
previous manuscript used `[XX]` for the same purpose. There are currently **7** such slots.

## Scientific scope that must survive later edits

- **The central claim is NOT established.** What is established: the canvas trains, the schedule
  demonstrably fires in production, the protocol is implemented and unit-tested, and early
  evidence says anchor dominance is *not yet* broken. The mutual-completion matrix, the
  closed-loop action numbers, and the all-anchor (`F0`) control are all pending.
- **Every fusion number currently in the paper is n=1 episode, one seed, and from the BASE-tree
  arm at step 2,000 of 5,000.** It is presented as a direction indicator. The repository has a
  precedent (`ARM7_RESULTS.md`) of a headline number reversing under a properly powered re-test.
- The scaled-tree arm finished 5,000 steps (`TRAIN_EXIT=0`); its evaluation is queued and
  blocked on GPU availability, not on code.
- The base-tree arm was **stopped at step 3,202** of 5,000 to free GPUs, so the two arms are at
  unequal exposure (12,808 vs 20,000 samples). Per the current scope decision the base-tree arm
  is not a comparison target; if that changes, compare `checkpoint-3000` to `checkpoint-3000`,
  never `wide@5000` to `base@3000`.
- `normal` is analytically a function of `depth`. It is kept as the sharpest *probe* for mutual
  completion (a deterministic right answer exists), not as an independent information source. Any
  text implying it adds information is wrong.
- Action occupies 26.3% of predicted latent positions here against 50% in a four-segment
  template. This is a stated confound, not a tuned weight; do not "fix" it with per-modality loss
  weights, which would make the arm incomparable to every other.
