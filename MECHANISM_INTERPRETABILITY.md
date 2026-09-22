# XGenAct / Arm 7 mechanism interpretability

## Mechanism claim

Arm 0 and Arm 7 have matched architecture, action supervision, masks, data, and optimization;
the changed variable is Arm 7's `depth+action`, `segmentation+action`, and `normal+action`
training mixture.  The claim to test is therefore:

> Structured generative supervision makes the shared DiT route action prediction through
> task-relevant geometry rather than incidental RGB appearance.

Attention alone cannot establish this.  Use causal intervention as the main result and attention
only as a validated illustration.

## 1. Role-aware causal deletion (implemented)

For the same episode window, cameras, prompt, sampler, and diffusion noise, compare:

1. clean RGB anchor;
2. manipulated-object pixels removed by mask-aware inpainting;
3. the exact same number of background pixels removed by the same operator.

Report action-trajectory shift from clean, position-error increase, and **causal selectivity** =
object-deletion effect − equal-area background effect.  Compare Arm 7 and Arm 0 with a paired
difference-in-differences and bootstrap interval.

Primary task: `close_microwave`, checkpoint 4000.  Existing paired closed-loop success is 17/20
for Arm 7 versus 9/20 for Arm 0, and the door has an exact simulator instance mask.  The unseen
RLBench tree has no `seg_targets.json`; the implementation therefore reads `microwave_door`
handles directly rather than incorrectly calling its base `distractor` role the target.

Current 3-episode, 50-step engineering pilot:

- trajectory selectivity: Arm 0 **0.1649 m**, Arm 7 **0.2159 m**;
- paired Arm 7 − Arm 0: **+0.0376, +0.0187, +0.0967 m**, mean **+0.0510 m**;
- position-error selectivity: Arm 0 **0.1265 m**, Arm 7 **0.1405 m**; its n=3 paired interval
  crosses zero.

This is not yet a paper-level statistical result.  Run the predeclared 20 episodes and repeat
with Telea, local-mean, and strong-blur deletion operators before making a significance claim.

Code: `scripts/mechanism_causal_roles.py`, `scripts/compare_mechanism_roles.py`, and the GPU 6/7
launcher `scripts/run_mechanism_gpu67.sh`.

## 2. Action-to-anchor attention transport (implemented)

At selected DiT blocks and diffusion noise levels, reconstruct only sampled action-query rows
from post-RoPE Q/K and aggregate their attention to the two RGB anchor frames.  Sample rows near
the ground-truth-projected action trajectory (the same post-hoc row selection for both models),
not uniformly from the mostly blank action canvas.  Report target enrichment, target/background
AUROC, and segment-level attention mass.

The required faithfulness test is the across-episode correlation between target enrichment and
causal selectivity from Experiment 1.  A 10-step, one-episode pilot did not show a consistent Arm
7 enrichment across layers/times, so the attention panel should not yet be used as evidence.

Code: `scripts/mechanism_attention_transport.py`.

## 3. `pull_cube` 3-D counterfactual equivariance

Use ManiSkill state control to translate only the cube or goal by ±2 cm and ±4 cm in x/y; re-render
RGB, depth, normals, and role segmentation while fixing robot state, cameras, prompt, and noise.
Measure response-direction cosine, response gain, non-intervened-axis leakage, and monotonicity.

Use RGB for the matched Arm 7/Arm 0 comparison, then all four observation paths for Arm 7.  A
shared geometric mechanism predicts correct and cross-modally aligned action responses.  This is
the strongest next experiment because `pull_cube` has explicit target and goal annotations.

## 4. Cross-modal action-token alignment

Run the same `pull_cube` state through RGB/depth/segmentation/normal templates.  At each block,
pool action tokens by latent time and compute cross-modal linear CKA.  The predicted signature is
low early visual alignment but increasing late action-token alignment.  Compare successful and
failed episodes and use the released initialization as an OOD control.

## 5. Activation-patching mediation

Run clean and target-deleted anchors with identical noise.  Patch clean target-region residuals
into the deleted run at one block and measure recovery of the clean action trajectory:

`1 - distance(patched, clean) / distance(deleted, clean)`.

Sweep blocks coarsely, then refine around the peak.  Use equal-area background patches and patches
from another episode as controls.  This is the strongest internal mechanism test, but should come
after deletion and 3-D equivariance establish a stable effect.

## Recommended paper figure

Use three panels: (a) clean/object-delete/matched-background anchors and world trajectories;
(b) paired Arm 0/Arm 7 causal-selectivity bars; (c) attention maps only if their enrichment
correlates with causal selectivity.  Put `pull_cube` equivariance in the quantitative table or
appendix.  Do not claim that larger raw attention by itself means a better mechanism.
