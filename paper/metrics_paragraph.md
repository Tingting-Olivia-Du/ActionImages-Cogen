%% ---------------- BODY ----------------

\paragraph{Metrics and protocol.}
Depth is scored by absolute relative error (AbsRel), surface normals by mean cosine
similarity, predicted RGB by LPIPS \citep{zhang2018lpips}, and actions by median
end-effector position error, median geodesic rotation error and gripper accuracy.
Segmentation carries two numbers: \textbf{mIoU}, macro-averaged over the \emph{named} scene
roles present in an episode, and \textbf{bo-IoU}, class-agnostic best overlap, which is the
only quantity defined for a model that emits unnamed regions --- and which rewards
over-segmentation, so it favours a counterpart producing many masks.
Perception is evaluated with the \tmpl{FIFI} plan, which supplies the real frame as every
external counterpart receives; action and RGB use \tmpl{IIII}, where predicting the future
is the task. Closed-loop control uses receding-horizon replanning with motion-planned
end-effector poses, capped at $1.5\times$ the demonstration length. Each model runs the
sampling configuration it was trained under rather than a shared one, since a checkpoint
run off its training stride or guidance scale is being evaluated out of distribution.
Paired comparisons use an exact McNemar test for closed-loop success, which is binary, and
a Wilcoxon signed-rank test with percentile-bootstrap intervals for the continuous offline
metrics. \appref{app:protocol} gives the settings and the reasoning.


%% ---------------- APPENDIX ----------------

\section{Evaluation protocol}
\label{app:protocol}

\paragraph{Why segmentation needs two metrics.}
They answer different questions and neither subsumes the other. mIoU asks whether the model
put the right label on the right region; it is undefined for SAM, which emits regions
without names. bo-IoU matches each ground-truth region to whichever predicted mask covers it
best and ignores labels, so it asks only whether the region was found. The two disagree in a
predictable direction: more masks can only improve a best-match score, so bo-IoU flatters a
model that over-segments. SAM emits ${\approx}63$ masks per frame against our ${\approx}4$,
for ${\approx}4$ scene regions.

\paragraph{Action metrics use medians.}
A single mis-decoded frame produces a pose error large enough to dominate a clip mean, so
the per-clip statistic is the median over frames. Means are recorded alongside in the
released per-episode files.

\paragraph{Sampling configuration is per-model, not shared.}
Fine-tuned models use frame stride 3, tagged prompts and classifier-free guidance $7.5$; the
released initialization uses stride 4, its original untagged prompts and guidance $10.0$.
Guidance scale extrapolates between the conditional and unconditional predictions, trading
diversity for fidelity. Stride is not a free parameter: a checkpoint emits one pose per
stride native steps because that is the spacing it was trained on, and running it at another
stride misaligns its output from what it learned. The consequence is an asymmetry we cannot
remove and therefore state --- against the native $20$\,Hz control rate, our models plan at
$6.7$\,Hz covering $6.2$\,s per generation while the released checkpoint plans at $5$\,Hz
covering $8.2$\,s, so it replans less often over a longer horizon at coarser resolution. The
step budget is scaled by stride for the same reason; budgeting in native units would hand a
stride-3 policy three times the intended wall clock.

\paragraph{Two paired tests, for two kinds of outcome.}
Closed-loop success is binary and every model sees the same scene seeds, so arms are
compared with an \emph{exact} McNemar test on the discordant pairs --- the trials where one
arm succeeded and the other did not. The exact binomial form is used rather than the
$\chi^2$ approximation because discordant counts here are often single digits. Offline
metrics are continuous and paired by episode, so they use a Wilcoxon signed-rank test, which
ranks the per-episode differences by magnitude and asks whether positive and negative ranks
balance. Ranking rather than averaging matters here: our depth distribution is bimodal, and
a handful of catastrophic episodes would otherwise decide a mean-based test. Percentile
bootstrap intervals on the mean paired difference are reported alongside so the effect size
is visible, not only its significance.
