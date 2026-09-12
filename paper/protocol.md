% Prose for the body: everything that used to sit under Table 1 as footnotes.
% Drop wherever it reads best -- as a "Protocol" paragraph before the table, or split
% between the setup section and the discussion of each row.

\subsection{How the comparisons are set up}
\label{sec:protocol}

\paragraph{What the three columns mean.}
\textbf{U-CoGen} is one checkpoint prompted into each output space.
\textbf{In-domain specialist} is a separate model per output, warm-started from the same
released checkpoint and trained on the same data for the number of updates that output
receives inside U-CoGen; it is the matched-exposure control for whether sharing a
checkpoint costs anything. Across the seen split, the held-out variation and the unseen
tasks, all twelve paired differences have $95\%$ bootstrap intervals containing zero
(Wilcoxon $p = 0.14$--$0.79$).
\textbf{Best counterpart} is the strongest of every model we could run for that row --- seven
for depth, three for surface normals, two for segmentation --- not one representative. The
choice matters: for normals, a baseline we built by differentiating Depth Anything V2's
depth reaches $0.915$ and beats the dedicated \emph{Marigold-Normals} at $0.789$, so
reporting a single dedicated model would have understated the counterpart by $13$ points.
The full pool is \appref{app:external}.

\paragraph{Alignment, and who is allowed to fit the ground truth.}
No external depth model outputs metres. Each is fitted to the ground truth before it can be
scored --- two free parameters per frame in disparity space for Depth Anything V2, one under
median scaling for the others --- and which of those a model needs is decided from the sign of
its correlation with ground-truth depth rather than assumed. Unaligned they score $0.40$ to
$6.73$ AbsRel. U-CoGen and the specialists emit metric depth from the codec and are scored
with no free parameters. The comparison is therefore not symmetric, and it is not symmetric
in our favour: a fitted counterpart is being given information we are not.

\paragraph{What the tiers do and do not mean.}
\emph{Held-out variation} is variation~1 of the eight trained tasks, which never entered
training; \emph{unseen task} is six tasks from a tree kept out of training entirely. Both
labels are defined relative to \emph{our} training set and carry no meaning for a zero-shot
counterpart --- Depth Anything V2 scores $0.133$, $0.125$ and $0.119$ on seen, held-out and
never-trained episodes, improving as the scenes get simpler rather than degrading as the
shift grows.

\paragraph{Depth is reported as a mean, and that is the unflattering choice.}
Our depth is the only bimodal distribution in the table. On held-out variations, $36$ of
$40$ episodes fall below $0.13$ while four exceed $1.0$, so the median is $0.038$ and the
mean is ten times it; every counterpart's mean and median agree to within $10\%$ and none
has a single episode above $1.0$. We report means because moving the table to medians would
improve us tenfold and everyone else not at all. What separates us on this row is the
failure rate, not the average. On unseen tasks the distribution is much tighter --- one
episode of $66$ above $1.0$, median $0.029$ --- which we can report but not explain.

\paragraph{Two segmentation metrics, because they are not interchangeable.}
mIoU is macro-averaged over \emph{named} scene roles. Its counterpart is CLIPSeg, because
open-vocabulary segmentation takes the role name as input, which is our task; SAM cannot
name a region at all. CLIPSeg is a floor for an off-the-shelf model here rather than a
ceiling for the approach --- it sees one frame at a time, is trained on photographs rather
than renders, and role names like \emph{fixture} are ours rather than language it was
trained on. bo-IoU is class-agnostic best overlap, the only quantity defined for a model
emitting unnamed regions, and it rewards over-segmentation: SAM emits ${\approx}63$ masks
per frame against our ${\approx}4$, for ${\approx}4$ scene regions. That row is won on a
metric tilted towards the counterpart.

\paragraph{Why perception uses \tmpl{FIFI} and action does not.}
Every dense metric scores the generated output against the ground truth of the \emph{real}
episode, pixel for pixel and frame for frame, with no correspondence step. \tmpl{FIFI} is
given the real frame, so its number is conditional geometric accuracy --- the quantity the
counterparts also report. \tmpl{IIII} generates the frame it then interprets, so any
divergence between its predicted future and the real one is charged to the perception
metric. The two are separable: with the frame supplied, per-frame depth error stays flat at
$0.037$ across the clip; generating it, the same model rises from $0.003$ at the given
anchor frame to $0.196$ by frame nine and then plateaus near $0.11$. Per-episode depth
error also correlates with that episode's RGB LPIPS at $\rho = 0.75$ ($p < 10^{-4}$).
Action and RGB have no conditional analogue --- predicting the future is the task --- so those
rows use \tmpl{IIII} and their numbers include future-prediction error.
