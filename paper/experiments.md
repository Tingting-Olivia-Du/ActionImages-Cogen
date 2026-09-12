\section{Experiments}
\label{sec:experiments}

We ask three questions.
First, can one checkpoint support action, depth, scene-role segmentation, and
surface normals through a shared template-conditioned interface while
approaching modality-specific specialists under matched training exposure?
Second, does unified fine-tuning preserve the world-action capability of the
pretrained model, and what is the remaining trade-off relative to an
action-only specialist?
Third, how do perception and control generalize beyond the fine-tuning
distribution?


\subsection{Experimental setup}
\label{sec:exp_setup}

\paragraph{Data.}
We use a self-generated RLBench tree \citep{james2020rlbench} rendered at $512^2$ with
Colosseum-style domain randomization \citep{pumacay2024colosseum}, carrying four camera
views, per-frame metric depth, and instance masks mapped to the scene-role vocabulary of
\secref{sec:codecs}. Training uses variation~0 of 16 tasks (788 episodes); variations~1
and~2 of those tasks (240 episodes) are held out, which changes task horizon or reference object. For example,  variation~1 of \texttt{push\_buttons} asks for two button presses where variation~0 asks for one. Also, a separate tree of 6 tasks with variation 0 that appear nowhere in training (72 episodes) is kept outside it as unseen tasks in evaluation. Appendix ~\ref{app:data} gives the per-task counts. Windows are 41 frames at stride 3 --- $6.0$\,s at $6.7$\,Hz --- with two of four views per
example. 

\paragraph{Models and training.}
All fine-tuned models start from the same released Action Images checkpoint
\citep{zhen2026action}, which we also report unadapted as the \emph{initialization}. We
train one unified model and four single-modality specialists --- action, depth, segmentation
and surface normal. Each specialist samples only its own template; the unified model samples
\tmpl{video+action} with probability $0.4$ and each of \tmpl{video+depth},
\tmpl{video+segmentation} and \tmpl{video+normal} with probability $0.2$. We match specialists by \emph{modality exposure}. Over $T=10{,}000$ steps, the unified model receives $4{,}000$ action updates and $2{,}000$ updates per perception modality; specialists are therefore evaluated at the same respective counts. This controls for supervision budget and isolates the cost of sharing one checkpoint.

Backbone, frozen
VAE, objective, optimizer, data tree and conditioning distribution are identical across every
fine-tuned model, so the template distribution is the only variable. We fine-tune all
parameters with ZeRO-2 and CPU-offloaded AdamW in bf16, batch size 1 per device on two GPUs,
learning rate $5\times10^{-7}$ with $1{,}000$ warmup steps and a constant schedule thereafter, gradient clipping at $1.0$ and no accumulation.




\paragraph{Metrics and protocol.}
Depth is scored by absolute relative error (AbsRel), segmentation by
macro-mIoU over roles present in the scene, normals by mean cosine similarity,
and actions by median end-effector position error, median geodesic rotation
error, and gripper accuracy.
Predicted RGB futures are scored by LPIPS.
Unless otherwise stated, evaluation uses the \texttt{IIII} conditioning plan,
in which only the first latent frame of each stream is observed.

Closed-loop success is evaluated with receding-horizon replanning and
motion-planned end-effector control, capped at $1.5\times$ the demonstration
length.
Closed-loop comparisons use identical scene seeds and exact McNemar tests.
Offline comparisons on shared episodes use paired Wilcoxon signed-rank tests,
with percentile-bootstrap intervals on mean paired differences where
appropriate.
Each model is evaluated with its training-time rollout configuration:
fine-tuned models use stride 3, tagged prompts, and CFG 7.5, whereas the
released initialization uses stride 4, its original untagged prompts, and
CFG 10.


\subsection{One model matches specialists across four outputs}
\label{sec:exp_unified}

Our primary experiment asks whether heterogeneous robot outputs can share one
parameter set, output head, and objective without a large interference
penalty.
Table~\ref{tab:main} compares the unified model with four
single-modality specialists at matched modality-specific exposure. \todo{and add more figure illustration, only 2 tables will be enough, why 0.385}


\input{Tables/tab_main}

\todo{size}
As shown in Table~\ref{tab:main}, the unified model is nominally better
on action position and surface normals. While the depth specialist and segmentation specialist outperform in it's corresponding skill when trained separately.

The absolute differences are small, and none of the four paired comparisons is statistically significant ($p=0.20$--$0.84$).

Table~\ref{tab:main} therefore shows all four output types can coexist in one model without the substantial degradation expected if they strongly competed for the shared parameters or output head.


Figure~\ref{fig:qualitative} complements Table~\ref{tab:main} by showing
that these quantitative capabilities reside in the same model.
For the same episode, the unified model produces action, depth, segmentation,
and surface-normal futures by instantiating the corresponding output template
with its modality tag and modality-specific initial anchor.
The model parameters and output head remain unchanged across all four rows.


\subsection{Unified training preserves pretrained world-action capability}
\todo{closed loop eval here}
\label{sec:exp_preservation}

We next evaluate whether adding three dense perception outputs compromises
the world-action capability inherited from pretraining.
Table~\ref{tab:wam_preservation} compares the unified model against two
references: the released initialization before adaptation to our data and an
action specialist whose continued fine-tuning is devoted entirely to
video+action.

\input{Tables/action_eval}


\paragraph{Preservation relative to the pretrained model.}
Table~\ref{tab:wam_preservation} shows that unified fine-tuning improves every
reported world-action metric relative to the initialization.


Closed-loop success on the common paired campaign also increases from
$30.0\%$ to $43.8\%$.
On the complete $n=100$ grid, the corresponding initialization and unified
scores are $24\%$ and $35\%$.

Thus, the unified model does not erase the pretrained world-action capability.
It retains and adapts that capability while adding metric depth, dense
scene-role segmentation, and surface-normal generation.

\paragraph{Specialization trade-off.}
Table~\ref{tab:wam_preservation} also shows that dedicating continued
fine-tuning entirely to action yields a stronger closed-loop point estimate.
The action specialist reaches $57.5\%$ success compared with $43.8\%$ for the
unified model, which shows a
specialization trade-off.
The unified model improves over its initialization and gains
three additional output spaces, while an action-only model can devote all of
its adaptation budget to maximizing control performance.




