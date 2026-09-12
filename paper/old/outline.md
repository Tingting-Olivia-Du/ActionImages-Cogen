% REQUIRED PACKAGES for the merged main table:
%   \usepackage{multirow}   % row-group labels in the leftmost column
%   \usepackage{graphicx}   % \rotatebox for those labels
%
\section{Experiments}
\label{sec:experiments}

% Main questions:
% 1. Can one shared checkpoint support action, depth, segmentation,
%    and surface-normal generation through prompt switching while
%    matching modality specialists under matched modality exposure?
%
% 2. Does adding perception outputs preserve world-action capability?
%
% 3. Does the unified capability generalize to unseen task variations?


\subsection{Experimental setup}
\label{sec:exp_setup}

\paragraph{Dataset.}
% RLBench, 16 tasks, 4 views, 512x512.
% Variation 0 for training.
% Held-out variations for evaluation.
% Depth, scene-role segmentation, and surface-normal annotations.

\paragraph{Models.}
\begin{itemize}
    \item \textbf{Initialization}: released Action Images checkpoint.

    \item \textbf{Action specialist}: video+action only.

    \item \textbf{Depth specialist}: video+depth only.

    \item \textbf{Segmentation specialist}:
    video+scene-segmentation only.

    \item \textbf{Surface-normal specialist}: video+normal only.

    \item \textbf{Unified model}: video+action, video+depth,
    video+scene-segmentation, and video+normal.
\end{itemize}

\paragraph{Training.}
% All fine-tuned models use the same initialization, backbone,
% frozen VAE, objective, optimizer, and data.
%
% Unified-model template mixture:
% action 0.4,
% depth 0.2,
% segmentation 0.2,
% normal 0.2.
%
% Each specialist samples only its corresponding template.
% Main evaluation uses IIII.

\paragraph{Matched modality exposure.}
% The headline comparison controls modality-specific training exposure.
%
% If the unified model is evaluated after T total updates:
%
% Action specialist:
% evaluated after 0.4T action updates.
%
% Depth specialist:
% evaluated after 0.2T depth updates.
%
% Segmentation specialist:
% evaluated after 0.2T segmentation updates.
%
% Surface-normal specialist:
% evaluated after 0.2T normal updates.
%
% This prevents specialists from receiving more direct supervision
% than the unified model for the compared modality.

\paragraph{Metrics.}
% Action:
% position error, rotation error, gripper accuracy.
%
% Depth:
% AbsRel.
%
% Segmentation:
% macro-mIoU.
%
% Surface normals:
% cosine similarity.
%
% RGB future:
% LPIPS / SSIM.
%
% Closed-loop:
% task success.


\subsection{One checkpoint, multiple robot outputs}
\label{sec:exp_unified}

% Headline question:
% Can one shared checkpoint support all output types through prompt
% switching while approaching modality specialists under matched exposure?

\begin{table}[t]
\caption{
Main results. One grid, three comparability classes, and the columns that make
them comparable are stated rather than assumed. \textbf{RGB} is what the model
is handed: \emph{real} means a ground-truth frame, \emph{gen.} means the model
generated the frame it then interprets. \textbf{Scale} is whether depth comes
out in metres or needs a free parameter fitted against the ground truth.
\textbf{Dom.} is whether the model ever saw this data distribution. All cells
use the same $n{=}40$ held-out episodes (8 tasks $\times$ 5, variation~1);
our-versus-ours differences are paired.
}
\label{tab:main}
\begin{center}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccc}
\hline
& Model & RGB & Scale & Dom. & Depth & Seg. & Normal & Action \\
& & & & & AbsRel $\downarrow$ & IoU $\uparrow$ & $\cos\uparrow$ & pos.\ (m) $\downarrow$ \\
\hline
\multirow{4}{*}{\rotatebox{90}{\scriptsize external}}
& Depth Anything V2-L & real & aligned & zero-shot & \textbf{[XX]} & -- & -- & -- \\
& Depth Anything 3-L  & real & aligned & zero-shot & [XX] & -- & -- & -- \\
& VGGT-1B             & real & aligned & zero-shot & [XX] & -- & -- & -- \\
& SAM ViT-H           & real & ---     & zero-shot & -- & [XX]$^{\ddagger}$ & -- & -- \\
\hline
\multirow{4}{*}{\rotatebox{90}{\scriptsize specialists}}
& Depth specialist    & gen. & metric & in-dom. & [XX] & -- & -- & -- \\
& Seg.\ specialist    & gen. & ---    & in-dom. & -- & [XX] & -- & -- \\
& Normal specialist   & gen. & ---    & in-dom. & -- & -- & [XX] & -- \\
& Action specialist   & gen. & ---    & in-dom. & -- & -- & -- & [XX] \\
\hline
\multirow{2}{*}{\rotatebox{90}{\scriptsize ours}}
& Unified, \texttt{FIFI} & real & metric & in-dom. & [XX] & [XX]$^{\ddagger}$ & [XX] & -- \\
& Unified, \texttt{IIII} & gen. & metric & in-dom. & [XX] & [XX] & [XX] & [XX] \\
\hline
\end{tabular}
\end{center}
\end{table}

% ROW GROUPS ARE NOT INTERCHANGEABLE, and the table says so in its columns rather
% than in a footnote nobody reads.
%
% - External rows are zero-shot and are handed a real frame. They bound how hard
%   the task is; they are not a ranking against us, because we are fine-tuned on
%   this exact distribution and they are not.
% - Every external depth model needs its output aligned to the ground truth
%   before it can be scored at all (median scale for the scale-ambiguous ones,
%   affine-in-disparity for Depth Anything, whose relative models emit inverse
%   depth). Ours is metric out of the codec. The "Scale" column is the result,
%   not bookkeeping: raw, unaligned AbsRel for the external models is 0.36-6.95.
% - FIFI and IIII differ only in whether the RGB is given. FIFI is the row
%   comparable with the external models; IIII is the regime the rest of the
%   paper reports, and is strictly harder because the model must first generate
%   the frame it then interprets.
%
% dagger-dagger (ddagger): the SAM cell and the two FIFI/IIII cells beside it are
% class-agnostic best-overlap IoU -- for each ground-truth region, the best IoU
% over any predicted mask -- because SAM emits unnamed regions and macro-mIoU is
% undefined for it. That metric rewards over-segmentation, and SAM emits ~48
% masks per frame against our ~4 and the scene's ~4 regions, so it is reported
% with the mask counts in the text. The unmarked Seg. cells are macro-mIoU over
% named roles, which is the stricter quantity and the one the rest of the paper
% uses.

\paragraph{Prompt-switched generation.}
% Qualitative visualization using the same unified checkpoint.
%
% Keep fixed:
% - checkpoint
% - episode
% - views
% - seed
%
% Change only:
% - modality prompt/tag
% - corresponding initial modality anchor
%
% Show:
% <video><action>
% <video><depth>
% <video><scene-seg>
% <video><normal>
%
% Main takeaway:
% All quantitative capabilities in Table~\ref{tab:headline}
% coexist in one checkpoint and are selected through the prompt.


\subsection{Preserving world-action capability}
\label{sec:exp_preservation}

% Compare:
%
% Initialization:
% common released Action Images checkpoint.
%
% Action specialist:
% continued training using video+action only.
%
% Unified model:
% continued training using action + three perception templates.
%
% Initialization -> Action specialist:
% effect of continued training / domain adaptation.
%
% Action specialist -> Unified model:
% effect of adding perception outputs.
%
% Main question:
% Does the unified model remain a good world-action model?

\begin{table}[t]
\caption{
Preservation of world-action capability. Offline columns are the $n{=}8$
held-out (variation-1) episodes, paired. Success is the paired closed-loop
campaign restricted to the $n{=}80$ scenes all three models have run
(5 tasks $\times$ 20 seeds, less the 20 the action specialist had not
finished), so the three numbers are
directly comparable; on the full $n{=}100$ grid the initialization scores
$24\%$ and the unified model $35\%$.
}
\label{tab:wam_preservation}
\centering
\small
\begin{tabular}{lccccc}
\toprule
Model &
Pos. err. $\downarrow$ &
Rot. err. $\downarrow$ &
Grip. acc. $\uparrow$ &
RGB LPIPS $\downarrow$ &
Success $\uparrow$ \\
\midrule
Initialization &
0.218 & 77.5$^\circ$ & 0.732 & 0.240 & 30.0\% \\
Action specialist &
0.166 & \textbf{60.2}$^\circ$ & 0.820 & \textbf{0.220} & \textbf{57.5}\% \\
Unified model &
\textbf{0.157} & 67.2$^\circ$ & \textbf{0.890} & 0.226 & 43.8\% \\
\bottomrule
\end{tabular}
\end{table}

% Main takeaway:
% The unified model gains depth, segmentation, and normal generation while
% preserving action prediction, RGB prediction, and closed-loop control.
%
% Against the released initialization the unified model improves on every
% column, but that comparison bundles domain adaptation together with
% perception co-training. The action-specialist row separates them, and the two
% arrows say different things:
%
%   init -> action specialist    30.0% -> 57.5%   (+27.5pp, p < 1e-4)
%   action specialist -> unified 57.5% -> 43.8%   (-13.8pp, p = 0.099)
%
% So most of the closed-loop gain is domain adaptation, and adding three dense
% perception outputs costs about 14 points of it back. That cost is not
% significant at n=80 (13 discordant pairs favour the unified model against 24
% for the action specialist), but it is not evidence of preservation either --
% the interval comfortably includes a real degradation, and the point estimate
% is a loss.
%
% OFFLINE ACTION METRICS DO NOT SHOW THIS. On the same held-out episodes the
% unified model has the better median position error (0.157 vs 0.166 m) and the
% better gripper accuracy (0.890 vs 0.820), and is behind only on rotation.
% Whatever the perception streams cost the policy, it does not appear in the
% decoded single-step trajectory error -- it appears once the model has to act
% on its own predictions over a full episode. We report both rather than the
% one that flatters the method.
%
% The honest claim is therefore narrower than "preservation": the unified model
% keeps most of the control gained by domain adaptation, and clearly beats its
% own initialization, while giving up part of what an action-only model of the
% same budget achieves.


\subsection{How far does the unified capability generalize?}
\label{sec:exp_generalization}

% Three nested tiers, not one. Reporting a single "held-out" number hides that
% perception and control fail at DIFFERENT boundaries, which is the most
% interesting thing we measured.
%
%   (1) seen           variation 0, episodes literally in the training set
%   (2) unseen variation   variation 1 of a TRAINED task
%   (3) unseen task    a task absent from the training tree entirely
%
% Tier 3 required rendering a separate evaluation-only tree for six tasks that
% are absent from the training tree (close_box, close_drawer, close_microwave,
% take_lid_off_saucepan, toilet_seat_down, basketball_in_hoop), kept out of the
% training tree so no future run can absorb them. The open-loop harness scores
% against on-disk ground truth, so this tier was previously unmeasurable. Six
% tasks rather than two because the numbers moved by 5-10% between n=2 and n=6:
% two tasks carry visible task-specific bias.

\begin{table}[t]
\caption{
Generalization across three tiers, and whether unification is what provides it.
The left block tracks the unified model (\texttt{IIII}) as the distribution
shifts; the \emph{unseen task} block additionally reports a matched-exposure
single-modality specialist on the same episodes.
Mean over episodes with 95\% bootstrap CI.
Offline $n$: [XX] (seen), [XX] (unseen variation), 48 over six tasks (unseen
task, paired -- both models ran the identical episodes).
Closed-loop $n$: 60 rollouts (seen), 60 (unseen variation), 80 over six tasks.
}
\label{tab:generalization}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
& & & \multicolumn{4}{c}{\textbf{Unseen task} (paired, $n{=}48$)} \\
\cmidrule(lr){4-7}
Output & Seen & Unseen var. & Unified & Specialist & $\Delta$ [95\% CI] & $p$ \\
\midrule
Depth AbsRel $\downarrow$      & [XX] & [XX] & 0.144 & 0.137 & $+0.007$ $[-0.007,+0.022]$ & 0.50 \\
Segmentation mIoU $\uparrow$   & [XX] & [XX] & 0.546$^{\dagger}$ & 0.549 & $-0.003$ $[-0.024,+0.018]$ & 0.55 \\
Normal cosine $\uparrow$       & [XX] & [XX] & 0.908 & 0.913 & $-0.005$ $[-0.009,-0.000]$ & 0.14 \\
RGB LPIPS $\downarrow$         & [XX] & [XX] & 0.219 & ---   & ---                        & --- \\
\midrule
Closed-loop success $\uparrow$ & 35\% & 1.7\% & 40\% & ---   & ---                        & --- \\
\quad GT-replay ceiling        & 100\% & 92\% & ---  & ---   & ---                        & --- \\
\quad tasks / episodes         & [XX]  & [XX] & \multicolumn{4}{c}{6 / 48} \\
\bottomrule
\end{tabular}
\end{table}

% dagger: mIoU is NOT comparable across the task boundary. close_box has two
% task objects and close_drawer five, against a dozen or more in the trained
% tasks; fewer roles, each covering more pixels, inflates macro-mIoU. The
% honest reading of that cell is "does not collapse", never "improves".

% Main takeaway:
%
% Perception holds across all three tiers, with heavily overlapping CIs, and
% surface normals barely move even on tasks never trained on. [XX: refresh the
% tier-1/tier-2 magnitudes once the n=40 scale-up lands.]
%
% Control does not, and -- the part worth stating carefully -- it does not fail
% MONOTONICALLY. Success is 35% on trained tasks, 1.7% on unseen variations of
% those same tasks, and 40% on tasks never trained on at all. Tier 2 is the
% hard one, not tier 3.
%
% The likely reason is that the tiers vary different things. A new variation
% changes what the instruction REFERS TO -- which button, which drawer -- so
% the policy has to resolve a referent it never resolved during training. A new
% task with an unambiguous goal does not: close_box has one box. So "unseen
% variation" is a language-grounding shift, while "unseen task" is closer to a
% pure appearance shift, and only the first breaks the policy.
%
% Do NOT read tier 3's 40% as arm6 generalizing well. The released checkpoint
% scores 43.8% on those same six tasks -- BETTER than arm6 -- so that column is
% mostly the pretrained prior surviving, minus what our fine-tuning forgot. It
% belongs in the same discussion as the never-trained block of
% tab:closed_loop, not as evidence of transfer.
%
% Ceilings are reported because none of these numbers are readable without
% them: replaying each demo's own actions under the same actuation mode
% succeeds ~92-100% everywhere, including variation 1, so every gap above is a
% model limit and not an unsolvable scene.
%
% What this does and does not claim:
% - Prompt-selected perception outputs survive a distribution shift that the
%   action stream does not. The shared output space is not what limits control;
%   the action prior's own generalization is.
% - It does NOT claim perception is task-agnostic. Six unseen tasks is a probe,
%   not a survey.
% - It does NOT claim the unified model is what makes perception generalize:
%   see the specialist block on the right of the same table.
% - Tier 3's closed-loop 40% is NOT transfer either. The released checkpoint
%   scores 43.8% on the same six tasks, slightly ahead of ours, so that column
%   is the pretrained prior surviving rather than anything fine-tuning added.
%   Two of the six (close_microwave 9/10, take_lid_off_saucepan 8/10) are near
%   ceiling while two others (basketball_in_hoop 0/10, toilet_seat_down 1/10)
%   are near zero -- the aggregate hides a bimodal split by task difficulty.

\paragraph{Is cross-task perception a property of unification? No.}
% Control experiment, and it comes out NEGATIVE -- which is the honest and, we
% think, the more useful answer. It now lives INSIDE tab:generalization as the
% right-hand block, because it is a drill-down into that table's last column,
% not a separate finding: "here is tier 3, and here is what a specialist gets
% on the very same episodes."
%
% Each single-modality specialist, read at the same matched-exposure
% checkpoint, was evaluated on the same never-trained tasks as the unified
% model, paired over the same 48 episodes:
%
%   modality        unified   specialist   paired diff   95% CI            p
%   depth AbsRel     0.144      0.137       +0.007    [-0.007, +0.022]   0.50
%   seg mIoU         0.546      0.549       -0.003    [-0.024, +0.018]   0.55
%   normal cosine    0.908      0.913       -0.005    [-0.009, -0.000]   0.14
%
% Every absolute gap is under 0.007; depth and seg CIs contain zero and the
% normal CI barely excludes it at a magnitude of 0.005 (Wilcoxon p=0.14). A
% specialist that only ever saw ONE modality holds up on unseen tasks exactly
% as well as the unified model does.
%
% IMPORTANT -- why the whole unseen-task block must use n=48 and not n=66:
% arm6 ran 66 unseen-task episodes but the specialists only 48. Quoting the
% n=66 unified number (depth 0.150) next to the n=48 specialist number (0.137)
% makes the table lie arithmetically: 0.150-0.137 is not the paired difference
% (+0.007). The merge is only valid on the common subset, so the entire
% unseen-task column is the n=48 paired subset. The extra 18 episodes move
% depth 0.144 -> 0.150 and are reported in the appendix instead.
%
% So cross-task perception robustness is a property of the frozen VAE and the
% released backbone, NOT something unification buys. We state this explicitly
% because the opposite claim is the easy one to make from the left block of
% Table~\ref{tab:generalization} alone, and it would be wrong.
%
% What this does to the paper's overall claim: it sharpens it rather than
% weakening it. The result is not "unification improves perception" -- it is
% "unification costs nothing", which is exactly what Section~\ref{sec:exp_sharing}
% measures at matched exposure and what makes one checkpoint replacing four a
% real saving rather than a trade.

% Appendix:
% - per-task / per-variation results
% - specialist full-training ceilings
% - closed-loop per-task breakdown + exact McNemar
% - GT-replay ceilings per task and variation
% - confidence intervals / statistical tests
% - action-image diagnostics
% - checkpoint curves
% - failure cases


% Optional:
%
% \subsection{Joint action--perception generation}
% \label{sec:exp_joint}
%
% Train/evaluate video+depth+action.
%
% Purpose:
% demonstrate literal simultaneous co-generation in one packed
% sequence, rather than only prompt-switched outputs.