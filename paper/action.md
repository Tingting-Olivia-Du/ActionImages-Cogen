% Requires: booktabs. \tabref / \appref macros.
\begin{table}[t]
\caption{\small
\textbf{World-action capability.} Whether adding three dense perception outputs costs the
action capability inherited from pretraining. Both tiers are held out: \emph{held-out
variation} is variation~1 of the trained tasks, \emph{unseen task} is a tree kept out of
training entirely --- the same two tiers as \tabref{tab:main}, so the tables are directly
comparable. Offline metrics are $n{=}40$ and $n{=}66$ episodes; closed-loop is $10$
rollouts on each of twelve tasks the harness can express, and on each of six never-trained
tasks. The released checkpoint runs under its own protocol (cfg $10.0$, frame interval $4$,
no prompt tags), which is what it was trained with. \textsc{inp}: in progress.}
\label{tab:wam_preservation}
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llccc|c}
\toprule
& & \multicolumn{3}{c|}{\textbf{Offline action}} & \textbf{Closed-loop} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-6}
Model & Tier
& Pos.\ (m) $\downarrow$ & Rot.\ $\downarrow$ & Grip.\ acc.\ $\uparrow$
& Success $\uparrow$ \\
\midrule
Released initialization & held-out var.
& 0.207 & \textbf{69.6}$^\circ$ & 0.779 & \textsc{inp} \\
Action specialist & held-out var.
& \textbf{0.162} & 67.3$^\circ$ & 0.824 & \textsc{inp} \\
Unified model & held-out var.
& 0.172 & 76.0$^\circ$ & \textbf{0.848} & \textbf{4.2\%} \\
\midrule
Released initialization & unseen task
& 0.137 & 64.0$^\circ$ & 0.804 & \textbf{43.8\%} \\
Action specialist & unseen task
& \textbf{0.128} & \textbf{56.3}$^\circ$ & \textbf{0.840} & \textsc{inp} \\
Unified model & unseen task
& 0.136 & 62.2$^\circ$ & 0.835 & 40.0\% \\
\bottomrule
\end{tabular}

\vspace{1mm}
{\scriptsize
\raggedright
The three models' video streams are within $0.008$ LPIPS of each other on the same
generations ($0.223$--$0.231$), so the action differences above are not a side effect of one
model generating worse video; RGB quality is reported once, in \tabref{tab:main}.
Closed-loop rollouts are not reproducible at the trajectory level: the sampling-based motion
planner is not seeded, so re-running an identical configuration flips $7\%$ of rollouts and
a $120$-rollout column carries about $\pm 2.5$ points. Scene seeds are shared across models,
so the comparison is paired at the scene level but not at the trajectory level. The offline
columns are deterministic. Per-task closed-loop results are in \tabref{tab:pertask}.
}
\end{table}
