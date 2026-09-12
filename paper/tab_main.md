% Requires: booktabs. \tmpl macro for template names.
\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}

\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l c c c @{}}
\toprule[1pt]
\textbf{Capabilities} & \textbf{Benchmarks and Metrics}
& \textbf{U-CoGen} & \textbf{In-dom.\ spec.} & \textbf{Best Counterpart} \\
\midrule

3D Understanding
& Metric depth: held-out variation (AbsRel $\downarrow$)
& 0.385 & \textbf{0.164} & 0.125 (\emph{Depth Anything V2-L}) \\
& Metric depth: unseen task (AbsRel $\downarrow$)
& \textbf{0.067} & 0.137 & 0.119 (\emph{Depth Anything V2-L}) \\
& Surface normal: held-out variation ($\cos\uparrow$)
& \textbf{0.969} & 0.892 & 0.946 (\emph{Lotus-G v1-1}) \\
& Surface normal: unseen task ($\cos\uparrow$)
& \textbf{0.973} & 0.913 & 0.944 (\emph{Lotus-G v1-1}) \\

\midrule
2D Understanding
& Scene-role seg.: held-out variation (mIoU $\uparrow$)
& \textbf{0.812} & 0.491 & 0.284 (\emph{CLIPSeg}) \\
& Scene-role seg.: unseen task (mIoU $\uparrow$)
& \textbf{0.822} & 0.549 & 0.341 (\emph{CLIPSeg}) \\
& Class-agnostic seg.: held-out variation (bo-IoU $\uparrow$)
& \textbf{0.838} & \textsc{inp} & 0.632 (\emph{SAM ViT-H}) \\
& Class-agnostic seg.: unseen task (bo-IoU $\uparrow$)
& \textsc{inp} & \textsc{inp} & 0.670 (\emph{SAM ViT-H}) \\

\midrule
Action
& End-effector position: held-out variation (m $\downarrow$)
& 0.172 & \textbf{0.162} & 0.207 (\emph{ActionImages}, released) \\
& End-effector position: unseen task (m $\downarrow$)
& 0.136 & \textbf{0.128} & 0.137 (\emph{ActionImages}, released) \\

\midrule
Visual Generation
& Future RGB: held-out variation (LPIPS $\downarrow$)
& \textbf{0.228} & 0.230 & 0.231 (\emph{ActionImages}, released) \\

\bottomrule[1pt]
\end{tabular*}

\caption{\small
\textbf{Main results.} One checkpoint, prompted into each output space, against a
matched-exposure specialist for that output and against the strongest counterpart we could
run for it. Perception rows use the \texttt{FIFI} conditioning plan, which is given the real
frame like every counterpart; action and RGB use \texttt{IIII}, since predicting the future
is the task. $n{=}40$ held-out variation-1 episodes over eight tasks, or $n{=}66$ over six
never-trained tasks, one seed and identical camera views for every model.
\textsc{inp}: in progress. Protocol, model pool and per-model results are in
\secref{sec:protocol} and \appref{app:external}.
}
\label{tab:main}
\end{table}
