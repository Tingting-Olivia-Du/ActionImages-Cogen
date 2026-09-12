% Requires: booktabs. \tabref macro.
\begin{table}[t]
\caption{\small
\textbf{Dataset composition.} Two trees. Training uses variation~0 of the upper tree only;
everything else is held out. \textbf{Off.} marks the eight tasks used for offline perception
and action evaluation, \textbf{CL} the twelve used for closed-loop rollouts. The two
evaluation sets differ because they are limited by different things: offline scoring needs
only on-disk ground truth, while a closed-loop rollout needs the task to be expressible by
the end-effector-pose controller. \texttt{close\_jar} passes the first and fails the second
--- its ground-truth replay scores $0/10$ --- so it is excluded from closed-loop aggregates as
a harness floor rather than charged to the model. Three tasks
(\texttt{slide\_block\_to\_target}, \texttt{stack\_wine}, \texttt{sweep\_to\_dustpan}) have
\texttt{variation\_count()}${=}1$ in RLBench --- they expose no task parameter to vary --- and so
can contribute no held-out episodes at all; \texttt{meat\_off\_grill} and \texttt{turn\_tap}
have exactly two.}
\label{tab:data}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrcc}
\toprule
Task & var.\ 0 & var.\ 1 & var.\ 2 & Off. & CL \\
\midrule
\multicolumn{6}{l}{\emph{Training tree} --- variation 0 trained, variations 1--2 held out} \\
\texttt{close\_jar}                   & 50 & 10 & 10 & \checkmark & floor \\
\texttt{insert\_onto\_square\_peg}    & 50 & 10 & 10 & \checkmark & \checkmark \\
\texttt{light\_bulb\_in}              & 50 & 10 & 10 & \checkmark & \checkmark \\
\texttt{meat\_off\_grill}             & 50 & 10 & -- & \checkmark & \checkmark \\
\texttt{open\_drawer}                 & 50 & 10 & 10 & \checkmark & \checkmark \\
\texttt{push\_buttons}                & 50 & 10 & 10 & \checkmark & \checkmark \\
\texttt{put\_item\_in\_drawer}        & 50 & 10 & 10 & \checkmark & \checkmark \\
\texttt{reach\_and\_drag}             & 50 & 10 & 10 & \checkmark & \checkmark \\
\texttt{place\_shape\_in\_shape\_sorter} & 50 & 10 & 10 & & \checkmark \\
\texttt{put\_groceries\_in\_cupboard} & 50 & 10 & 10 & & \checkmark \\
\texttt{put\_money\_in\_safe}         & 50 & 10 & 10 & & \checkmark \\
\texttt{stack\_blocks}                & 50 & 10 & 10 & & \checkmark \\
\texttt{turn\_tap}                    & 50 & 10 & -- & & \checkmark \\
\texttt{slide\_block\_to\_target}     & 50 & -- & -- & & \\
\texttt{stack\_wine}                  & 50 & -- & -- & & \\
\texttt{sweep\_to\_dustpan}           & 38 & -- & -- & & \\
\cmidrule(lr){1-6}
\textbf{16 tasks} & \textbf{788} & \textbf{130} & \textbf{110} & 8 & 12 \\
\midrule
\multicolumn{6}{l}{\emph{Evaluation-only tree} --- never trained, variation 0 only} \\
\texttt{close\_box}                   & 20 & -- & -- & \checkmark & \checkmark \\
\texttt{close\_drawer}                & 20 & -- & -- & \checkmark & \checkmark \\
\texttt{basketball\_in\_hoop}         &  8 & -- & -- & \checkmark & \checkmark \\
\texttt{close\_microwave}             &  8 & -- & -- & \checkmark & \checkmark \\
\texttt{take\_lid\_off\_saucepan}     &  8 & -- & -- & \checkmark & \checkmark \\
\texttt{toilet\_seat\_down}           &  8 & -- & -- & \checkmark & \checkmark \\
\cmidrule(lr){1-6}
\textbf{6 tasks} & \textbf{72} & & & 6 & 6 \\
\bottomrule
\end{tabular}

\vspace{1mm}
{\scriptsize
\raggedright
We render variations $0$--$2$, not all of them. Several tasks expose many more ---
\texttt{close\_jar} has $20$ (one per jar colour) and \texttt{stack\_blocks} $60$ --- so the
held-out-variation tier samples a small part of each task's parameter space rather than
covering it. Since training uses variation~0 only, rendering deeper into variation~0 buys
more training data than rendering wider does, which is why the budget went there. A wider
sweep would test referent generalization more thoroughly than we do.
The evaluation-only tree is rendered at variation~0 alone, deliberately: a variation index
selects a parameter \emph{within} a task, so for a task the model has never seen,
variation~0 is already entirely unseen and a second variation would add an unrelated axis
rather than more shift. The two held-out tiers are therefore orthogonal --- one moves the
variation with the task set fixed, the other moves the task with the variation fixed --- and
should not be read as one ordered difficulty scale.
Closed-loop scenes come from neither tree: the harness generates a fresh demonstration per
trial from a seed derived from \texttt{(task, variation, trial)}, and those seeds
($5{,}592$--$996{,}990$) are disjoint from the training tree's ($0$--$49$).
}
\end{table}
