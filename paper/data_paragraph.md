%% ---------------- BODY ----------------

\paragraph{Data.}
We use a self-generated RLBench tree \citep{james2020rlbench} rendered at $512^2$ with
Colosseum-style domain randomization \citep{pumacay2024colosseum}, carrying four camera
views, per-frame metric depth, and instance masks mapped to the scene-role vocabulary of
\secref{sec:codecs}. Training uses variation~0 of 16 tasks (788 episodes); variations~1
and~2 of those tasks (240 episodes) are held out, and a separate tree of six tasks that
appear nowhere in training (72 episodes) is kept outside it so no future run can absorb it.
Windows are 41 frames at stride 3 --- $6.0$\,s at $6.7$\,Hz --- with two of four views per
example. One caveat governs the held-out split: an RLBench variation index can change the
task rather than re-render it, so variation~1 of \texttt{push\_buttons} asks for two button
presses where variation~0 asks for one. We therefore do not read that split as a pure
appearance shift. \appref{app:data} gives the per-task counts.


%% ---------------- APPENDIX ----------------

\section{Dataset composition}
\label{app:data}

\paragraph{Training tree.}
16 tasks, 1{,}028 episodes. Variation~0 holds 788 of them and is the only split used for
training: 50 episodes per task, except \texttt{sweep\_to\_dustpan}, which yielded 38.
Variation~1 holds 130 (10 each for the 13 tasks that have a second variation) and
variation~2 holds 110 (10 each for 11 tasks); both are held out entirely. Three tasks
(\texttt{slide\_block\_to\_target}, \texttt{stack\_wine}, \texttt{sweep\_to\_dustpan})
exist only at variation~0 and so contribute no held-out episodes.

\paragraph{What a variation changes.}
The index selects a task parameter, not a rendering seed, and how much it changes varies by
task. \texttt{push\_buttons} goes from one button press to two, which lengthens the horizon;
\texttt{meat\_off\_grill} substitutes a steak for the chicken, which changes object
identity; \texttt{open\_drawer} and \texttt{turn\_tap} name a different drawer or tap, which
changes only the referent. The held-out split therefore mixes three kinds of shift, and
\tabref{tab:pertask} shows they do not cost the same: closed-loop success on
\texttt{push\_buttons} falls from $6/10$ to $0/10$ and on \texttt{meat\_off\_grill} from
$5/10$ to $1/10$, while \texttt{turn\_tap} holds at $4/10$ in both.

\paragraph{The two held-out tiers vary different things.}
The held-out-variation tier fixes the task set and moves the variation index; the
unseen-task tier fixes the variation index at~0 and moves the task. The evaluation-only tree
is rendered at variation~0 only, and deliberately so: a variation index selects a parameter
\emph{within} a task, so for a task the model has never seen, variation~0 is already
entirely unseen and a second variation would add an unrelated axis rather than more shift.
Reading the two tiers as a single ordered scale would therefore be wrong --- they are
orthogonal, and \tabref{tab:main} shows the model is not uniformly worse on the further
one.

\paragraph{Evaluation-only tree.}
Six tasks absent from training --- \texttt{basketball\_in\_hoop}, \texttt{close\_box},
\texttt{close\_drawer}, \texttt{close\_microwave}, \texttt{take\_lid\_off\_saucepan},
\texttt{toilet\_seat\_down} --- with 20 episodes each for \texttt{close\_box} and
\texttt{close\_drawer} and 8 for the rest, 72 in total. It is stored outside the training
tree rather than filtered out of it, so a future training run cannot pick it up by
accident.

\paragraph{Closed-loop scenes are not drawn from either tree.}
The rollout harness generates a fresh demonstration per trial from a seed derived from
\texttt{(task, variation, trial)}. Those seeds ($5{,}592$--$996{,}990$) are disjoint from
the training tree's ($0$--$49$), so a closed-loop evaluation at variation~0 is a new scene
from the training distribution rather than a replay of a training episode.
