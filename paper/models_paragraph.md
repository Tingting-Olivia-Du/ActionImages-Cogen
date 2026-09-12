%% ---------------- BODY ----------------
%% Replaces both "Models and Training" and "Matched modality exposure": the second was
%% mostly a restatement of the first's step counts, and its closing claim about the
%% specialists' later checkpoints does not hold -- each specialist has exactly one, at the
%% matched-exposure step.

\paragraph{Models and training.}
All fine-tuned models start from the same released Action Images checkpoint
\citep{zhen2026action}, which we also report unadapted as the \emph{initialization}. We
train one unified model and four single-modality specialists --- action, depth, segmentation
and surface normal. Each specialist samples only its own template; the unified model samples
\tmpl{video+action} with probability $0.4$ and each of \tmpl{video+depth},
\tmpl{video+segmentation} and \tmpl{video+normal} with probability $0.2$. Backbone, frozen
VAE, objective, optimizer, data tree and conditioning distribution are identical across every
fine-tuned model, so the template distribution is the only variable. We fine-tune all
parameters with ZeRO-2 and CPU-offloaded AdamW in bf16, batch size 1 per device on two GPUs,
learning rate $5\times10^{-7}$ with $1{,}000$ warmup steps and a constant schedule
thereafter, gradient clipping at $1.0$ and no accumulation.

Specialists are trained to matched \emph{exposure} rather than matched wall-clock. The
unified model trains for $T = 10{,}000$ steps, so its action stream receives $0.4T = 4{,}000$
gradient updates and each perception stream $0.2T = 2{,}000$; we train the action specialist
for $4{,}000$ steps and each perception specialist for $2{,}000$, and compare there. Equal
step counts would instead hand a perception specialist five times the supervision on its own
output and the action specialist two and a half times, and the resulting gap would measure that budget rather than interference. The
question we are asking is narrower and answerable: at equal supervision, does an output lose
anything by sharing a checkpoint with three others?


%% ---------------- APPENDIX ----------------

\paragraph{Scope of the matched-exposure claim.}
\label{app:matched}
Across the seen split, the held-out variation and the unseen tasks, all twelve paired
unified-minus-specialist differences have $95\%$ bootstrap intervals containing zero
(Wilcoxon $p = 0.14$--$0.79$). We state this as the absence of a detectable cost rather than
as equality: at $n{=}40$ the intervals exclude degradations larger than roughly $0.015$
AbsRel and $0.021$ mIoU, but not smaller ones. The comparison is also silent about the
ceiling of a dedicated model --- a specialist trained to convergence would answer how good
one output can get, which is a different question and one we do not run.

The optimizer is forced rather than tuned: full-parameter fine-tuning of a 5B model at this
batch size does not fit with the Adam states on device, so they are offloaded to host
memory. Learning rate and warmup are inherited from the released checkpoint's own recipe, so
that the comparison against it is not confounded by a schedule change.
