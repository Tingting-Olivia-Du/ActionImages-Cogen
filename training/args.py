import transformers
from dataclasses import dataclass, field
from typing import Optional


def _translate_visual_modality_mix(mix: str) -> str:
    """'video@0.5,depth@0.5' -> 'video+action@0.5,depth+action@0.5' (deprecated path).

    The old flag chose which modality REPLACED video in the 4-segment layout; the action
    segment was always present. That is exactly the '<modality>+action' family of templates.
    """
    out = []
    for item in [s.strip() for s in str(mix).split(",") if s.strip()]:
        name, sep, ratio = item.partition("@")
        name = name.strip()
        if name not in ("video", "depth", "segmentation"):
            raise ValueError(
                f"unknown modality {name!r} in --visual_modality_mix; expected "
                f"video|depth|segmentation"
            )
        out.append(f"{name}+action{sep}{ratio}" if sep else f"{name}+action")
    if not out:
        raise ValueError(f"--visual_modality_mix is empty: {mix!r}")
    return ",".join(out)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Extended TrainingArguments for ActionImages training with additional custom parameters.
    """

    dataset_path: str = field(default="./data", metadata={"help": "The path of the Dataset."})
    dataset_name: str = field(
        default="rlbench",
        metadata={
            "help": "Dataset name(s): rlbench, bridge, or droid. "
            "Use name@ratio for mixed sampling (e.g. rlbench@0.5,bridge@0.3,droid@0.2); "
            "a single name without @ defaults to ratio 1.0."
        },
    )
    output_dir: str = field(
        default="./checkpoints",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of subprocesses to use for data loading."})

    # Model paths
    model_id: str = field(default="Wan-AI/Wan2.2-TI2V-5B", metadata={"help": "Path of model id."})

    # VAE tiling options
    tiled: bool = field(
        default=False, metadata={"help": "Whether enable tile encode in VAE. This option can reduce VRAM required."}
    )
    tile_size_height: int = field(default=34, metadata={"help": "Tile size (height) in VAE."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size (width) in VAE."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride (height) in VAE."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride (width) in VAE."})

    # FORK: which TASK TEMPLATES the run trains on, and in what proportion. A template is its
    # modality set Pi written in canonical order and joined with "+"; the prompt announces Pi
    # and the mask decides which of its segments are given (see training/templates.py and
    # /workspace/ttdu/ttd/plan/core/16-multitask_template_design.md).
    #   video+action        official recipe, 4 segments
    #   video+depth         RGB given -> depth predicted: perception, 4 segments (same cost)
    #   video+segmentation  RGB given -> referring seg predicted, 4 segments
    #   video+depth+action  co-generation, 6 segments (~1.5x sequence, ~2.25x attention)
    #   depth+action        the substitution family: a WORLD MODEL in depth space, not
    #   segmentation+action perception -- there is no RGB in the sequence at all. Unlike the
    #   normal+action       video+X perception templates these DO carry action supervision,
    #                       which is what arm7's menu is built from: one action stream
    #                       conditioned on four different renderings of the same scene.
    # `name@ratio` comma-separated, same grammar as dataset_name; a bare name means @1.0.
    # Only `rlbench_selfgen` can serve depth/segmentation (it is the only tree with GT).
    template_mix: str = field(
        default="video+action@1.0",
        metadata={"help": "Task-template mix as name@ratio, e.g. "
                          "'video+action@0.6,video+depth@0.2,video+segmentation@0.2'. The "
                          "default 'video+action@1.0' is the upstream recipe."},
    )
    # FORK: per-dataset template menus. The feasible menu is a property of the DATA (only
    # rlbench_selfgen* has depth/seg GT; bridge has no usable action), so a single global mix
    # would have to be silently filtered per tree, making the realised task proportions differ
    # from the requested ones with nothing in the log saying so.
    template_mix_per_dataset: str = field(
        default="",
        metadata={"help": "Per-dataset template menus, ';'-separated 'dataset=mix' entries, "
                          "e.g. 'rlbench_selfgen_512_aug=video+action@0.67,video+depth@0.33;"
                          "droid=video+action@1.0;bridge=video@1.0'. Datasets not listed fall "
                          "back to --template_mix. A menu the dataset cannot serve is a "
                          "startup error, not a silent downgrade."},
    )
    prompt_tag_style: str = field(
        default="explicit",
        metadata={"help": "'explicit' (default) writes Pi into the prompt for EVERY template, "
                          "including <video><action> on the baseline. 'none' reproduces "
                          "upstream text verbatim and exists so the numeric regression against "
                          "upstream still has a hook -- it is not a training setting."},
    )
    action_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability that a template containing <action> drops it for this "
                          "sample, becoming the visual-only template. Reproduces upstream's "
                          "`random.random() < 0.9` action gate (train.py:286), moved into the "
                          "dataset so the prompt and the segment layout are decided together."},
    )
    segmentation_mode: str = field(
        default="referring",
        metadata={"help": "Which segmentation protocol the `segmentation` modality encodes. "
                          "'referring' (default) = the current target-only mask: the instruction's "
                          "referred objects get a colour from seg_targets.json, everything else is "
                          "black. 'scene_roles' = dense cross-task functional roles from "
                          "scene_segments.json (SEGMENTATION_SCENE_ROLES_PLAN.md). The default "
                          "keeps every existing command, checkpoint and eval bit-identical; the "
                          "two use DIFFERENT prompt tags so the model can tell them apart."},
    )
    perception_mask_mix: Optional[str] = field(
        default=None,
        metadata={"help": "The M axis (SEGMENTATION_SCENE_ROLES_PLAN.md §10.3): how a PERCEPTION "
                          "template (one with no <action>, e.g. video+depth / video+segmentation) "
                          "splits its conditioning, as 'iiii,fiii,fifi,single_frame' summing to 1, "
                          "or a preset M0/M1/M2. M0 (0.1,0,0.9,0) is the default and reproduces "
                          "today's behaviour exactly. M2 (0.9,0.05,0.05,0) matches the official "
                          "video+action split, so template identity stops predicting conditioning "
                          "level and the auxiliary stream trains in the same generative regime the "
                          "closed-loop rollout uses. Does NOT affect templates containing <action>."},
    )
    action_mask_mix: Optional[str] = field(
        default=None,
        metadata={"help": "The A axis: the exact COMPLEMENT of --perception_mask_mix. How a "
                          "template that CONTAINS <action> (video+action, depth+action, "
                          "segmentation+action, normal+action) splits its conditioning, as "
                          "'iiii,fiii,fifi,policy' summing to 1, or a preset A0/A1/A2. A0 is the "
                          "default and reproduces the upstream literals exactly (81% IIII / 4.5% "
                          "FIII / 4.5% FIFI / 10% single-frame policy mode), so leaving this unset "
                          "changes nothing. A1 (0.75,0,0.05,0.20) is arm7's action-heavy mix: FIII "
                          "is the cross-view mode and the least relevant of the four to action "
                          "quality, so it is spent on policy mode, whose loss lands entirely on "
                          "the action segments under the same single-frame conditioning the "
                          "closed-loop rollout provides. A2 (0.9,0.05,0.05,0) is 90/5/5 with no "
                          "policy mode. Does NOT affect action-free templates."},
    )
    fusion_mask_mix: Optional[str] = field(
        default=None,
        metadata={"help": "The F axis: modality dropout on a multi-modality (fusion) canvas, as "
                          "'full_anchor,rgb_only,rgb_given,one_out,policy' summing to 1, or a "
                          "preset F0/F1/F2. It REPLACES the M and A axes for the sample rather "
                          "than composing with them. Unset (the default) is meaningful: the "
                          "template keeps the M/A axes, which is the `fusion0-anchor` control. "
                          "Why it exists: assemble() hands every non-fully-given segment a free "
                          "first latent frame, so a 10-segment video+depth+segmentation+normal+"
                          "action canvas gets TEN free anchors. That makes cross-modal completion "
                          "a copy task (the tag-swap ablation measured the anchor as ~9x more "
                          "load-bearing than the prompt tag) and trains a conditioning regime "
                          "deployment cannot supply, since rollout only ever has RGB. F1 "
                          "(0.30,0.30,0.15,0.15,0.10) is the Stage-1 default. Requires a template "
                          "with at least two modalities."},
    )
    # DEPRECATED alias kept so existing commands keep working. 'video|depth|segmentation' maps
    # to the substitution templates 'video+action|depth+action|segmentation+action'.
    visual_modality_mix: Optional[str] = field(
        default=None,
        metadata={"help": "DEPRECATED, use --template_mix. Old visual-slot mix "
                          "(video|depth|segmentation as name@ratio); translated to the "
                          "corresponding '<modality>+action' substitution templates."},
    )
    variations: str = field(
        default="all",
        metadata={"help": "Which RLBench variations this split may see. 'all' (default), "
                          "'0' = train split (variation0 only), '!0' = test split (everything "
                          "else), or a list/range like '0,1' / '0-2'. The train/test split is "
                          "BY VARIATION -- training without '0' here silently trains on the "
                          "held-out set."},
    )
    strict_getitem: bool = field(
        default=False,
        metadata={"help": "Re-raise dataset exceptions instead of retrying. Use for smoke runs "
                          "so a broken data path gives a traceback instead of silently serving "
                          "a different sample."},
    )

    # Training configuration
    # NOTE: read by nothing in the training loop -- `max_steps` is the only stop condition.
    # Kept because existing launchers pass it; do not add logic that depends on it without
    # also making the launchers agree.
    steps_per_epoch: int = field(
        default=500,
        metadata={"help": "UNUSED. Retained for launcher compatibility; max_steps stops training."},
    )
    num_frames: int = field(default=41, metadata={"help": "Number of frames."})
    frame_interval: int = field(
        default=1,
        # FORK: was hardcoded to 1 at the train.py call site, so there was no way to change the
        # TEMPORAL SPAN of a training window without editing source. It matters because the
        # official RLBench release stores video already 4x-downsampled and realigns actions with
        # `actions[::4]` (dataset/rlbench.py:105), while rlbench_selfgen writes every stream at
        # native 20 Hz 1:1 and must NOT downsample (dataset/rlbench_selfgen.py:267-275). Same
        # `num_frames`, 4x different horizon: official 41 frames span (41-1)*4/20 = 8.0 s,
        # selfgen 41 frames span (41-1)/20 = 2.0 s.
        #
        # Raising it costs window diversity, because a window now needs `1 + (num_frames-1) *
        # frame_interval` native steps to fit and `num_frames * frame_interval` to have more
        # than one legal start. Measured over 2818 selfgen episodes (median length 145 steps):
        #   interval  span   eff.Hz   mean pad   episodes with >1 window
        #      1      2.0 s   20.0       0.0%           100.0%
        #      2      4.0 s   10.0       0.5%            96.5%
        #      3      6.0 s    6.7       5.1%            61.3%
        #      4      8.0 s    5.0      15.3%            34.4%
        # (official at interval 1 sits at 5.7% mean pad / 43.9% with >1 window, so 3 is the
        # closest match on padding while keeping most of the temporal augmentation.)
        metadata={"help": "Stride between sampled frames within a training window. 1 (default) "
                          "keeps upstream behaviour. Only raise it for datasets stored at native "
                          "control rate (rlbench_selfgen); the official rlbench loader already "
                          "downsamples actions by 4 internally, so raising this there compounds "
                          "to a 4x-larger stride. DROID sets its own interval independently."},
    )
    height: int = field(default=512, metadata={"help": "Image height."})
    width: int = field(default=512, metadata={"help": "Image width."})
    full_param: bool = field(default=False, metadata={"help": "Whether to train all parameters."})
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the backbone model."})

    # Legacy argument mappings (for backward compatibility)
    accumulate_grad_batches: Optional[int] = field(
        default=None,
        metadata={
            "help": "Legacy: The number of batches in gradient accumulation. Maps to gradient_accumulation_steps."
        },
    )
    max_epochs: Optional[int] = field(
        default=None, metadata={"help": "Legacy: Number of epochs. Maps to num_train_epochs."}
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "Legacy: Path to save the model. Maps to output_dir."}
    )

    # Gradient checkpointing
    use_gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing."})
    use_gradient_checkpointing_offload: bool = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing offload."}
    )

    # Checkpoint configuration
    metadata_file_name: str = field(default="metadata.csv", metadata={"help": "Name of the metadata file."})
    resume_ckpt_path: Optional[str] = field(default=None, metadata={"help": "Path to resume checkpoint."})
    init_ckpt_path: Optional[str] = field(default=None, metadata={"help": "Path to initialize checkpoint."})
    allow_step_restart: bool = field(
        default=False,
        metadata={"help": "Permit a weights-only resume into a non-empty output_dir. The step counter restarts at 0, so existing checkpoint-N directories WILL be overwritten; off by default."},
    )
    checkpoint_every_n_steps: int = field(default=1000, metadata={"help": "Save checkpoint every N training steps."})
    checkpoint_save_top_k: int = field(
        default=-1, metadata={"help": "Number of best checkpoints to keep. -1 saves all checkpoints."}
    )
    checkpoint_monitor: str = field(
        default="train_loss", metadata={"help": "Metric to monitor for best checkpoint selection."}
    )
    keep_optimizer_last_only: bool = field(
        default=True,
        metadata={"help": "Keep DeepSpeed resume state (global_step*/) only in the NEWEST "
                          "checkpoint, and never keep HF's redundant fp32 pytorch_model.bin. "
                          "Takes a checkpoint directory from ~144GB to ~12GB once it is no "
                          "longer the resume tip, while every stepN.ckpt stays for eval. "
                          "Set False to keep every checkpoint independently resumable."},
    )

    def __post_init__(self):
        # Handle legacy argument mappings
        if self.accumulate_grad_batches is not None:
            self.gradient_accumulation_steps = self.accumulate_grad_batches
        if self.max_epochs is not None:
            self.num_train_epochs = self.max_epochs
        if self.output_path is not None:
            self.output_dir = self.output_path

        if self.visual_modality_mix is not None:
            # Refuse to guess which one the user meant: silently preferring one would make a
            # run train on a different task menu than its command line says.
            if self.template_mix != "video+action@1.0":
                raise ValueError(
                    "--visual_modality_mix is deprecated and cannot be combined with "
                    "--template_mix. Pass only --template_mix."
                )
            self.template_mix = _translate_visual_modality_mix(self.visual_modality_mix)

        if self.prompt_tag_style not in ("explicit", "none"):
            raise ValueError(
                f"--prompt_tag_style must be 'explicit' or 'none', got {self.prompt_tag_style!r}"
            )
        if not 0.0 <= self.action_dropout_prob <= 1.0:
            raise ValueError(f"--action_dropout_prob must be in [0,1], got {self.action_dropout_prob}")

        # Validate now rather than at step 40k: a malformed mix is a typo in a launch script,
        # and the run would otherwise load a 12.8GB checkpoint before finding out.
        if self.fusion_mask_mix:
            from training.templates import parse_fusion_mask_mix

            parse_fusion_mask_mix(self.fusion_mask_mix)

        # Call parent __post_init__
        super().__post_init__()


def parse_args():
    """Parse arguments using HfArgumentParser with dataclass"""
    parser = transformers.HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()
    return training_args
