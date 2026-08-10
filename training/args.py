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
    #   depth+action        the previous substitution arm: a depth-space WORLD MODEL, not
    #                       perception -- there is no RGB in that sequence at all
    # `name@ratio` comma-separated, same grammar as dataset_name; a bare name means @1.0.
    # Only `rlbench_selfgen` can serve depth/segmentation (it is the only tree with GT).
    template_mix: str = field(
        default="video+action@1.0",
        metadata={"help": "Task-template mix as name@ratio, e.g. "
                          "'video+action@0.6,video+depth@0.2,video+segmentation@0.2'. The "
                          "default 'video+action@1.0' is the upstream recipe."},
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
    steps_per_epoch: int = field(default=500, metadata={"help": "Number of steps per epoch."})
    num_frames: int = field(default=41, metadata={"help": "Number of frames."})
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

        # Call parent __post_init__
        super().__post_init__()


def parse_args():
    """Parse arguments using HfArgumentParser with dataclass"""
    parser = transformers.HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()
    return training_args
