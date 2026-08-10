import copy
import os
import torch
import torch.nn as nn
import random
import wandb
import re
from transformers import Trainer
from training.wan_video_action_images import WanVideoActionImagesPipeline
from training.models import ModelConfig
from training.args import parse_args, TrainingArguments
from training.dataset import CombDataset
import logging
from training.utils import (
    get_rank,
    project_actions_7d_to_5d_torch_batch,
    project_action_5d_to_rgb_torch,
    get_plucker_embeddings_torch,
)
from training.templates import (
    ACTION,
    VISUAL_MODALITIES,
    assemble,
    parse_template,
    plan_segments,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assert_own_training_package():
    """Fail loudly if `training` resolved to a DIFFERENT ActionImages checkout.

    The `ttd_train` conda env has an editable install of a distribution also named
    `actionimages` whose finder maps `training` -> /workspace/ttdu/ActionImages/training.
    That finder is *appended* to sys.meta_path, so the stdlib PathFinder (which searches
    sys.path / PYTHONPATH) wins whenever this repo's root is on sys.path -- but if it is
    not, `import training` silently loads the OTHER repo and every change here is a no-op
    against a run that looks completely healthy. Turn that into an immediate crash.

    Fix when it fires: run from this directory with
        export PYTHONPATH=<this repo root>
    `cd` alone is not enough under torchrun (runpy does not prepend the script's dir to
    sys.path); `cd` is separately required because ModelConfig uses the relative
    local_model_path="checkpoints".
    """
    import training

    repo_root = os.path.dirname(os.path.abspath(__file__))
    want = os.path.realpath(os.path.join(repo_root, "training"))
    got = os.path.realpath(os.path.dirname(training.__file__))
    if got != want:
        raise RuntimeError(
            f"'training' resolved to {got}, not this repo's {want}. An editable install of "
            f"'actionimages' shadows it. Run:  cd {repo_root} && export PYTHONPATH={repo_root}"
        )
    logger.info(f"training package: {got}")
    return got


class MinimalConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def to_dict(self):
        return {"hidden_size": self.hidden_size}


def find_latest_checkpoint(output_dir, checkpoint_prefix=None):
    """
    Find the latest checkpoint in the output directory.
    Args:
        output_dir: Directory to search for checkpoints
        checkpoint_prefix: Optional prefix for checkpoint filename (e.g., "policy", "backbone").
                         If None, looks for "step{step}.ckpt". If provided, looks for "{prefix}_step{step}.ckpt".
    Returns the path to the latest checkpoint file, or None if no checkpoints exist.
    """
    if not os.path.exists(output_dir):
        return None

    checkpoint_dirs = []
    # Look for checkpoint directories in the format checkpoint-{step}
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            # Extract step number from directory name
            step_match = re.search(r"checkpoint-(\d+)", item)
            if step_match:
                step = int(step_match.group(1))
                # Look for the checkpoint file inside this directory
                if checkpoint_prefix is None:
                    ckpt_file = os.path.join(item_path, f"step{step}.ckpt")
                else:
                    ckpt_file = os.path.join(item_path, f"{checkpoint_prefix}_step{step}.ckpt")
                if os.path.exists(ckpt_file):
                    checkpoint_dirs.append((step, ckpt_file))

    if not checkpoint_dirs:
        return None

    # Sort by step number and return the latest one
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_step, latest_ckpt_path = checkpoint_dirs[-1]

    prefix_str = f"{checkpoint_prefix} " if checkpoint_prefix else ""
    logger.info(f"Found latest {prefix_str}checkpoint at step {latest_step}: {latest_ckpt_path}")
    return latest_ckpt_path


class ActionImagesModel(nn.Module):
    def __init__(
        self,
        model_id,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        init_ckpt_path=None,
        resolution=(512, 512),
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        full_param=False,
    ):
        super().__init__()

        # Load all models
        logger.info(f"Loading models: {model_id}")
        model_configs = [
            ModelConfig(
                local_model_path="checkpoints",
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                skip_download=True,
            ),
            ModelConfig(
                local_model_path="checkpoints",
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16*.pth",
                offload_device="cpu",
                skip_download=True,
            ),
            ModelConfig(
                local_model_path="checkpoints",
                model_id=model_id,
                origin_file_pattern="Wan*_VAE.pth",
                offload_device="cpu",
                skip_download=True,
            ),
        ]
        tokenizer_config = ModelConfig(
            local_model_path="checkpoints",
            model_id=model_id,
            origin_file_pattern="google/*",
            skip_download=True,
        )
        self.pipe = WanVideoActionImagesPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add camera encoding modules
        H_l = resolution[0] // self.pipe.vae.upsampling_factor
        W_l = resolution[1] // self.pipe.vae.upsampling_factor
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        # Minimal config needed by HF DeepSpeed & integrations (expects model.config.hidden_size and to_dict())
        self.config = MinimalConfig(hidden_size=dim)
        for block in self.pipe.dit.blocks:
            # input: [B, 4T_l, C_l, H_l, W_l]
            block.cam_encoder = nn.Sequential(
                nn.AdaptiveAvgPool3d((48, 16, 16)),
                nn.Flatten(start_dim=2),
                nn.Linear(32 * 32 * 12, dim),
            )
            block.projector = nn.Linear(dim, dim)
            # Initialize the linear layer within the sequential module
            for module in block.cam_encoder:
                if isinstance(module, nn.Linear):
                    module.weight = nn.Parameter(torch.zeros_like(module.weight))
                    module.bias = nn.Parameter(torch.zeros_like(module.bias))
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))

        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
        elif init_ckpt_path is not None:
            state_dict = torch.load(init_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)

        self.freeze_parameters()
        if full_param:
            logger.info("Training all parameters")
            for param in self.pipe.denoising_model().parameters():
                param.requires_grad = True
        else:
            for name, module in self.pipe.denoising_model().named_modules():
                if any(
                    keyword in name
                    for keyword in [
                        "cam_encoder",
                        "projector",
                        "self_attn",
                        "text_embedding",
                    ]
                ):
                    for param in module.parameters():
                        param.requires_grad = True
        # Convert to float32
        for name, p in self.pipe.denoising_model().named_parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
                p.grad = None

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        logger.info(f"Total number of trainable parameters: {trainable_params}")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def get_inputs(self, **inputs):
        text = inputs.get("text")
        video = inputs.get("video")
        # TODO: camera can be calculated from extrinsics_c2w
        camera = inputs.get("camera")
        action_7d = inputs.get("action_7d")
        extrinsics_c2w = inputs.get("extrinsics")
        intrinsics_c2w = inputs.get("intrinsics")
        # Ensure all inputs are on the correct device and dtype
        device = self.device
        video = video.to(dtype=self.pipe.torch_dtype, device=device)
        camera = camera.to(dtype=self.pipe.torch_dtype, device=device)
        action_7d = action_7d.to(dtype=self.pipe.torch_dtype, device=device)
        extrinsics_c2w = extrinsics_c2w.to(dtype=self.pipe.torch_dtype, device=device)
        intrinsics_c2w = intrinsics_c2w.to(dtype=self.pipe.torch_dtype, device=device)
        return text, video, camera, action_7d, extrinsics_c2w, intrinsics_c2w

    def _batch_template(self, inputs):
        """The modality set Pi this batch's sequence is packed from.

        Batch-level, not per-sample: different templates give different segment counts, hence
        different sequence lengths, which cannot share one packed tensor. `per_device_train_
        batch_size=1` is the recipe, so this is a guard rather than a restriction -- but it
        must be a loud one, because a silently-picked template would train a layout the
        prompts of the other samples do not describe.
        """
        names = inputs.get("template")
        if not names:
            return ("video", ACTION)  # datasets that predate templates (bridge/droid/rlbench)
        if len(set(names)) != 1:
            raise ValueError(
                f"mixed templates in one batch: {sorted(set(names))}. Different templates have "
                f"different segment counts and cannot be packed together; use "
                f"per_device_train_batch_size=1 or a single-template mix."
            )
        return parse_template(names[0])

    def forward(self, **inputs):
        device = self.device
        text, video, camera, action_7d, extrinsics_c2w, intrinsics_c2w = self.get_inputs(**inputs)

        # Encode text prompt
        text = [text[i].replace(".", "").replace("_", " ").replace("  ", " ") for i in range(len(text))]
        if random.random() < 0.05:
            text = [""] * len(text)  # null-conditioning
        prompt_emb = self.pipe.encode_prompt(text)  # "context"

        # Prepare action video
        B, _, N, H, W = video.shape
        T = action_7d.shape[1]
        num_views = N // T

        modalities = self._batch_template(inputs)
        # `streams` carries one [B, 3, num_views*T, H, W] tensor per VISUAL modality. Datasets
        # that predate templates only ship `video`, which is exactly Pi = {video, action}.
        streams = inputs.get("streams") or {"video": video}

        missing = [m for m in modalities if m in VISUAL_MODALITIES and m not in streams]
        if missing:
            # The dataset promised this template in the prompt but did not ship its pixels.
            # Failing here beats packing a shorter sequence than the text describes.
            raise KeyError(
                f"template {'+'.join(modalities)} needs streams {missing}, batch has "
                f"{sorted(streams)}"
            )

        # prepare latents: one entry per (visual modality, view)
        visual_latents = {}
        for modality in modalities:
            if modality not in VISUAL_MODALITIES:
                continue
            pixels = streams[modality].to(dtype=self.pipe.torch_dtype, device=device)
            visual_latents[modality] = [
                self.pipe.encode_video(pixels[:, :, v * T : (v + 1) * T, ...], **self.tiler_kwargs)
                for v in range(num_views)
            ]

        # Prepare plucker embeddings
        extrinsics_3x4 = camera.reshape(B, -1, 3, 4)  # [B, 2T, 3, 4]
        plucker_emb = get_plucker_embeddings_torch(extrinsics_3x4, intrinsics_c2w, (H, W))  # [B, 2T, H, W, 6]
        plucker_emb = plucker_emb.permute(0, 4, 1, 2, 3)  # [B, 6, 2T, H, W]
        direction, moment = plucker_emb[:, :3], plucker_emb[:, 3:]
        moment = moment / 3.0
        camera_latents = []  # per view: [B, T_l, C_l, H_l, W_l]
        for v in range(num_views):
            sl = slice(v * T, (v + 1) * T)
            moment_lat = self.pipe.encode_video(moment[:, :, sl, ...], **self.tiler_kwargs)
            direction_lat = self.pipe.encode_video(direction[:, :, sl, ...], **self.tiler_kwargs)
            cam_lat = torch.cat([direction_lat, moment_lat], dim=1)  # [B, 2C, T_l, ...]
            camera_latents.append(cam_lat.permute(0, 2, 1, 3, 4))

        # The action stream is rendered here, not loaded: the same 3D trajectory is reprojected
        # through each view's own camera. A dataset with no usable actions (bridge) keeps the
        # upstream behaviour of collapsing to the visual-only sequence.
        if ACTION in modalities and torch.sum(action_7d**2) == 0:
            modalities = tuple(m for m in modalities if m != ACTION)
        action_latents = None
        if ACTION in modalities:
            action_7d = action_7d.repeat(1, num_views, 1)  # [B, 2T, 7]
            action_5d = project_actions_7d_to_5d_torch_batch(action_7d, extrinsics_c2w, intrinsics_c2w)  # [B, 2T, 5]
            action_video = project_action_5d_to_rgb_torch(action_5d, H, W)  # [B, 2T, H, W, 3]
            action_video = action_video.permute(0, 4, 1, 2, 3)  # [B, 3, 2T, H, W]
            action_video = action_video * 2 - 1
            action_latents = [
                self.pipe.encode_video(action_video[:, :, v * T : (v + 1) * T, ...], **self.tiler_kwargs)
                for v in range(num_views)
            ]

        latents_by_modality = dict(visual_latents)
        if action_latents is not None:
            latents_by_modality[ACTION] = action_latents

        plan = plan_segments(
            modalities,
            num_views,
            is_rlbench="rlbench" in inputs["path"][0],
        )
        latents, camera_emb, masks = assemble(plan, latents_by_modality, camera_latents)

        # Loss computation
        noise = torch.randn_like(latents, device=device)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=device)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

        # Split latents into target and condition
        noisy_latents[masks] = origin_latents[masks]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            cam_emb=camera_emb,
            **prompt_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        # calculate loss across all but batch dimension
        # then average over batch dimension
        valid = (~masks).float()
        se = (noise_pred - training_target).pow(2) * valid
        loss = (se.sum((1, 2, 3, 4)) / valid.sum((1, 2, 3, 4)).clamp_min(1)).mean()

        loss = loss * self.pipe.scheduler.training_weight(timestep)

        return {"loss": loss}

    @property
    def device(self):
        return next(self.parameters()).device


class ActionImagesDataCollator:
    """Forwards an EXPLICIT key list, so dataset-internal fields never reach the model.

    `template` and `streams` are the two additions over upstream. They are what turn the
    prompt from an after-the-fact label into the control signal: without forwarding them,
    `forward` could only infer the task from tensor VALUES (upstream inferred it from whether
    `action_7d` happened to be all zeros), which is precisely the coupling doc 15 §6.2
    identified as the structural problem.
    """

    def __init__(self):
        pass

    def __call__(self, examples):
        batch = {}
        batch["path"] = [example["path"] for example in examples]
        batch["text"] = [example["text"] for example in examples]
        for key in ["video", "camera", "action_7d", "action_8d", "extrinsics", "intrinsics"]:
            batch[key] = torch.stack([example[key] for example in examples])

        if "template" in examples[0]:
            batch["template"] = [example["template"] for example in examples]
        if "streams" in examples[0]:
            keys = set(examples[0]["streams"])
            for example in examples[1:]:
                if set(example["streams"]) != keys:
                    # Would silently drop a modality for part of the batch. forward's
                    # single-template guard catches this too, but failing here names the
                    # actual mismatch.
                    raise ValueError(
                        f"examples disagree on stream keys: {sorted(keys)} vs "
                        f"{sorted(example['streams'])}"
                    )
            batch["streams"] = {
                k: torch.stack([example["streams"][k] for example in examples]) for k in sorted(keys)
            }
        return batch


class ActionImagesTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_actual_model(self, model):
        """Helper method to get the actual model, handling DDP wrapper"""
        if hasattr(model, "module"):
            return model.module
        return model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation for ActionImages training
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        # gather loss from all processes
        loss_gather = torch.tensor(loss.item(), device=self.args.device)
        torch.distributed.all_reduce(loss_gather, op=torch.distributed.ReduceOp.AVG)

        # Log additional metrics to wandb (only on rank 0)
        if self.is_world_process_zero() and wandb.run is not None:
            wandb.log(
                {
                    "train/loss": loss_gather.item(),
                    "train/step": self.state.global_step,
                },
                step=self.state.global_step,
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if return_outputs:
            return loss, outputs
        return loss

    def _save_checkpoint(self, model, trial, metrics=None):
        """Custom checkpoint saving to save only trainable parameters"""
        # Call parent method for other checkpoint data
        super()._save_checkpoint(model, trial)

        if not self.is_world_process_zero():
            return

        # Get checkpoint directory
        output_dir = self.args.output_dir
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)

        checkpoint_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get the actual model (handle DDP wrapper)
        actual_model = self._get_actual_model(model)

        # Save only the trainable parameters
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad, actual_model.pipe.denoising_model().named_parameters()
            )
        )
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = actual_model.pipe.denoising_model().state_dict()

        # Save the model state dict
        model_save_path = os.path.join(checkpoint_dir, f"step{self.state.global_step}.ckpt")
        torch.save(state_dict, model_save_path)
        logger.info(f"Saved checkpoint to {model_save_path}")

        # Log checkpoint save to wandb
        if wandb.run is not None:
            wandb.log(
                {"checkpoint/saved_step": self.state.global_step, "checkpoint/path": model_save_path},
                step=self.state.global_step,
            )


def train(args: TrainingArguments):
    """Training function for RLBench multi-view data using HuggingFace Trainer"""

    training_pkg = assert_own_training_package()

    # Get local rank from distributed environment
    # Set training-specific configurations.
    # FORK: upstream set `report_to = "wandb"` unconditionally, which silently overrode an
    # explicit `--report_to none` and made every smoke run fail at wandb.init with "No API key
    # configured" -- ~3 minutes into the run, after the 12.8GB checkpoint had already loaded.
    # Honour the opt-out; default to wandb as before.
    _report = [args.report_to] if isinstance(args.report_to, str) else list(args.report_to or [])
    use_wandb = bool(_report) and set(_report) != {"none"}
    args.report_to = ["wandb"] if use_wandb else []
    args.save_steps = args.checkpoint_every_n_steps
    args.save_total_limit = args.checkpoint_save_top_k if args.checkpoint_save_top_k > 0 else None

    if args.resume_ckpt_path is None:
        latest_checkpoint = find_latest_checkpoint(args.output_dir)
        print(f"Latest checkpoint: {latest_checkpoint}")
        if latest_checkpoint:
            args.resume_ckpt_path = latest_checkpoint

    # Dataset and DataLoader
    dataset = CombDataset(
        dataset_path=args.dataset_path,
        dataset_specs=args.dataset_name,
        num_frames=args.num_frames,
        frame_interval=1,
        height=args.height,
        width=args.width,
        template_mix=args.template_mix,
        prompt_tag_style=args.prompt_tag_style,
        action_dropout_prob=args.action_dropout_prob,
        strict_getitem=args.strict_getitem,
        variations=args.variations,
    )

    # Create model
    model = ActionImagesModel(
        model_id=args.model_id,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        init_ckpt_path=args.init_ckpt_path,
        resolution=(args.height, args.width),
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
        full_param=args.full_param,
    )

    # Data collator
    data_collator = ActionImagesDataCollator()

    # Create trainer
    trainer = ActionImagesTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Initialize wandb (only on rank 0 to avoid duplicate runs)
    print(f"Local rank: {args.local_rank}, is main process: {trainer.is_world_process_zero()}")
    if use_wandb and trainer.is_world_process_zero():
        wandb.init(
            name=f"actionimages-{args.output_dir.split('/')[-1]}",
            config={
                "num_frames": args.num_frames,
                "height": args.height,
                "width": args.width,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "use_gradient_checkpointing": args.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": args.use_gradient_checkpointing_offload,
                "tiled": args.tiled,
                "tile_size_height": args.tile_size_height,
                "tile_size_width": args.tile_size_width,
                "tile_stride_height": args.tile_stride_height,
                "tile_stride_width": args.tile_stride_width,
                "resume_checkpoint_path": args.resume_ckpt_path,
                "full_param": args.full_param,
                # Which task templates the run actually trained on, how the prompts were
                # written, and which checkout served `training` -- all silent-failure sources,
                # so record them with the run rather than trusting the launch command.
                "template_mix": args.template_mix,
                "prompt_tag_style": args.prompt_tag_style,
                "action_dropout_prob": args.action_dropout_prob,
                "variations": args.variations,
                "dataset_name": args.dataset_name,
                "training_package": training_pkg,
            },
        )
    else:
        logger.addHandler(logging.NullHandler())

    # To activate: TORCH_DETECT_ANOMALY=1 torchrun ... train.py ...
    if os.environ.get("TORCH_DETECT_ANOMALY", "").lower() in ("1", "true", "yes"):
        torch.autograd.set_detect_anomaly(True)
        logger.warning("TORCH_DETECT_ANOMALY is set: autograd anomaly detection ON (training will be much slower).")

    # Start training
    if args.resume_ckpt_path is not None:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final model (only on rank 0)
    if trainer.is_world_process_zero():
        final_save_path = os.path.join(args.output_dir, "final_model.ckpt")
        # Get the actual model (handle DDP wrapper)
        actual_model = trainer._get_actual_model(model)
        state_dict = actual_model.pipe.denoising_model().state_dict()
        torch.save(state_dict, final_save_path)
        logger.info(f"Saved final model to {final_save_path}")

        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    args.seed += get_rank()
    train(args)
