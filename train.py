import copy
import os
import torch
import torch.nn as nn
import random
import warnings
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
    describe_plan,
    fusion_regime_of,
    segment_spans,
    parse_action_mask_mix,
    parse_fusion_mask_mix,
    parse_perception_mask_mix,
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
        perception_mask_mix=None,
        action_mask_mix=None,
        fusion_mask_mix=None,
    ):
        super().__init__()
        # The M and A axes (SEGMENTATION_SCENE_ROLES_PLAN.md §10.3, templates.py). They partition
        # the template space: M applies to templates WITHOUT <action>, A to those with it. Parsed
        # once here rather than per forward: an invalid mix must fail before the 12.8 GB
        # checkpoint load, not 3 minutes in.
        self.perception_mask_mix = parse_perception_mask_mix(perception_mask_mix)
        self.action_mask_mix = parse_action_mask_mix(action_mask_mix)
        # The F axis (templates.py FUSION_MASK_MIX_PRESETS). Stays None unless asked for, and
        # None routes multi-modality templates back to M/A -- that is the `fusion0-anchor`
        # control, not an omission.
        self.fusion_mask_mix = parse_fusion_mask_mix(fusion_mask_mix)

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
        # visual_latents = {
        #     "video": [video_view0_latent, video_view1_latent],
        #     "depth": [depth_view0_latent, depth_view1_latent],
        # }

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
            # UPSTREAM BEHAVIOUR, and a prompt/layout inconsistency wherever it fires. The text
            # was built in the dataset from a Pi that still contained `action`, so dropping the
            # segment here leaves the prompt promising `<action>` for a sequence that has none.
            #
            # It never fires on rlbench_selfgen (measured: 0 of 788 variation0 episodes have a
            # zero action_7d, all quaternions unit-norm). It fires on EVERY bridge sample, whose
            # actions are all zero for want of calibration -- so the multi-dataset path must give
            # bridge an explicit `bridge=video@1.0` menu rather than relying on this fallback.
            # That is MULTIDATASET_DEPTH_PLAN.md's G5, and this is where it would bite.
            #
            # Refusing outright would break the documented bridge fallback, so this warns loudly
            # and once per process instead: a silent mismatch is what must not happen.
            if "<action>" in "".join(text) and not getattr(self, "_warned_action_drop", False):
                self._warned_action_drop = True
                warnings.warn(
                    f"action_7d is all zeros for {inputs['path'][0]}, so the <action> segment was "
                    f"dropped -- but the prompt still says '<action>'. The text now promises a "
                    f"stream the sequence does not contain. Give this dataset an explicit "
                    f"action-free menu (e.g. `bridge=video@1.0` in --template_mix_per_dataset).",
                    RuntimeWarning,
                )
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
            perception_mask_mix=self.perception_mask_mix,
            action_mask_mix=self.action_mask_mix,
            fusion_mask_mix=self.fusion_mask_mix,
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

        out = {"loss": loss}
        out.update(self._segment_diagnostics(se, valid, plan, latents_by_modality))
        return out

    def _segment_diagnostics(self, se, valid, plan, latents_by_modality):
        """Per-segment loss, attributed back to (modality, role) and to sequence position.

        Reported as a RATIO to the step's own overall loss, never as an absolute. The absolute
        scale of a step is set almost entirely by its `timestep` draw -- one uniform sample per
        step, spanning three orders of magnitude -- so absolute per-segment numbers averaged
        over a logging window measure the timestep lottery, not the segments. The ratio divides
        that common factor out; `train/loss` already carries the scale.

        Three readings, each answering a question the aggregate loss cannot:

          segloss/<modality>/<role>  role is `anchored` (frame 0 given) or `absent` (nothing
              given). `depth/absent` vs `depth/anchored` IS the cross-modal completion measure:
              how much worse is depth when it has to come from the other modalities instead of
              from its own first frame.
          segloss_pos/<k>  loss of the k-th segment in packing order. The RoPE-extrapolation
              probe: segments 8-9 sit at temporal positions 88-109, which neither Wan2.2-TI2V-5B
              (pretrained to ~31 latent frames) nor the step125750 warm start (44) has ever
              seen. A systematic rise with k is that extrapolation showing up.
          regime/<name>  one-hot on the F-axis draw. Averaged over a logging window it is the
              realised regime distribution, so a mistyped --fusion_mask_mix is visible in the
              run itself rather than only in the config.

        Costs a handful of reductions over a tensor that is already materialised.
        """
        with torch.no_grad():
            # [B, T]: collapse channels and space, keep the latent-frame axis the spans index.
            se_t = se.sum((1, 3, 4))
            valid_t = valid.sum((1, 3, 4))
            total_valid = valid_t.sum()
            if total_valid <= 0:
                return {}
            overall = se_t.sum() / total_valid
            if not bool(torch.isfinite(overall)) or float(overall) <= 0:
                return {}

            # Reduce on the GPU and cross to the host ONCE. A `.item()` per segment is a
            # separate device sync, and on the 10-segment canvas that is 10 stalls per step
            # bought for nothing -- the values are only ever written to a log.
            kept, sums, counts = [], [], []
            for idx, ((a, b), seg) in enumerate(zip(segment_spans(plan, latents_by_modality), plan)):
                v = valid_t[:, a:b].sum()
                if v <= 0:
                    continue  # fully-given segment: no predicted position, nothing to report
                kept.append((idx, seg))
                sums.append(se_t[:, a:b].sum())
                counts.append(v)
            if not kept:
                return {}
            rels = ((torch.stack(sums) / torch.stack(counts)) / overall).tolist()

            stats = {}
            by_role = {}
            for (idx, seg), rel in zip(kept, rels):
                stats[f"segloss_pos/{idx}"] = rel
                role = "absent" if seg.absent else "anchored"
                by_role.setdefault((seg.modality, role), []).append(rel)
            for (modality, role), vals in by_role.items():
                # Mean over the two views: they are the same modality under the same role, and
                # a per-view split would double the key count to say the same thing.
                stats[f"segloss/{modality}/{role}"] = sum(vals) / len(vals)

            if self.fusion_mask_mix is not None:
                stats[f"regime/{fusion_regime_of(plan)}"] = 1.0

            # The allocator's own high-water mark, not a poll. Sampling nvidia-smi every few
            # seconds MISSES transient peaks: the 20-step fusion smoke sat at 39.7 GB in 149 of
            # 150 samples and touched 44.2 GB in the remaining one, so a poll-based reading
            # understated the peak by 4.5 GB and would have argued for a GPU count that OOMs.
            # Reset each step so the number means "this step", not "since process start".
            if torch.cuda.is_available():
                stats["mem/peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                stats["mem/reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
                torch.cuda.reset_peak_memory_stats()
            return stats

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
        # Everything except "loss" is a rank-local diagnostic scalar (per-segment loss ratios,
        # the F-axis regime one-hot). Not all-reduced on purpose: the plan differs per rank, so
        # a mean over ranks would average `depth/absent` against `depth/anchored` and destroy
        # exactly the contrast the numbers exist to show. Rank 0's stream is a fair sample of
        # the same distribution over thousands of steps.
        diagnostics = {k: v for k, v in outputs.items() if k != "loss"}

        # gather loss from all processes
        loss_gather = torch.tensor(loss.item(), device=self.args.device)
        # Guarded like the barrier above: plain `python train.py` never initialises a process
        # group, and an unguarded all_reduce turns that into
        # "ValueError: Default process group has not been initialized".
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_gather, op=torch.distributed.ReduceOp.AVG)

        # Log additional metrics to wandb (only on rank 0)
        if self.is_world_process_zero() and wandb.run is not None:
            wandb.log(
                {
                    "train/loss": loss_gather.item(),
                    "train/step": self.state.global_step,
                    **diagnostics,
                },
                step=self.state.global_step,
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if return_outputs:
            return loss, outputs
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Skip HF's fp32 `pytorch_model.bin` during checkpointing -- do not write it at all.

        Pruning it after the fact was not enough. On 2026-08-10 arm1 died at checkpoint-750
        with

            PytorchStreamWriter failed writing file data/4: file write failed
            unexpected pos 25662237504 vs 25662237336
            OSError: [Errno 28] No space left on device

        -- 25,662,237,504 bytes is exactly `pytorch_model.bin`. The volume filled up WHILE
        writing a 25.6GB file that `_prune_resume_state` deletes seconds later. Writing then
        deleting only reclaims space; it still requires the space to exist first, so on a
        tight volume the write itself is what kills the run.

        `_save_checkpoint` calls this (transformers 4.57.3, line 13 of its body) and then
        calls `_save_optimizer_and_scheduler` SEPARATELY (line 33), which is what writes
        DeepSpeed's `global_step*/`. Suppressing this call therefore costs nothing that
        resume needs, and the weights are still saved -- as the DiT-only bf16 `stepN.ckpt`
        that `_save_checkpoint` writes below and that everything here actually reads.

        Only the internal checkpointing call is suppressed; an explicit `save_model()` by a
        caller still behaves normally.
        """
        if _internal_call and getattr(self.args, "keep_optimizer_last_only", True):
            return
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _save_checkpoint(self, model, trial, metrics=None):
        """Custom checkpoint saving to save only trainable parameters"""
        # Call parent method for other checkpoint data
        super()._save_checkpoint(model, trial)

        # EVERY rank must execute this barrier. A collective that only some ranks call is not
        # a no-op for the others: NCCL matches collectives by CALL ORDER, not by name, so a
        # rank-0-only barrier pairs with rank 1's next allreduce, training continues on a
        # desynchronised communicator, and the watchdog aborts the job ~10 minutes later with
        # SIGABRT and no useful traceback. That is exactly how arm0 and arm1 died on
        # 2026-08-10 (checkpoint written 17:16:47, abort 17:27:08 -- the 600s NCCL timeout).
        # It belongs here, ABOVE the rank-0 early return, not inside _prune_resume_state.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

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

        # Saves the FULL denoiser state_dict, not just the trainable subset. The comment here
        # used to claim otherwise and the filtered `trainable_param_names` set it built was
        # never applied -- harmless under --full_param True (every parameter is trainable), but
        # it silently makes a partial-finetune checkpoint far larger than the docs imply.
        # Filtering is left off deliberately: the resume path and the released
        # step125750.ckpt both expect a complete denoiser state_dict.
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

        if getattr(self.args, "keep_optimizer_last_only", True):
            self._prune_resume_state(run_dir, keep=checkpoint_dir)

    def _prune_resume_state(self, run_dir, keep):
        """Keep resume state only in the NEWEST checkpoint; every older one keeps just weights.

        Measured on arm0 (2026-08-10), one checkpoint directory is 144GB:

            global_stepN/       108G   DeepSpeed ZeRO-2: fp32 master + Adam m/v + grads
                                       (6.42e9 params x ~16 bytes)
            pytorch_model.bin    24G   HF's fp32 copy of the model (6.42e9 x 4)
            stepN.ckpt           12G   DiT-only bf16 -- the ONLY file eval and warm-start read

        At 250-step cadence over 10k steps that projects to 5.8TB, and it had already consumed
        490GB of a volume with 530GB free. The model is 9% of what gets written.

        `pytorch_model.bin` is deleted unconditionally because nothing in this codebase reads
        it: `find_latest_checkpoint` looks for `step{N}.ckpt`, `ActionImagesModel.__init__`
        loads that `.ckpt`, and DeepSpeed resume reads `global_step*/`. (Same finding as ttd
        DECISIONS.md D-040, which removed it by skipping `super()._save_checkpoint()` entirely
        -- at the cost of losing optimizer resume. Pruning after the fact keeps both.)

        Older `global_step*/` are droppable because they only let you resume from THAT step,
        and a run resumes from its newest checkpoint. Steady state becomes 12GB per retained
        checkpoint plus ~120GB for the resumable tip.
        """
        import glob
        import shutil

        # No barrier here: this method runs on rank 0 ONLY, so a collective inside it
        # deadlocks/desynchronises the job (see the note in _save_checkpoint). The
        # all-ranks barrier that guarantees every shard has landed is already done there.
        keep = os.path.abspath(keep)
        freed = 0
        for ckpt_dir in sorted(glob.glob(os.path.join(run_dir, "checkpoint-*"))):
            if not os.path.isdir(ckpt_dir):
                continue
            if os.path.abspath(ckpt_dir) == keep:
                # NEVER touch the resume tip. Everything this run might need to restart lives
                # here, and a half-pruned tip is unrecoverable. Nothing is gained by trimming
                # it either: `pytorch_model.bin` is no longer written at all (see save_model),
                # so there is nothing redundant left in it to remove.
                continue
            # Resume state for a step nothing will resume from.
            victims = ["pytorch_model.bin", "model.safetensors",
                       "optimizer.pt", "scheduler.pt", "latest", "zero_to_fp32.py",
                       "rng_state_0.pth", "rng_state_1.pth"]
            victims += [os.path.basename(p)
                        for p in glob.glob(os.path.join(ckpt_dir, "global_step*"))]
            for name in victims:
                path = os.path.join(ckpt_dir, name)
                if not os.path.exists(path):
                    continue
                try:
                    if os.path.isdir(path):
                        size = sum(os.path.getsize(os.path.join(r, f))
                                   for r, _, fs in os.walk(path) for f in fs)
                        shutil.rmtree(path)
                    else:
                        size = os.path.getsize(path)
                        os.remove(path)
                    freed += size
                except OSError as e:
                    # Never let cleanup kill a training run that has otherwise succeeded.
                    logger.warning(f"could not prune {path}: {e}")
        if freed:
            logger.info(f"pruned {freed / 1e9:.1f} GB of redundant checkpoint state "
                        f"(resume state kept only in {os.path.basename(keep)})")


def _latest_resumable_checkpoint(output_dir):
    """Newest checkpoint-* directory that DeepSpeed can actually resume from, or None.

    "Resumable" is stricter than find_latest_checkpoint's "has a stepN.ckpt": HF's
    deepspeed_load_checkpoint needs `global_step*/` (the sharded optimizer state) and
    TrainerState needs `trainer_state.json`. --keep_optimizer_last_only deliberately removes
    the former from all but the newest checkpoint, so most of the tree is weights-only.
    """
    if not os.path.isdir(output_dir):
        return None
    best = None
    for item in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", item)
        if not m:
            continue
        d = os.path.join(output_dir, item)
        has_state = any(n.startswith("global_step") and os.path.isdir(os.path.join(d, n))
                        for n in os.listdir(d))
        if has_state and os.path.exists(os.path.join(d, "trainer_state.json")):
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, d)
    return best[1] if best else None



def _step_of(path):
    """step number encoded in a `.../checkpoint-N` dir or a `.../stepN.ckpt` file, else None."""
    if not path:
        return None
    m = re.search(r"step(\d+)\.ckpt$", str(path)) or re.fullmatch(
        r".*checkpoint-(\d+)", str(path).rstrip("/")
    )
    return int(m.group(1)) if m else None


def _check_resume_consistency(output_dir, weights_path, resumable_dir, allow_step_restart=False):
    """Refuse the two ways resume can silently do the wrong thing.

    Weights and DeepSpeed state are chosen by two independent scans
    (`find_latest_checkpoint` picks the highest stepN.ckpt, `_latest_resumable_checkpoint`
    picks the highest checkpoint-N that still has global_step*/), and nothing made them agree.

    A) mismatched steps -- the log says it loaded step 6000 while DeepSpeed rewinds model,
       optimizer and scheduler to 3000. Silent, and the loss curve looks merely "a bit odd".
    B) weights-only resume into a non-empty output_dir -- the step counter restarts at 0, so
       the next save writes checkpoint-3000/ ON TOP of the existing one. The old warning said
       "step numbers restart from 0" but did nothing to stop the overwrite.
    """
    w_step, r_step = _step_of(weights_path), _step_of(resumable_dir)
    if w_step is not None and r_step is not None and w_step != r_step:
        raise RuntimeError(
            f"resume mismatch: weights are step {w_step} ({weights_path}) but the newest "
            f"DeepSpeed state is step {r_step} ({resumable_dir}). DeepSpeed would rewind the "
            f"model to {r_step} while the log claims {w_step}. Delete the stale checkpoint dir, "
            f"or point --resume_ckpt_path at step{r_step}.ckpt explicitly."
        )
    if w_step is not None and r_step is None and not allow_step_restart:
        existing = sorted(
            d for d in os.listdir(output_dir)
            if re.fullmatch(r"checkpoint-\d+", d)
        ) if os.path.isdir(output_dir) else []
        if existing:
            raise RuntimeError(
                f"weights-only resume from step {w_step}, but {output_dir} already holds "
                f"{existing}. The step counter restarts at 0, so the next save would overwrite "
                f"those directories and their names would stop meaning total steps. Use a fresh "
                f"--output_dir, or pass --allow_step_restart True if overwriting is intended."
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
        frame_interval=args.frame_interval,
        height=args.height,
        width=args.width,
        template_mix=args.template_mix,
        prompt_tag_style=args.prompt_tag_style,
        action_dropout_prob=args.action_dropout_prob,
        strict_getitem=args.strict_getitem,
        variations=args.variations,
        segmentation_mode=args.segmentation_mode,
        template_mix_per_dataset=args.template_mix_per_dataset,
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
        perception_mask_mix=args.perception_mask_mix,
        action_mask_mix=args.action_mask_mix,
        fusion_mask_mix=args.fusion_mask_mix,
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
            # Honour --run_name when the launcher set one; the derived name is only a
            # fallback. HfArgumentParser defaults run_name to output_dir, so treat that as
            # "not set" rather than as an explicit choice.
            name=(
                args.run_name
                if getattr(args, "run_name", None) and args.run_name != args.output_dir
                else f"actionimages-{args.output_dir.split('/')[-1]}"
            ),
            config={
                "num_frames": args.num_frames,
                # FORK: the temporal span of a window is num_frames * frame_interval, not
                # num_frames. Two runs that differ only in this are otherwise indistinguishable
                # in wandb, so it has to be logged alongside num_frames.
                "frame_interval": args.frame_interval,
                "window_span_seconds": (args.num_frames - 1) * args.frame_interval / 20.0,
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
                "segmentation_mode": args.segmentation_mode,
                "perception_mask_mix": args.perception_mask_mix,
                # Logged RESOLVED, not as passed: `None` in the config would read as "no mask
                # policy" when it actually means the A0/M0 defaults, and the whole point of
                # recording these is that a checkpoint must be askable with the regime it saw.
                "action_mask_mix": args.action_mask_mix,
                "action_mask_mix_resolved": parse_action_mask_mix(args.action_mask_mix),
                "perception_mask_mix_resolved": parse_perception_mask_mix(args.perception_mask_mix),
                "fusion_mask_mix": args.fusion_mask_mix,
                "fusion_mask_mix_resolved": parse_fusion_mask_mix(args.fusion_mask_mix),
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

    # Start training.
    # `resume_from_checkpoint=True` makes HF pick the highest-numbered checkpoint-* directory
    # and hand it to DeepSpeed, which REQUIRES a `global_step*/` inside it. But
    # find_latest_checkpoint (above) only requires a `stepN.ckpt`, and --keep_optimizer_last_only
    # strips `global_step*/` from every checkpoint except the newest. So as soon as the newest
    # checkpoint is removed -- by manual disk cleanup, by a failed write, or by rotation -- the
    # two disagree, and the run dies with
    #     ValueError: Can't find a valid checkpoint at .../checkpoint-250
    # even though the WEIGHTS at that step loaded fine a moment earlier (observed 2026-08-10).
    # Resolve the disagreement here: resume optimizer state only from a directory that really
    # has it, and otherwise continue from the weights alone rather than refusing to start.
    resumable = _latest_resumable_checkpoint(args.output_dir)
    _check_resume_consistency(
        args.output_dir, args.resume_ckpt_path, resumable,
        allow_step_restart=getattr(args, "allow_step_restart", False),
    )
    if resumable is not None:
        logger.info(f"resuming optimizer/scheduler state from {resumable}")
        trainer.train(resume_from_checkpoint=resumable)
    else:
        if args.resume_ckpt_path is not None:
            logger.warning(
                f"{args.resume_ckpt_path} gives the WEIGHTS, but no checkpoint directory under "
                f"{args.output_dir} still has DeepSpeed resume state (global_step*/). Continuing "
                f"from those weights with a fresh optimizer and step counter. Reported step "
                f"numbers will restart from 0 -- account for that when plotting against step."
            )
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
    # NOTE: do NOT offset the seed per rank. `args.seed` is what Trainer uses to seed the
    # RandomSampler's generator, and Accelerate's batch sharding assumes every rank draws the
    # SAME global permutation and then takes its own slice. Giving rank r seed 42+r makes the
    # permutations differ, so the slices stop partitioning: measured on a 2-rank / 1000-sample
    # run, 258 indices were visited by both ranks and 258 were never visited at all.
    # Per-rank randomness is already handled where it belongs -- transformers seeds each
    # dataloader worker with `num_workers * rank + init_seed` (trainer_utils.seed_worker,
    # wired at trainer.py with rank=args.process_index), so frame windows, view pairs and
    # template draws still differ across ranks with this line gone.
    train(args)
