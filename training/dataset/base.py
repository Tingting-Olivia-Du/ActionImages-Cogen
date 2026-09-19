import os
import json
import pickle
import hashlib
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from torchvision.transforms import v2
import imageio
import random
import traceback
from einops import rearrange
from training.utils import get_relative_pose_batch, CenterCropToAspect, convert_intrinsics_after_center_crop_resize
from training.templates import (
    ACTION,
    assert_menu_supported,
    draw_template,
    format_template,
    parse_template_mix,
    prompt_prefix,
)


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for multi-view video datasets.

    This template provides a common interface and shared functionality
    for datasets used in multi-view video generation tasks.
    """

    # FORK: which modalities this tree can actually put in a segment. The template menu is
    # validated against it at construction, so asking bridge for `video+action` fails loudly
    # at startup instead of silently degrading to `video` for the whole run.
    AVAILABLE_MODALITIES: Tuple[str, ...] = ("video", ACTION)
    # Whether BaseDataset.getitem applies the template itself. RLBenchSelfgenDataset sets this
    # False because it overrides getitem to also build depth/segmentation streams, and applying
    # the prompt prefix in both places would emit the tags twice.
    TEMPLATE_IN_BASE: bool = True

    def __init__(
        self,
        base_path: str,
        num_frames: int = 81,
        frame_interval: int = 1,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ):
        """
        Initialize the base dataset.

        Args:
            base_path: Root path to the dataset
            num_frames: Number of frames to sample from each video
            frame_interval: Interval between sampled frames
            height: Target height for video frames
            width: Target width for video frames
            **kwargs: Additional dataset-specific parameters
        """
        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.height = height
        self.width = width

        # FORK: template machinery. Guarded by hasattr because RLBenchSelfgenDataset parses its
        # own menu BEFORE delegating here -- overwriting it with the default would silently turn
        # a depth arm into a plain video+action arm.
        if not hasattr(self, "_templates"):
            self._init_templates(
                template_mix=kwargs.pop("template_mix", "video+action@1.0"),
                prompt_tag_style=kwargs.pop("prompt_tag_style", "explicit"),
                action_dropout_prob=kwargs.pop("action_dropout_prob", 0.1),
                strict_getitem=kwargs.pop("strict_getitem", False),
            )

        # Initialize frame processing pipeline
        self.frame_process = v2.Compose(
            [
                CenterCropToAspect(height, width),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Initialize dataset-specific data structures
        self.episodes = []

        # Load dataset (with simple hash-based cache under ./cache)
        if not self._try_load_cache():
            self._load_dataset()
            self._save_cache()

    # ==== BEGIN: abstract methods ====
    @abstractmethod
    def _load_dataset(self) -> None:
        pass

    @abstractmethod
    def get_8d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> torch.Tensor:
        # Default to null 8D actions
        if frame_indices is None:
            length = self.num_frames
        else:
            length = len(frame_indices)
        # [x, y, z, qw, qx, qy, qz, openness]
        return np.zeros((length, 8), dtype=np.float32)

    @abstractmethod
    def get_7d_action(self, episode_path: str, frame_indices: Optional[List[int]] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def get_instruction(self, **kwargs) -> str:
        pass

    @abstractmethod
    def get_all_views(self, **kwargs) -> str:
        pass

    # ======== END ========

    # ======== FORK: task templates ========
    def _init_templates(
        self,
        template_mix: str = "video+action@1.0",
        prompt_tag_style: str = "explicit",
        action_dropout_prob: float = 0.1,
        strict_getitem: bool = False,
    ) -> None:
        assert_menu_supported(type(self).__name__, template_mix, self.AVAILABLE_MODALITIES)
        self._templates, self._template_cum = parse_template_mix(template_mix)
        self.prompt_tag_style = prompt_tag_style
        self.action_dropout_prob = float(action_dropout_prob)
        self.strict_getitem = bool(strict_getitem)

    def _apply_template(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Attach `template` / `streams` / prompt tags to a plain visual sample.

        This is the version for trees that own no perception GT, i.e. every modality they can
        serve is either `video` (already in the sample) or `action` (rendered inside forward
        from action_7d, so it never needs pixels here). RLBenchSelfgenDataset does the richer
        variant that also loads depth/segmentation.
        """
        mods = draw_template(self._templates, self._template_cum)
        # Upstream's action gate (train.py:286 `random.random() < 0.9`) lives in the dataset so
        # the prompt and the segment layout are decided together.
        if ACTION in mods and random.random() < self.action_dropout_prob:
            mods = tuple(m for m in mods if m != ACTION)
        sample["streams"] = {"video": sample["video"]}
        sample["template"] = format_template(mods)
        sample["visual_modality"] = "video"
        sample["text"] = prompt_prefix(mods, None, self.prompt_tag_style) + sample["text"]
        return sample

    def select_views(self, candidate_indices: List[int], view_indices: List[int]) -> List[int]:
        return [candidate_indices[i] for i in view_indices]

    @abstractmethod
    def getitem(self, index: int) -> Dict[str, Any]:
        pass

    def getitem(self, index: int) -> Dict[str, Any]:
        """Get a single RLBench data sample."""
        episode_info = self.episodes[index]
        episode_path = episode_info["path"]

        # Get text description
        text = self.get_instruction(episode_info)

        # ==== Load videos from different views ====
        videos, frame_indices, extrinsics, intrinsics, raw_resolutions, has_global_extrinsics = self.get_all_views(
            episode_path
        )
        intrinsics = convert_intrinsics_after_center_crop_resize(
            intrinsics, raw_resolutions, [(self.height, self.width)] * len(intrinsics)
        )

        # ==== Load action data aligned with video frames ====
        actions_8d = self.get_8d_action(episode_path, frame_indices=frame_indices)
        actions_7d = self.get_7d_action(episode_path, frame_indices=frame_indices)

        # ==== view selection start ====
        # Randomly select condition and target views
        if len(videos) == 1:
            view_indices = [0, 0]
        else:
            view_indices = random.sample(range(len(videos)), 2)
        video_cond, video_target = self.select_views(videos, view_indices)
        # Concatenate condition and target videos
        combined_video = torch.cat([video_cond, video_target], dim=1)  # C, 2*T, H, W

        # ==== Calculate relative camera poses ====
        extrinsics_cond, extrinsics_target = self.select_views(extrinsics, view_indices)
        intrinsics_cond, intrinsics_target = self.select_views(intrinsics, view_indices)
        base_extrinsics = extrinsics_cond[0]  # 3, 4
        relative_poses_target = get_relative_pose_batch(base_extrinsics, extrinsics_target)  # T, 3, 4
        relative_poses_cond = get_relative_pose_batch(base_extrinsics, extrinsics_cond)  # T, 3, 4
        # Concatenate condition and target relative poses
        relative_poses_target = rearrange(torch.from_numpy(relative_poses_target), "t c d -> t (c d)")
        relative_poses_cond = rearrange(torch.from_numpy(relative_poses_cond), "t c d -> t (c d)")
        camera_poses = torch.cat([relative_poses_cond, relative_poses_target], dim=0)
        camera_poses = camera_poses.to(torch.float32)  # T * 2, 12
        extrinsics = torch.from_numpy(np.concatenate([extrinsics_cond, extrinsics_target], axis=0))  # T * 2, 3, 4
        intrinsics = torch.from_numpy(np.concatenate([intrinsics_cond, intrinsics_target], axis=0))  # T * 2, 3, 3

        # ==== to torch ====
        actions_7d = torch.from_numpy(actions_7d)
        actions_8d = torch.from_numpy(actions_8d)

        sample = {
            "text": text,  # str
            "video": combined_video,  # 3, T * 2, H, W
            "camera": camera_poses,  # T * 2, 12
            "extrinsics": extrinsics,  # T * 2, 3, 4 - camera to world
            "intrinsics": intrinsics,  # T * 2, 3, 3
            "action_7d": actions_7d,  # T, 7
            "action_8d": actions_8d,  # T, 8
            "path": episode_path,  # str
            # FORK: provenance of THIS sample -- which frame window and which two view dirs the
            # pixels above were built from. Subclasses that add a second modality (depth/seg)
            # must read it from the same window and the same views, and re-deriving that later
            # by calling get_all_views()/random.sample() again yields a DIFFERENT draw. The
            # collator forwards an explicit key list, so these never reach the model.
            # Datasets whose get_all_views does not record view dirs (bridge, droid) get [].
            "frame_indices": list(frame_indices) if frame_indices is not None else [],
            "view_indices": list(view_indices),  # [cond, target], indexes into view_dirs
            "view_dirs": list(getattr(self, "_last_view_dirs", None) or []),
        }
        if self.TEMPLATE_IN_BASE:
            sample = self._apply_template(sample)
        return sample

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item with retry logic for robustness.

        FORK: upstream retried 50 times and printed `str(e)`, then re-rolled the index. That
        turns a SYSTEMATIC failure (e.g. a perception path that raises for every episode) into
        log spam plus silently serving whatever sample happens to succeed -- training looks
        healthy while learning something else entirely. A transient IO error fails once; a
        systematic one fails on every index, so 5 consecutive failures means "broken", not
        "unlucky". Keep the retry for the transient case, raise with the first full traceback
        for the systematic one. `strict_getitem=True` disables retries entirely (smoke runs).
        """
        if getattr(self, "strict_getitem", False):
            return self.getitem(index)

        first_traceback = None
        for attempt in range(50):
            try:
                return self.getitem(index)
            except Exception as e:
                if first_traceback is None:
                    first_traceback = traceback.format_exc()
                self._getitem_failures = getattr(self, "_getitem_failures", 0) + 1
                print(f"Error in getitem: {e}")
                if attempt >= 4:
                    raise RuntimeError(
                        f"{type(self).__name__}: 5 consecutive getitem failures "
                        f"({self._getitem_failures} total this process). This is systematic, not "
                        f"transient -- first traceback:\n{first_traceback}"
                    )
                index = random.randint(0, len(self.episodes) - 1)
        raise Exception(f"Failed to get item after 50 attempts: {index}")

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.episodes)

    # ======== Simple caching helpers ========
    def _get_cache_key(self) -> str:
        """Compute a stable hash key from dataset identity and init parameters."""
        key_obj = {
            "class": self.__class__.__name__,
            "base_path": os.path.abspath(self.base_path),
            "num_frames": self.num_frames,
            "frame_interval": self.frame_interval,
            "height": self.height,
            "width": self.width,
        }
        payload = json.dumps(key_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _get_cache_dir(self) -> str:
        return os.path.join(".", "cache")

    def _get_cache_path(self) -> str:
        cache_name = f"{self.__class__.__name__}-{self._get_cache_key()}.pkl"
        return os.path.join(self._get_cache_dir(), cache_name)

    def _try_load_cache(self) -> bool:
        try:
            cache_path = self._get_cache_path()
            if os.path.isfile(cache_path):
                with open(cache_path, "rb") as f:
                    self.episodes = pickle.load(f)
                return isinstance(self.episodes, list) and len(self.episodes) >= 0
            return False
        except Exception:
            return False

    def _save_cache(self) -> None:
        try:
            cache_dir = self._get_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            with open(self._get_cache_path(), "wb") as f:
                pickle.dump(self.episodes, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Silently ignore caching errors to avoid disrupting dataset loading
            pass


class CombDataset(torch.utils.data.Dataset):
    """
    Combine multiple datasets and sample from them according to provided ratios.

    Input specs format: list of strings like "{name}@{ratio}", e.g., ["bridge@0.7", "rlbench@0.3"].

    - Supported names: "bridge", "rlbench" (extendable).
    - The dataset length is defined as the sum of underlying dataset lengths.
    - __getitem__ ignores the incoming index for fairness and samples a source
      dataset by ratio, then a random episode within that dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_specs: str,
        num_frames: int = 81,
        frame_interval: int = 1,
        height: int = 512,
        width: int = 512,
        template_mix: str = "video+action@1.0",
        prompt_tag_style: str = "explicit",
        action_dropout_prob: float = 0.1,
        strict_getitem: bool = False,
        variations: str = "all",
        segmentation_mode: str = "referring",
        template_mix_per_dataset: str = "",
    ) -> None:
        super().__init__()

        if "@" not in dataset_specs:
            dataset_specs = f"{dataset_specs}@1.0"

        dataset_specs = [s.strip() for s in dataset_specs.split(",") if s.strip()]

        # FORK: the feasible template menu is a property of the data, not of the run, so each
        # tree gets its own. Unlisted trees fall back to the global --template_mix.
        from training.templates import parse_per_dataset_template_mix

        _selected = [sp.split("@", 1)[0].strip().lower() for sp in dataset_specs if sp.strip()]
        per_ds = parse_per_dataset_template_mix(template_mix_per_dataset, template_mix, known=_selected)

        def mix_for(name: str) -> str:
            return per_ds.get(name, template_mix)

        # Lazy import to avoid circular imports with BaseDataset subclasses
        from training.dataset.bridge import BridgeMVDataset
        from training.dataset.rlbench import RLBenchMVDataset
        from training.dataset.droid import DROIDMVDataset
        from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset
        from training.dataset.maniskill3 import ManiSkill3Dataset
        from training.dataset.behavior import BehaviorDataset

        name_to_ctor = {
            "bridge": lambda: BridgeMVDataset(
                base_path=os.path.join(dataset_path, "bridge"),
                num_frames=num_frames,
                # Pinned to 1, like droid is pinned to 4, because the stride that makes sense
                # is a property of the SOURCE's frame rate, not of the run. Bridge is natively
                # 5 Hz, so 41 frames already span 8.0 s -- the same window the official
                # checkpoint was trained on. Raising it is not merely unnecessary, it is
                # destructive: bridge episodes have a median of 32 frames, so frame_interval=3
                # needs 121 and NO episode has them. load_video_frames then clamps, and a
                # typical sample degenerates to 12 distinct frames followed by 29 copies of the
                # last one -- i.e. "predict a video that freezes", on 10% of all steps.
                frame_interval=1,
                height=height,
                width=width,
                template_mix=mix_for("bridge"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
            ),
            "rlbench": lambda: RLBenchMVDataset(
                base_path=os.path.join(dataset_path, "rlbench"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
                template_mix=mix_for("rlbench"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
            ),
            "droid": lambda: DROIDMVDataset(
                base_path=os.path.join(dataset_path, "droid"),
                num_frames=num_frames,
                frame_interval=4,
                height=height,
                width=width,
                template_mix=mix_for("droid"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
            ),
            # FORK: self-generated RLBench with per-frame depth/mask GT alongside RGB. The only
            # source that can serve a perception modality; `template_mix` decides which task
            # templates it draws from. With the default "video+action@1.0" it behaves like
            # `rlbench` on a different data tree.
            "rlbench_selfgen": lambda: RLBenchSelfgenDataset(
                base_path=os.path.join(dataset_path, "rlbench_selfgen"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
                template_mix=mix_for("rlbench_selfgen"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
                variations=variations,
                segmentation_mode=segmentation_mode,
            ),
            # FORK: the same loader against the 512x512 tree (ttd/data/rlbench_selfgen_512),
            # rendered natively at the official ActionImages resolution rather than the 256 of
            # `rlbench_selfgen`. A SEPARATE NAME rather than a repointed `data/rlbench_selfgen`
            # symlink, deliberately: _load_dataset globs episode paths that still contain the
            # symlink component and only resolve at open() time, so flipping the symlink under a
            # live 256 run silently swaps its data mid-training instead of failing. Two names let
            # a 256 arm and a 512 arm run side by side.
            # NOTE its episodes are NOT the 256 tree's episodes at higher resolution -- different
            # seeds, different scene layouts -- so a 256 arm's r_peak is not a baseline for a 512
            # arm. Re-run the control arm on this tree (see train_arm.sh, "WHY Arm-0 IS NOT
            # OPTIONAL").
            "rlbench_selfgen_512": lambda: RLBenchSelfgenDataset(
                base_path=os.path.join(dataset_path, "rlbench_selfgen_512"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
                template_mix=mix_for("rlbench_selfgen_512"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
                variations=variations,
                segmentation_mode=segmentation_mode,
            ),
            # FORK: 512x512 WITH Colosseum-style domain randomisation -- camera pose (spherical
            # orbit with a per-task look-at), table colour/texture, background texture and light
            # colour. See ActionImages-Cogen/scripts/colosseum_aug.py for the measurement that
            # fixed each range against the official ActionImages release, and for the three
            # factors deliberately left out (object colour breaks "close the RED jar"-style
            # instruction grounding; distractors need a spawn boundary stock RLBench lacks;
            # physics is not randomised in the official release either).
            #
            # This is the only self-gen tree whose visual diversity is comparable to the official
            # one. `rlbench_selfgen` (256) and `rlbench_selfgen_512` vary the back wall ONLY: the
            # table, the lights and all four cameras are byte-identical across every episode.
            # 1028 episodes, 788 of them variation0.
            #
            # NOTE `rlbench_selfgen_512` above is currently a DANGLING symlink -- that tree was
            # deleted to free disk. Selecting it will fail at glob time. It is kept in the table
            # because regenerating it is the only way to get an un-augmented 512 control arm.
            "rlbench_selfgen_512_aug": lambda: RLBenchSelfgenDataset(
                base_path=os.path.join(dataset_path, "rlbench_selfgen_512_aug"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
                template_mix=mix_for("rlbench_selfgen_512_aug"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
                variations=variations,
                segmentation_mode=segmentation_mode,
            ),
            # FORK: the SCALED-UP tree, same recipe as rlbench_selfgen_512_aug (same 16 tasks,
            # same Colosseum randomisation, same 512) with 250 variation0 seeds per task instead
            # of 50 -- ~4000 training episodes against 788.
            #
            # A SEPARATE NAME, not more episodes in the tree above, and that is not a disk
            # decision. Training globs the whole tree under `--variations 0`, so appending here
            # would silently change what arm0..arm7u were trained on and destroy every
            # arm-to-arm comparison in the paper -- the same class of silent substitution the
            # `_fi<N>` / `_sr` / tree suffixes in train_arm.sh's OUT path exist to prevent.
            #
            # It exists because 788 episodes is the binding constraint on the fusion arm: a
            # 10-segment canvas (video+depth+segmentation+normal+action) extracts far more
            # supervision per sample, but supervision per EPISODE is not episode diversity, and
            # train_arm.sh already cites Argus (CVPR 2025) Tab.13 for four auxiliary streams
            # being the ceiling at the old scale. Required before any from-Wan-base run.
            #
            # Task set is deliberately UNCHANGED: `scene_roles` resolves handles through
            # per-task rules in training/percep/scene_segments_gen.py TASK_ROLES (22 tasks
            # covered), and _check_scene_roles_ready raises on a single `unknown` pixel. Adding
            # tasks means writing those rules first; adding seeds costs nothing.
            "rlbench_selfgen_512_aug_wide": lambda: RLBenchSelfgenDataset(
                base_path=os.path.join(dataset_path, "rlbench_selfgen_512_aug_wide"),
                num_frames=num_frames,
                frame_interval=frame_interval,
                height=height,
                width=width,
                template_mix=mix_for("rlbench_selfgen_512_aug_wide"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
                variations=variations,
                segmentation_mode=segmentation_mode,
            ),
            # FORK: ManiSkill3 -- official ManiSkill demonstrations replayed into the selfgen
            # layout by scripts/maniskill3_gen.py (4 randomized 512 cameras per episode, full
            # five-modality GT; see training/dataset/maniskill3.py for the byte-level contract).
            # frame_interval is PINNED to 1, like bridge's, because the sensible stride is a
            # property of the SOURCE: episodes are native 20 Hz and 49-103 frames long, so the
            # RLBench run default of 3 needs 121-frame windows and would degrade nearly every
            # sample into a freeze-frame tail. At 1, a 41-frame window (~2 s) always fits.
            # The `variations` split applies unchanged: the tree writes variation0 = training
            # seeds, variation1 = held-out seeds of the same tasks.
            "maniskill3": lambda: ManiSkill3Dataset(
                base_path=os.path.join(dataset_path, "maniskill3"),
                num_frames=num_frames,
                frame_interval=1,
                height=height,
                width=width,
                template_mix=mix_for("maniskill3"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
                variations=variations,
                segmentation_mode=segmentation_mode,
            ),
            # FORK: BEHAVIOR-1K -- 2026-challenge raw demos replayed into the selfgen layout
            # by behavior_demos/render_selfgen.py (manipulation chunks only, 4 randomized 512
            # cameras; see training/dataset/behavior.py for the byte-level contract).
            # frame_interval PINNED to 1 for the same source-property reason as maniskill3:
            # chunks are rendered at 15 Hz (raw 30 Hz teleop, stride 2) and 48-240 frames
            # long, so a 41-frame window (~2.7 s) always fits; the run default of 3 would
            # need 121-frame windows and turn most chunks into freeze-frame tails.
            # variation0 = train demos, variation1 = held-out demos of the same tasks.
            "behavior": lambda: BehaviorDataset(
                base_path=os.path.join(dataset_path, "behavior"),
                num_frames=num_frames,
                frame_interval=1,
                height=height,
                width=width,
                template_mix=mix_for("behavior"),
                prompt_tag_style=prompt_tag_style,
                action_dropout_prob=action_dropout_prob,
                strict_getitem=strict_getitem,
                variations=variations,
                segmentation_mode=segmentation_mode,
            ),
        }

        datasets: List[torch.utils.data.Dataset] = []
        ratios: List[float] = []
        names: List[str] = []

        for spec in dataset_specs:
            if "@" not in spec:
                raise ValueError(f"Invalid dataset spec: {spec}. Expected format 'name@ratio'.")
            name, ratio_str = spec.split("@", 1)
            name = name.strip().lower()
            try:
                ratio = float(ratio_str)
            except Exception:
                raise ValueError(f"Invalid ratio in dataset spec: {spec}")
            if ratio <= 0:
                continue  # skip non-positive ratios

            if name not in name_to_ctor:
                raise ValueError(f"Unknown dataset name '{name}' in spec '{spec}'.")

            datasets.append(name_to_ctor[name]())
            ratios.append(ratio)
            names.append(name)

        if not datasets:
            raise ValueError("No valid datasets constructed from specs.")

        # A positive-weight dataset with zero episodes silently redirects its share to whichever
        # other dataset __getitem__ finds first (see the `local_len == 0` fallback below), so the
        # realised mix stops matching the requested one with nothing in the log saying so. A
        # broken symlink or a half-finished preprocessing run is exactly how this happens.
        empty = [n for n, d in zip(names, datasets) if len(d) == 0]
        if empty:
            raise ValueError(
                f"dataset(s) {empty} were given a positive ratio but contain 0 episodes. "
                f"Check the data/ symlinks and that preprocessing finished; refusing to "
                f"silently redistribute their share to the other datasets."
            )

        # Normalize ratios and build cumulative distribution for sampling
        ratio_sum = float(sum(ratios))
        if ratio_sum <= 0:
            raise ValueError("Sum of ratios must be positive.")
        probs = [r / ratio_sum for r in ratios]
        cum_probs = []
        c = 0.0
        for p in probs:
            c += p
            cum_probs.append(c)
        # Ensure the last value is exactly 1.0 to avoid floating issues
        cum_probs[-1] = 1.0

        self._datasets = datasets
        self._names = names
        self._probs = probs
        self._cum_probs = cum_probs
        self._lengths = [len(ds) for ds in datasets]
        self._total_length = int(sum(self._lengths))

    def __len__(self) -> int:
        # Use the aggregate size as epoch length. Trainer will sample indices randomly.
        return max(self._total_length, 1)

    def _uniform_from_index(self, index: int, salt: int = 0) -> float:
        """Deterministically map an integer index (+ optional salt) to [0, 1).

        This allows __getitem__ to be purely index-driven, which plays nicely
        with DistributedSampler to avoid cross-rank duplication.
        """
        # Use a stable 64-bit hash and scale to [0, 1)
        key = f"{index}_{salt}".encode("utf-8")
        h = hashlib.blake2b(key, digest_size=8).digest()
        u64 = int.from_bytes(h, byteorder="big", signed=False)
        return (u64 & ((1 << 64) - 1)) / float(1 << 64)

    def _choose_dataset_index(self, r: Optional[float] = None) -> int:
        # If r is not provided, fall back to RNG (non-DDP/inference cases)
        if r is None:
            r = random.random()
        # Binary search over cumulative probabilities
        lo, hi = 0, len(self._cum_probs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= self._cum_probs[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Deterministically map the sampler-provided index to a dataset id
        # so that different ranks (with disjoint indices from DistributedSampler)
        # do not collide on the same underlying sample.
        ds_idx = self._choose_dataset_index(r=self._uniform_from_index(index, salt=0))
        ds = self._datasets[ds_idx]

        # Deterministically pick an item from the chosen dataset based on index
        # while still looking "random" but remaining index-driven.
        local_len = len(ds)
        if local_len == 0:
            # Fallback: try another dataset
            for alt_idx, alt_ds in enumerate(self._datasets):
                if len(alt_ds) > 0:
                    ds_idx = alt_idx
                    ds = alt_ds
                    local_len = len(ds)
                    break
        if local_len == 0:
            raise IndexError("All underlying datasets are empty.")

        # Use a second deterministic uniform mapped to [0, local_len)
        r_local = self._uniform_from_index(index, salt=1)
        local_index = int(r_local * local_len)
        if local_index >= local_len:
            local_index = local_len - 1
        sample = ds[local_index]

        # Optionally tag the sample with its source dataset name
        if isinstance(sample, dict):
            sample = dict(sample)
            sample["_source_dataset"] = self._names[ds_idx]
        return sample
