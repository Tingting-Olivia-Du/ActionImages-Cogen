"""Generate lossless Figure 1/2 assets with explicit episode/view/time provenance.

Run with CUDA_VISIBLE_DEVICES=0 set before Python starts. No future ground-truth
pixels or poses are supplied to the sampler: each input clip repeats its anchor.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import json
import os
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import av
import numpy as np
import torch
from PIL import Image
from training.dataset import RLBenchSelfgenDataset
from training.utils import project_actions_7d_to_5d_torch_batch, project_action_5d_to_rgb_torch


class FigureDataset(RLBenchSelfgenDataset):
    def __init__(self, episode, start, instruction):
        self.figure_episode = Path(episode).resolve()
        self.figure_start = start
        self.figure_instruction = instruction
        super().__init__(base_path=str(self.figure_episode.parents[3]), num_frames=41,
                         frame_interval=3, height=512, width=512,
                         template_mix='video+action@1.0', prompt_tag_style='explicit',
                         action_dropout_prob=0, strict_getitem=True,
                         segmentation_mode='scene_roles')

    def _try_load_cache(self):
        return False

    def _save_cache(self):
        pass

    def _load_dataset(self):
        p = self.figure_episode
        self.episodes = [dict(path=str(p), task=p.parents[2].name,
                              variation=p.parents[1].name)]

    def get_instruction(self, episode_info):
        return self.figure_instruction

    def get_all_views(self, episode_path):
        total = len(self._load_8d_actions_base(episode_path))
        indices = [min(self.figure_start + i * 3, total - 1) for i in range(41)]
        videos, extrinsics, intrinsics, sizes = [], [], [], []
        self._last_view_dirs = [str(Path(episode_path) / v) for v in ('view1', 'view2')]
        for vd in self._last_view_dirs:
            with av.open(str(Path(vd) / 'rgb/video.mp4')) as container:
                selected = {i: f.to_image() for i, f in enumerate(container.decode(video=0))
                            if i in set(indices)}
            videos.append(torch.stack([self.frame_process(selected[i]) for i in indices], 1))
            image = selected[indices[0]]
            sizes.append((image.height, image.width))
            ex, intr = self.get_camera_params(str(Path(vd) / 'camera_params.json'), indices)
            extrinsics.append(ex)
            intrinsics.append(intr)
        return videos, indices, extrinsics, intrinsics, sizes, True

    def getitem(self, index, force_template=None):
        with patch('training.dataset.base.random.sample', return_value=[0, 1]):
            return super().getitem(index, force_template=force_template)


def uint8_clip(tensor):
    return np.clip(np.round((tensor.permute(1, 2, 3, 0).float().numpy()+1)*127.5), 0, 255).astype('uint8')


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def prepare_sample(dataset, template):
    sample = dataset.getitem(0, force_template=template)
    assert sample['template'] == template, (sample['template'], template)
    assert sample['view_indices'] == [0, 1]
    assert [Path(x).name for x in sample['view_dirs']] == ['view1', 'view2']
    for v in range(2):
        sl = slice(v*41, (v+1)*41)
        assert torch.allclose(sample['extrinsics'][sl], sample['extrinsics'][v*41].expand(41, -1, -1))
        assert torch.allclose(sample['intrinsics'][sl], sample['intrinsics'][v*41].expand(41, -1, -1))
    return sample


def generate(pipe, sample, out, args, shown):
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'generation.json'
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        assert old['checkpoint'] == str(Path(args.ckpt).resolve())
        assert old['seed'] == args.seed and old['steps'] == args.steps
        assert old['frame_indices'] == sample['frame_indices']
        print(f'SKIP complete: {out}', flush=True)
        return
    T = 41
    modality = sample['visual_modality']
    gt = uint8_clip(sample['streams'][modality]).reshape(2, T, 512, 512, 3)
    a5 = project_actions_7d_to_5d_torch_batch(
        sample['action_7d'][None].float().repeat(1, 2, 1),
        sample['extrinsics'][None].float(), sample['intrinsics'][None].float())
    act_gt = np.clip(np.round(project_action_5d_to_rgb_torch(a5, 512, 512)[0].numpy()*255), 0, 255).astype('uint8')
    act_gt = act_gt.reshape(2, T, 512, 512, 3)
    # The only scene-dependent model inputs are the current observation and pose.
    streams = {}
    for key, stream in sample['streams'].items():
        anchor_only = torch.cat([stream[:, v*T:v*T+1].repeat(1, T, 1, 1) for v in range(2)], 1)
        streams[key] = anchor_only[None].to(device=pipe.device, dtype=torch.bfloat16)
    current_action = sample['action_7d'][:1].repeat(T, 1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f'START {out.name} {sample["text"]} start={sample["frame_indices"][0]}', flush=True)
    started = time.time()
    frames = pipe(
        prompt=[sample['text']], negative_prompt='', template=sample['template'],
        streams=streams, conditioning_mode='iiii',
        camera=sample['camera'][None].to(device=pipe.device, dtype=torch.bfloat16),
        action_7d=current_action[None].to(device=pipe.device, dtype=torch.bfloat16),
        extrinsics=sample['extrinsics'][None].to(device=pipe.device, dtype=torch.bfloat16),
        intrinsics=sample['intrinsics'][None].to(device=pipe.device, dtype=torch.bfloat16),
        height=512, width=512, num_frames=T, cfg_scale=args.cfg,
        num_inference_steps=args.steps, seed=args.seed, tiled=False,
        tile_size=(32, 32), tile_stride=(16, 16), enable_usp=False, cfg_parallel=False)
    raw = np.stack([np.asarray(f) for f in frames])
    assert raw.shape == (4*T, 512, 512, 3), raw.shape
    obs_pred = np.stack([raw[:T], raw[2*T:3*T]])
    action_pred = np.stack([raw[T:2*T], raw[3*T:4*T]])
    np.savez_compressed(out/'frames.npz', perception=obs_pred, action=action_pred,
                        perception_gt=gt, action_gt=act_gt)
    for view in range(2):
        for i in range(T):
            save_frame(out/f'view{view+1}/perception/{i:03d}.png', obs_pred[view, i])
            save_frame(out/f'view{view+1}/action/{i:03d}.png', action_pred[view, i])
        for i in [0, *shown]:
            save_frame(out/f'view{view+1}/gt_perception/{i:03d}.png', gt[view, i])
            save_frame(out/f'view{view+1}/gt_action/{i:03d}.png', act_gt[view, i])
    metadata = dict(checkpoint=str(Path(args.ckpt).resolve()), checkpoint_size=Path(args.ckpt).stat().st_size,
                    template=sample['template'], instruction=sample['text'],
                    episode=sample['path'], views=['view1', 'view2'], frame_indices=sample['frame_indices'],
                    shown_indices=shown, shown_source_frames=[sample['frame_indices'][i] for i in shown],
                    conditioning='IIII', model_inputs='current observation and action anchors only; future pixels/poses repeated from anchors',
                    seed=args.seed, steps=args.steps, cfg=args.cfg, resolution=512, frame_interval=3,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    generation_seconds=time.time()-started, format='raw uint8 RGB VAE outputs; no recoloring or retouching')
    meta_path.write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'DONE {out.name}: {time.time()-started:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=7.5)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    episode = REPO/'data/rlbench_selfgen_512_aug_wide/open_drawer/variation0/episodes/episode2'
    tasks = [(12, m, f'figure1_{m}', [15, 29]) for m in ['video', 'depth', 'normal', 'segmentation']]
    tasks += [(56, 'depth', 'figure2_depth', [7, 14])]
    samples = []
    for start, modality, name, shown in tasks:
        ds = FigureDataset(episode, start, 'open bottom drawer')
        sample = prepare_sample(ds, f'{modality}+action')
        print(f'PREPARED {name}: {sample["text"]}; shown={[sample["frame_indices"][i] for i in shown]}', flush=True)
        samples.append((sample, name, shown))
    if args.prepare_only:
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'This run is restricted to physical GPU 0.'
    assert torch.cuda.device_count() == 1
    from inference import build_pipeline
    pipe = build_pipeline(SimpleNamespace(model_id='Wan-AI/Wan2.2-TI2V-5B', ckpt_path=args.ckpt,
                          height=512, width=512, use_usp=False, cfg_parallel=False,
                          dynamic_cache_schedule=False, torch_compile=False))
    for sample, name, shown in samples:
        generate(pipe, sample, Path(args.out)/name, args, shown)
    print('ALL_GENERATIONS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
