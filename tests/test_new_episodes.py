"""Validate NEWLY GENERATED episodes specifically, not whatever ds[0..N] happens to be.

`run_tests.sh` samples the dataset from index 0, which follows glob order -- it can pass
completely without ever touching an episode produced by the latest generation run. That makes
it useless as an acceptance gate for new data: the whole question is whether the new episodes
are byte-compatible with the old ones (camera_params.json key format, depth dtype/resolution,
handles.json naming, seg_targets.json coverage), and a test that never reads them cannot answer it.

This test selects episodes by seed -- `episodeN` where N >= --since-seed -- and runs the three
checks that would catch an incompatibility, on those episodes only:

  1. alignment  : recompute camera/extrinsics/intrinsics from camera_params.json on disk for the
                  window and views the sample recorded, and compare. Catches a changed key
                  format (selfgen uses "0" where the official layout uses "0000") and any
                  view-ordering drift.
  2. depth e2e  : decode the depth stream back to metres and compare against depth.npz. Catches
                  dtype/resolution/scale changes in the generator's depth export.
  3. seg        : seg_targets.json present, and the decoded seg matches handles.json by IoU.
                  Catches the silent-degrade-to-video failure, which is invisible in a loss curve.

Run:
    python tests/test_new_episodes.py                    # episodes with seed >= 20
    python tests/test_new_episodes.py --since-seed 0     # every variation0 episode
    python tests/test_new_episodes.py --max-per-task 3
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import RLBenchSelfgenDataset
from training.percep.depth_codec import MAX_VALID, MIN_VALID, decode_depth
from training.percep.seg_codec import build_referring_spec, decode_known_color, handles_to_mask
from training.utils import convert_intrinsics_after_center_crop_resize, get_relative_pose_batch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The live tree. This used to be "rlbench_selfgen", whose symlink DANGLES (the 256 v2 tree was
# deleted to free disk) -- so this file failed with `Found 0 episodes` on every run, and had been
# doing so silently in the suite for as long as the tree has been gone. SELFGEN_TEST_DATA
# overrides it. RES follows the tree: 512_aug renders at 512, and the loader would silently
# downsample a 256 request rather than complain.
SELFGEN = os.environ.get("SELFGEN_TEST_DATA", os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))
RES, NUM_FRAMES = 512, 41
ABSREL_MAX, IOU_MIN = 0.02, 0.95


def to_uint8(v):
    return np.clip(np.round((v.permute(1, 2, 3, 0).numpy() + 1) * 127.5), 0, 255).astype(np.uint8)


def check_alignment(s):
    frames = s["frame_indices"]
    out = []
    for which, vi in (("cond", 0), ("target", 1)):
        view = s["view_dirs"][s["view_indices"][vi]]
        with open(os.path.join(view, "camera_params.json")) as f:
            data = json.load(f)
        extr, intr = [], []
        for idx in frames:
            key = str(idx) if str(idx) in data else max(data.keys(), key=lambda k: int(k))
            extr.append(np.array(data[key]["extrinsics"]))
            intr.append(np.array(data[key]["intrinsics"]))
        out.append((view, np.stack(extr), np.stack(intr)))

    (vc, ec, ic), (vt, et, it_) = out
    want_extr = torch.from_numpy(np.concatenate([ec, et], axis=0))
    assert torch.allclose(s["extrinsics"].double(), want_extr.double(), atol=1e-6), \
        f"extrinsics mismatch (camera_params.json key format changed?)"
    base = ec[0]
    rel_c = rearrange(torch.from_numpy(get_relative_pose_batch(base, ec)), "t c d -> t (c d)")
    rel_t = rearrange(torch.from_numpy(get_relative_pose_batch(base, et)), "t c d -> t (c d)")
    assert torch.allclose(s["camera"], torch.cat([rel_c, rel_t], 0).to(torch.float32), atol=1e-5), \
        "relative camera pose mismatch"
    raw_c = np.load(os.path.join(vc, "depth.npz"))["depth"].shape[-2:]
    raw_t = np.load(os.path.join(vt, "depth.npz"))["depth"].shape[-2:]
    conv = convert_intrinsics_after_center_crop_resize(
        [ic, it_], [tuple(raw_c), tuple(raw_t)], [(RES, RES), (RES, RES)])
    assert np.allclose(s["intrinsics"].numpy(), np.concatenate(conv, 0), atol=1e-4), \
        "intrinsics mismatch (native render resolution changed?)"


def check_depth(s):
    T, u8 = NUM_FRAMES, to_uint8(s["streams"]["depth"])
    worst = 0.0
    for which, sl, vi in (("cond", slice(0, T), 0), ("target", slice(T, 2 * T), 1)):
        view = s["view_dirs"][s["view_indices"][vi]]
        gt = np.load(os.path.join(view, "depth.npz"))["depth"][s["frame_indices"]].astype(np.float32)
        dec = decode_depth(u8[sl])
        assert dec.shape == gt.shape, f"{which}: shape {dec.shape} vs {gt.shape}"
        m = np.isfinite(dec) & (gt > MIN_VALID) & (gt < MAX_VALID)
        assert m.mean() > 0.5, f"{which}: only {m.mean():.1%} decodable"
        a = float(np.mean(np.abs(dec[m] - gt[m]) / gt[m]))
        assert a < ABSREL_MAX, f"{which}: AbsRel {a:.3%} >= {ABSREL_MAX:.0%}"
        worst = max(worst, a)
    return worst


def check_seg(s):
    p = os.path.join(s["path"], "seg_targets.json")
    assert os.path.exists(p), (
        "seg_targets.json missing -- this episode would SILENTLY degrade to video in the seg "
        "arm (no crash, no log). Run src/percep/seg_targets_gen.py over the tree.")
    seg_t = json.load(open(p))
    instruction = s["text"].split("> ", 1)[-1] if s["text"].startswith("<") else s["text"]
    id_groups, color_map, _ = build_referring_spec(seg_t, instruction)
    T, u8 = NUM_FRAMES, to_uint8(s["streams"]["segmentation"])
    worst, n_empty, n_scored = 1.0, 0, 0
    for which, sl, vi in (("cond", slice(0, T), 0), ("target", slice(T, 2 * T), 1)):
        view = s["view_dirs"][s["view_indices"][vi]]
        mask_map = np.load(os.path.join(view, "mask.npz"))["mask"][s["frame_indices"]]
        dec = decode_known_color(u8[sl], color_map)
        for inst, ids in id_groups.items():
            gt, pred = handles_to_mask(mask_map, ids), dec[inst]
            if gt.sum() == 0:
                # The instance is genuinely not visible from this viewpoint in this window --
                # common and expected (an open_drawer target hidden from one camera; the
                # reach_and_drag target is occluded in ~94% of episodes). "Nothing to segment,
                # nothing predicted" is a CORRECT encode/decode, and IoU is undefined for it:
                # scoring it as intersection/union = 0/0 would fail a healthy episode. What must
                # still be checked is the other direction -- predicting pixels for an object
                # that is not there is a real defect.
                assert pred.sum() == 0, (
                    f"{which}/{inst}: GT is empty (not visible) but decode produced "
                    f"{pred.sum()} pixels -- false positive in the seg encoding")
                n_empty += 1
                continue
            iou = (gt & pred).sum() / (gt | pred).sum()
            assert iou >= IOU_MIN, f"{which}/{inst}: IoU {iou:.3f} < {IOU_MIN}"
            worst = min(worst, iou)
            n_scored += 1
    assert n_scored > 0, (
        f"{s['path']}: every instance was invisible in both views -- nothing was actually "
        f"verified for this episode")
    return worst, n_empty, n_scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-seed", type=int, default=20,
                    help="only episodes episodeN with N >= this (default 20 = the extension run)")
    ap.add_argument("--max-per-task", type=int, default=2)
    args = ap.parse_args()

    picked = {}
    for modality in ("depth", "segmentation"):
        ds = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                                   height=RES, width=RES, template_mix=f"video+{modality}@1.0",
                                   strict_getitem=True, variations="0")
        if not picked:
            per_task = {}
            for i, ep in enumerate(ds.episodes):
                n = int(re.sub(r"\D", "", os.path.basename(ep["path"])) or -1)
                if n < args.since_seed:
                    continue
                per_task.setdefault(ep["task"], [])
                if len(per_task[ep["task"]]) < args.max_per_task:
                    per_task[ep["task"]].append(i)
            picked = per_task
            total = sum(len(v) for v in picked.values())
            if total == 0:
                print(f"NO_NEW_EPISODES (seed >= {args.since_seed}) -- nothing to validate")
                return
            print(f"validating {total} episodes (seed >= {args.since_seed}) "
                  f"across {len(picked)} tasks\n")

        stats = []
        for task, idxs in sorted(picked.items()):
            for i in idxs:
                random.seed(1234 + i)
                s = ds[i]
                if modality not in s["streams"]:
                    # seg degrades to video when nothing resolves; that is a real failure here
                    # because check_seg would otherwise never run on this episode.
                    raise AssertionError(
                        f"{s['path']}: asked for {modality}, got streams {sorted(s['streams'])} "
                        f"(seg_targets.json missing or resolves to nothing)")
                check_alignment(s)
                stats.append(check_depth(s) if modality == "depth" else check_seg(s))
        if modality == "depth":
            print(f"DEPTH_OK   alignment + roundtrip on {len(stats)} eps, worst AbsRel="
                  f"{max(stats):.4%}")
        else:
            worst = min(x[0] for x in stats)
            empty = sum(x[1] for x in stats)
            scored = sum(x[2] for x in stats)
            print(f"SEG_OK     alignment + roundtrip on {len(stats)} eps, worst IoU={worst:.4f} "
                  f"({scored} instance-views scored, {empty} not visible -> skipped)")

    # coverage: every variation0 episode must carry seg_targets.json
    ds = RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                               height=RES, width=RES, variations="0")
    missing = [e["path"] for e in ds.episodes
               if not os.path.exists(os.path.join(e["path"], "seg_targets.json"))]
    assert not missing, (
        f"{len(missing)}/{len(ds.episodes)} variation0 episodes lack seg_targets.json "
        f"(first: {missing[0]}). Those degrade to video in the seg arm, silently.")
    print(f"SEG_COVERAGE_OK  {len(ds.episodes)}/{len(ds.episodes)} variation0 episodes")
    print("ALL_NEW_EPISODE_TESTS_PASSED")


if __name__ == "__main__":
    main()
