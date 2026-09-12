"""Scene-role dataset integration on REAL episodes (SEGMENTATION_SCENE_ROLES_PLAN.md §12.1/§12.3).

Run: python tests/test_scene_seg_dataset.py   (prints SCENE_SEG_DATASET_OK on success)

CPU-only: builds the dataset, draws forced `video+segmentation` samples in both modes and checks
the encoded stream, not the model. The load-bearing test is
test_referring_mode_is_bit_identical -- the whole back-compat story of the plan (§8.1) is the
claim that adding scene_roles changes nothing for existing runs, and that claim is worth exactly
as much as the test that pins it.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.dataset.rlbench_selfgen import RLBenchSelfgenDataset  # noqa: E402
from training.percep.seg_codec import (  # noqa: E402
    SCENE_ROLE_PALETTE,
    ROLE_TO_LABEL,
    UNKNOWN_LABEL,
    decode_scene_roles,
    scene_role_labels,
)

# The live tree. It used to be "rlbench_selfgen", whose symlink now DANGLES (the v2 tree was
# deleted to free disk), so main() took the SKIP branch and every check below silently stopped
# running -- which is how the NO_OBJECT_HANDLE bug (5.2% of all pixels decoding as `unknown`)
# survived a green test suite. DATA_ENV lets a caller point at another tree; a missing tree is a
# hard failure now, never a skip.
DATA = os.environ.get("SCENE_SEG_TEST_DATA", os.path.join(REPO, "data", "rlbench_selfgen_512_aug"))
FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def make_ds(mode):
    return RLBenchSelfgenDataset(
        base_path=DATA, num_frames=5, frame_interval=3, height=64, width=64,
        template_mix="video+segmentation@1.0", variations="0",
        segmentation_mode=mode, strict_getitem=True,
    )


def _sample(ds, idx, seed=0):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return ds.getitem(idx, force_template="video+segmentation")


def test_referring_mode_is_bit_identical():
    """§12.3: the default path must be untouched. Compare against a run of the SAME code with
    the new parameter left at its default -- i.e. pin that `scene_roles` existing changes
    nothing, which is the property the plan's back-compat promise actually rests on."""
    ds_default = RLBenchSelfgenDataset(
        base_path=DATA, num_frames=5, frame_interval=3, height=64, width=64,
        template_mix="video+segmentation@1.0", variations="0", strict_getitem=True,
    )
    ds_explicit = make_ds("referring")
    check(ds_default.segmentation_mode == "referring", "default segmentation_mode is 'referring'")
    same_text = same_pix = 0
    for i in range(6):
        a, b = _sample(ds_default, i), _sample(ds_explicit, i)
        if a["text"] == b["text"]:
            same_text += 1
        if torch.equal(a["streams"]["segmentation"], b["streams"]["segmentation"]):
            same_pix += 1
        check(a["text"].startswith("<video><seg:"), f"referring prompt keeps its tag: {a['text'][:40]}")
    check(same_text == 6, f"referring text identical on 6/6 samples (got {same_text})")
    check(same_pix == 6, f"referring pixels bit-identical on 6/6 samples (got {same_pix})")


def test_scene_roles_stream_shape_and_range():
    ds = make_ds("scene_roles")
    s = _sample(ds, 0)
    seg = s["streams"]["segmentation"]
    check(seg.shape == (3, 10, 64, 64), f"[C,2T,H,W] with both views: {tuple(seg.shape)}")
    check(float(seg.min()) >= -1.0 and float(seg.max()) <= 1.0, "range within [-1,1]")
    check(seg.dtype == torch.float32, f"float32 (got {seg.dtype})")
    check(s["streams"]["video"].shape == seg.shape, "seg stream matches RGB stream shape")


def test_scene_roles_prompt_tag():
    ds = make_ds("scene_roles")
    s = _sample(ds, 0)
    check("<scene-seg>" in s["text"], f"scene tag present: {s['text'][:60]}")
    check("<seg:" not in s["text"], "referring tag absent -- protocols must be distinguishable")


def test_every_pixel_is_a_palette_colour():
    """No interpolated colours: the GT is resized as a handle map, never as RGB (§8.2)."""
    ds = make_ds("scene_roles")
    legal = {tuple(int(v) for v in c) for c in SCENE_ROLE_PALETTE}
    bad = 0
    for i in range(4):
        seg = _sample(ds, i)["streams"]["segmentation"]
        u8 = ((seg.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
        got = {tuple(int(v) for v in c) for c in u8.reshape(-1, 3)}
        bad += len(got - legal)
    check(bad == 0, f"every encoded pixel is a palette colour across 4 samples ({bad} illegal)")


def test_no_unknown_pixels_in_training_data():
    """§12.1: unknown must be 0 before training. Orange in the stream = an unmapped handle."""
    ds = make_ds("scene_roles")
    # Spread the samples across the index rather than taking the first 6: episodes are grouped by
    # task on disk, so 0..5 is one or two tasks and the sky sentinel's incidence varies per task
    # and per camera (measured 0.006%-52.9% of a view). 16 evenly spaced draws cover every task.
    n = len(ds)
    idxs = sorted({(i * n) // 16 for i in range(16)})
    total_unknown = 0
    worst = (0.0, None)
    for i in idxs:
        seg = _sample(ds, i)["streams"]["segmentation"]
        u8 = ((seg.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
        dec = decode_scene_roles(u8)
        u = int(dec["unknown"].sum())
        total_unknown += u
        frac = u / dec["unknown"].size
        if frac > worst[0]:
            worst = (frac, ds.episodes[i] if hasattr(ds, "episodes") else i)
    check(
        total_unknown == 0,
        f"zero `unknown` pixels over {len(idxs)} samples "
        f"(got {total_unknown}; worst {worst[0] * 100:.3f}% at {worst[1]})",
    )


def test_target_overlays_base_role():
    """§6.3: a referred handle becomes `target` IF its base role is a manipulable object.

    Not "every referred handle becomes target" -- seg_targets.json also names receptacles and
    implements (see test_every_task_resolves_a_target_and_keeps_goal_tool_distinct), and those
    must keep their more specific role.
    """
    ds = make_ds("scene_roles")
    found_target = 0
    for i in range(8):
        s = _sample(ds, i)
        lut, present = ds._scene_role_lut(s["path"], s["text"])
        if lut is None:
            continue
        scene = ds._load_scene_segments(s["path"])
        base = {h: inst["base_role"] for inst in scene["instances"].values() for h in inst["handles"]}
        ids = [h for hs in ds._referred_id_groups(s["path"], s["text"])[0].values() for h in hs]
        promoted = [h for h in ids if base.get(h) == "distractor"]
        kept = [h for h in ids if base.get(h) not in ("distractor", None)]
        if promoted:
            found_target += 1
            check(
                all(lut[h] == ROLE_TO_LABEL["target"] for h in promoted),
                f"referred manipulable handles -> `target` (sample {i})",
            )
        check(
            all(lut[h] == ROLE_TO_LABEL[base[h]] for h in kept),
            f"referred goal/tool/fixture handles KEEP their role (sample {i}, {len(kept)} handles)",
        )
    check(found_target > 0, f"at least one sample resolved a target ({found_target}/8)")


def test_every_task_resolves_a_target_and_keeps_goal_tool_distinct():
    """The overlay rule (§6.3), checked on metadata only so it can cover all 16 tasks fast.

    seg_targets.json names EVERY object the instruction refers to, which is not the same as the
    object being manipulated: 9 of 16 tasks name two or three. Promoting all of them to `target`
    repaints the receptacle and the implement red -- sweep_to_dustpan would lose broom(tool) and
    dustpan(goal) in one go. This pins both halves: every task still yields a target, and the
    tasks that have a distinct goal/tool still have one after the overlay.
    """
    import glob
    from training.percep.seg_codec import build_referring_spec

    root = DATA
    PROMOTABLE = {"distractor"}
    no_target, lost_role = [], []
    n = 0
    for task in sorted(os.listdir(root)):
        eps = sorted(glob.glob(os.path.join(root, task, "variation0", "episodes", "episode0")))
        if not eps:
            continue
        ep = eps[0]
        try:
            scene = json.load(open(os.path.join(ep, "scene_segments.json")))
            seg = json.load(open(os.path.join(ep, "seg_targets.json")))
            meta = json.load(open(os.path.join(ep, "meta.json")))
        except FileNotFoundError:
            continue
        instr = (meta.get("descriptions") or [meta.get("description", "")])[0]
        base = {h: i["base_role"] for i in scene["instances"].values() for h in i["handles"]}
        try:
            groups, _, _ = build_referring_spec(seg, instr)
        except ValueError:
            continue
        n += 1
        after = dict(base)
        for hs in groups.values():
            for h in hs:
                if after.get(h) in PROMOTABLE:
                    after[h] = "target"
        roles = set(after.values())
        if "target" not in roles:
            no_target.append(task)
        # a goal/tool defined in the metadata must survive the overlay
        for r in ("goal", "tool"):
            if r in base.values() and r not in roles:
                lost_role.append(f"{task}:{r}")

    check(n >= 16, f"checked all {n} tasks")
    check(not no_target, f"every task resolves a `target` (missing: {no_target})")
    check(not lost_role, f"no task loses its goal/tool to the target overlay (lost: {lost_role})")


def test_roles_are_stable_across_views_and_time():
    """§12.1: a role's colour must not change between the two views or across the window."""
    ds = make_ds("scene_roles")
    s = _sample(ds, 0)
    seg = s["streams"]["segmentation"]
    u8 = ((seg.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
    T = u8.shape[0] // 2
    v0, v1 = u8[:T], u8[T:]
    roles0 = {r for r, m in decode_scene_roles(v0).items() if m.any()}
    roles1 = {r for r, m in decode_scene_roles(v1).items() if m.any()}
    check(roles0 and roles1, f"both views carry roles ({sorted(roles0)} / {sorted(roles1)})")
    # colours are global, so any role seen in either view uses the same RGB by construction;
    # what needs checking is that neither view collapsed to background only.
    check(roles0 != {"background"}, "view 0 is not background-only")
    check(roles1 != {"background"}, "view 1 is not background-only")


def test_scene_roles_is_denser_than_referring():
    """The whole point of the protocol (§4): more non-background supervision, same pixels."""
    ds_r, ds_s = make_ds("referring"), make_ds("scene_roles")
    fr, fs = [], []
    for i in range(6):
        for ds, acc in ((ds_r, fr), (ds_s, fs)):
            seg = _sample(ds, i)["streams"]["segmentation"]
            u8 = ((seg.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
            acc.append(float((u8.sum(-1) > 30).mean()))
    mr, ms = float(np.median(fr)), float(np.median(fs))
    print(f"     median non-black: referring {mr * 100:.2f}%  scene_roles {ms * 100:.2f}%")
    check(ms > mr, f"scene_roles is denser ({ms * 100:.2f}% vs {mr * 100:.2f}%)")
    check(ms > 5 * mr, f"and by a large factor ({ms / max(mr, 1e-9):.1f}x)")


def main():
    # Deliberately not a skip: a green run of this file is supposed to MEAN the scene-role stream
    # was checked against real episodes. See the DATA comment above.
    if not os.path.isdir(DATA):
        sys.exit(f"FAIL: no data tree at {DATA} (set SCENE_SEG_TEST_DATA= to override)")
    print(f"data tree: {DATA}")
    for fn in [
        test_referring_mode_is_bit_identical,
        test_scene_roles_stream_shape_and_range,
        test_scene_roles_prompt_tag,
        test_every_pixel_is_a_palette_colour,
        test_no_unknown_pixels_in_training_data,
        test_target_overlays_base_role,
        test_every_task_resolves_a_target_and_keeps_goal_tool_distinct,
        test_roles_are_stable_across_views_and_time,
        test_scene_roles_is_denser_than_referring,
    ]:
        print(f"\n{fn.__name__}:")
        fn()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\nSCENE_SEG_DATASET_OK")


if __name__ == "__main__":
    main()
