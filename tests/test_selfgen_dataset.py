"""Behaviour of RLBenchSelfgenDataset under each task template.

The load-bearing assertion here is `action_7d.abs().sum() > 0` ON A PERCEPTION SAMPLE. ttd's
equivalent test asserted the exact opposite -- there, a perception sample had its actions
zeroed, which is what made action and perception supervision mutually exclusive and what made
the project's main RQ answerable only indirectly. That one line is the whole change.

Also guards the things that are easy to get subtly wrong and impossible to notice in a loss
curve:
  - BOTH views of a perception modality are encoded (ttd encoded only the target view);
  - `video+depth` really carries RGB **and** depth, which is what makes it perception rather
    than the old substitution layout;
  - `depth+action` still carries no RGB, which is what makes it a world model rather than
    perception -- the distinction the whole design turns on;
  - a template's prompt lists exactly the modalities its streams contain.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_selfgen_dataset.py
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import RLBenchSelfgenDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELFGEN = os.path.join(REPO, "data", "rlbench_selfgen")
RES, NUM_FRAMES = 256, 41

OFFICIAL_KEYS = {"text", "video", "camera", "extrinsics", "intrinsics",
                 "action_7d", "action_8d", "path"}
FORK_KEYS = {"frame_indices", "view_indices", "view_dirs", "visual_modality",
             "streams", "template"}


def build(mix, seed=0, **kw):
    random.seed(seed)
    kw.setdefault("action_dropout_prob", 0.0)  # tests assert on the drawn template itself
    return RLBenchSelfgenDataset(base_path=SELFGEN, num_frames=NUM_FRAMES, frame_interval=1,
                                 height=RES, width=RES, template_mix=mix,
                                 strict_getitem=True, **kw)


def check_shapes(s):
    assert OFFICIAL_KEYS | FORK_KEYS <= set(s), f"missing keys: {(OFFICIAL_KEYS | FORK_KEYS) - set(s)}"
    assert tuple(s["video"].shape) == (3, 2 * NUM_FRAMES, RES, RES), tuple(s["video"].shape)
    assert tuple(s["action_7d"].shape) == (NUM_FRAMES, 7), tuple(s["action_7d"].shape)
    assert tuple(s["camera"].shape) == (2 * NUM_FRAMES, 12), tuple(s["camera"].shape)
    for name, stream in s["streams"].items():
        assert tuple(stream.shape) == (3, 2 * NUM_FRAMES, RES, RES), (name, tuple(stream.shape))
        assert float(stream.min()) >= -1.01 and float(stream.max()) <= 1.01, name
    # The prompt must name exactly the visual modalities that are actually present, otherwise
    # the tag is decoration rather than a control signal.
    visual_in_template = [m for m in s["template"].split("+") if m != "action"]
    assert sorted(visual_in_template) == sorted(s["streams"]), (
        f"template {s['template']!r} disagrees with streams {sorted(s['streams'])}"
    )


def main():
    # ---- 1. video+action is the upstream path ----
    ds = build("video+action@1.0")
    s = ds[0]
    check_shapes(s)
    assert s["template"] == "video+action"
    assert set(s["streams"]) == {"video"}
    assert s["text"].startswith("<video><action> "), s["text"][:48]
    assert float(s["action_7d"].abs().sum()) > 0
    print(f"VIDEO_ACTION_OK text={s['text'][:56]!r}")

    # ---- 1b. prompt_tag_style=none reproduces upstream text verbatim ----
    ds_none = build("video+action@1.0", prompt_tag_style="none")
    random.seed(7)
    tagged = build("video+action@1.0").getitem(0)
    random.seed(7)
    untagged = ds_none.getitem(0)
    assert not untagged["text"].startswith("<"), untagged["text"][:40]
    assert tagged["text"].endswith(untagged["text"]), (tagged["text"][:60], untagged["text"][:60])
    print(f"PROMPT_TAG_STYLE_NONE_OK {untagged['text'][:48]!r}")

    # ---- 2. video+depth: RGB *and* depth in the same sample -> perception is expressible ----
    ds = build("video+depth@1.0")
    s = ds[0]
    check_shapes(s)
    assert s["template"] == "video+depth"
    assert set(s["streams"]) == {"video", "depth"}, sorted(s["streams"])
    assert s["text"].startswith("<video><depth> "), s["text"][:48]
    assert not torch.allclose(s["streams"]["video"], s["streams"]["depth"], atol=1e-3), (
        "the depth stream equals the RGB stream -- the perception target is not being encoded"
    )
    print("PERCEPTION_TEMPLATE_OK video+depth carries both streams")

    # ---- 3. co-supervision: perception AND action in one sample ----
    ds = build("video+depth+action@1.0")
    s = ds[0]
    check_shapes(s)
    assert s["template"] == "video+depth+action"
    assert s["text"].startswith("<video><depth><action> "), s["text"][:48]
    # THE assertion. ttd zeroed these; keeping them is what makes co-supervision possible.
    assert float(s["action_7d"].abs().sum()) > 0, (
        "co-generation sample has zeroed actions -- the mutually-exclusive design is back "
        "(see FORK_CHANGES.md / doc 15 §2.5)"
    )
    assert float(s["action_8d"].abs().sum()) > 0
    print(f"COSUPERVISION_OK action_7d.abs().sum()={float(s['action_7d'].abs().sum()):.2f}")

    # ---- 4. depth+action is a WORLD MODEL: no RGB anywhere in the sample ----
    ds = build("depth+action@1.0")
    s = ds[0]
    check_shapes(s)
    assert set(s["streams"]) == {"depth"}, sorted(s["streams"])
    assert "video" not in s["template"].split("+"), s["template"]
    assert float(s["action_7d"].abs().sum()) > 0
    print("SUBSTITUTION_TEMPLATE_OK depth+action carries no RGB (world model, not perception)")

    # ---- 5. BOTH views of a perception modality, not just the target one ----
    # super().getitem() draws a random window and view pair, so the two calls below MUST be
    # given the same RNG state -- otherwise they describe different frames and "the halves
    # differ" would be true for a trivial reason, and the test would pass while asserting
    # nothing. Reseed immediately before each call and verify the draws actually coincided.
    ds = build("video+depth@1.0")
    random.seed(11)
    a = ds.getitem(0, force_template="video+action")
    random.seed(11)
    b = ds.getitem(0, force_template="video+depth")
    assert a["view_indices"] == b["view_indices"], "reseeding failed: different view pair"
    assert a["frame_indices"] == b["frame_indices"], "reseeding failed: different window"
    T = NUM_FRAMES
    for name, sl in (("cond", slice(0, T)), ("target", slice(T, 2 * T))):
        assert not torch.allclose(b["streams"]["depth"][:, sl], a["streams"]["video"][:, sl], atol=1e-3), (
            f"the {name} half of the depth stream equals the RGB for the SAME window/view -- "
            f"only one view was encoded (ttd's asymmetric behaviour), so a depth segment would "
            f"silently carry RGB pixels"
        )
    # ... and the RGB stream of the perception sample is byte-identical to the plain RGB path,
    # i.e. adding a depth stream did not perturb the observation.
    assert torch.equal(a["streams"]["video"], b["streams"]["video"])
    print(f"BOTH_VIEWS_ENCODED_OK (same draw: views={a['view_indices']} "
          f"frames={a['frame_indices'][0]}..{a['frame_indices'][-1]})")

    # ---- 6. mixed mix: every template appears and every sample is well-formed ----
    ds = build("video+action@0.34,video+depth@0.33,video+segmentation@0.33", seed=3)
    seen = {}
    for i in range(60):
        s = ds[i]
        check_shapes(s)
        assert float(s["action_7d"].abs().sum()) > 0  # true for EVERY template now
        seen[s["template"]] = seen.get(s["template"], 0) + 1
    assert {"video+action", "video+depth"} <= set(seen), f"mix produced only {set(seen)}"
    print(f"MIXED_MIX_OK {seen}")

    # ---- 7. action dropout reproduces upstream's 10% video-only gate ----
    ds = build("video+action@1.0", seed=5, action_dropout_prob=1.0)
    s = ds[0]
    assert s["template"] == "video", s["template"]
    assert s["text"].startswith("<video> "), s["text"][:32]
    print("ACTION_DROPOUT_OK template and prompt drop <action> together")

    # ---- 8. bad mix strings are rejected at construction, not at step 3000 ----
    for bad in ("", "rgb+action@1.0", "video+action@0", "video+action@-1", "action@1.0",
                "video+video@1.0"):
        try:
            build(bad)
        except (ValueError, AssertionError):
            continue
        raise AssertionError(f"template_mix={bad!r} should have been rejected")
    print("MIX_VALIDATION_OK")
    print("ALL_SELFGEN_DATASET_TESTS_PASSED")


if __name__ == "__main__":
    main()
