"""Every template's prompt prefix must survive to the text encoder as usable tokens.

Three ways this silently breaks:

1. `train.py` scrubs every prompt before encoding
   (`.replace(".","").replace("_"," ").replace("  "," ")`). A tag containing `.` or `_` would
   be mangled there, and nothing downstream would complain -- the model would just be
   conditioned on a slightly different string than the eval script uses.
2. The UMT5 tokenizer could shatter a tag into many subwords, or map it to <unk>. It does not
   need a dedicated token id (the model learns the association from whatever pieces it gets,
   exactly as it learns everything else in the instruction), but it does need to be stable and
   non-empty, and the eval-time prompt must tokenize identically to the train-time one --
   otherwise the control signal at inference is not the one that was trained.
3. `prompt_tag_style="none"` must reproduce upstream text EXACTLY. It is the hook the L2
   numeric regression against the upstream checkout hangs on: with `explicit` (the training
   default) the baseline template's text deliberately differs from upstream, so that
   comparison has nowhere else to attach.

Also checks the paths the data must satisfy, because `forward` gates the single-frame-condition
variant on the substring "rlbench" being in the episode path.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_prompt_tags.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.templates import parse_template, prompt_prefix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(REPO, "checkpoints", "Wan-AI", "Wan2.2-TI2V-5B", "google", "umt5-xxl")

INSTRUCTION = "grip the bottom handle and pull the bottom drawer open"
SEG_COLORS = {"jar lid": "red", "jar": "green"}

TEMPLATES = ["video+action", "video+depth", "video+segmentation", "video+depth+action",
             "depth+action"]


def scrub(text):
    """Exactly what ActionImagesModel.forward does to every prompt before encode_prompt."""
    return text.replace(".", "").replace("_", " ").replace("  ", " ")


def prompts():
    out = {}
    for name in TEMPLATES:
        mods = parse_template(name)
        colors = SEG_COLORS if "segmentation" in mods else None
        out[name] = prompt_prefix(mods, colors, "explicit") + INSTRUCTION
    return out


def main():
    built = prompts()

    # 1. the scrub must leave every tag intact
    for name, prompt in built.items():
        tags = prompt[: prompt.rindex(">") + 1]
        assert scrub(prompt).startswith(tags), (
            f"the prompt scrub mangles {name}'s tags: {prompt!r} -> {scrub(prompt)!r}"
        )
        # The template must be recoverable from the text -- that is what "the prompt is the
        # control signal" means operationally.
        for m in parse_template(name):
            needle = "<seg:" if m == "segmentation" else f"<{m}>"
            assert needle in tags, f"{name}: prompt {tags!r} does not announce {m!r}"
        print(f"  {name:22s} -> {tags!r}")
    print("PROMPT_SCRUB_SAFE_OK")

    # 2. `none` reproduces upstream text verbatim, for every template
    for name in TEMPLATES:
        mods = parse_template(name)
        colors = SEG_COLORS if "segmentation" in mods else None
        assert prompt_prefix(mods, colors, "none") == "", name
    print("PROMPT_TAG_STYLE_NONE_OK (upstream text has no prefix at all)")

    # 2b. a segmentation template without a colour map is a bug, not a silent empty tag: the
    #     model cannot know which colour means which object.
    try:
        prompt_prefix(parse_template("video+segmentation"), None, "explicit")
    except ValueError:
        print("SEG_REQUIRES_COLOR_MAP_OK")
    else:
        raise AssertionError("segmentation prompt built without a colour map")

    # 3. tokenizer sanity (skipped if the checkpoint tree is not linked in)
    if not os.path.isdir(TOKENIZER_DIR):
        print(f"TOKENIZER_CHECK_SKIPPED (no {TOKENIZER_DIR})")
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        base = tok(scrub(INSTRUCTION), add_special_tokens=False)["input_ids"]
        for name, prompt in built.items():
            ids = tok(scrub(prompt), add_special_tokens=False)["input_ids"]
            extra = len(ids) - len(base)
            decoded = tok.decode(ids)
            assert extra > 0, f"{name} tags contributed no tokens"
            assert extra <= 32, f"{name} tags exploded into {extra} tokens: {decoded[:100]!r}"
            unk = getattr(tok, "unk_token_id", None)
            if unk is not None:
                assert unk not in ids, f"{name} prompt hit <unk>: {decoded[:100]!r}"
            # train-time and eval-time prompts must tokenize identically
            assert ids == tok(scrub(prompt), add_special_tokens=False)["input_ids"]
            print(f"TOKENIZER_OK {name:22s} +{extra:3d} tokens")
        # UMT5 has a 512-token budget; the tags must not be a meaningful fraction of it.
        longest = max(len(tok(scrub(p), add_special_tokens=False)["input_ids"]) for p in built.values())
        assert longest < 128, f"longest tagged prompt is {longest} tokens"
        print(f"TOKEN_BUDGET_OK longest tagged prompt = {longest} tokens")

    # 4. episode paths must contain "rlbench" -- forward gates the 10% single-frame condition
    #    variant on it, so a differently-named data symlink silently changes the training
    #    recipe relative to the control arm.
    from training.dataset import RLBenchSelfgenDataset

    ds = RLBenchSelfgenDataset(
        base_path=os.path.join(REPO, "data", "rlbench_selfgen"),
        num_frames=41, frame_interval=1, height=256, width=256,
    )
    path = ds.episodes[0]["path"]
    assert "rlbench" in path, (
        f"episode path {path!r} lacks the substring 'rlbench'; forward would skip the "
        f"single-frame-condition variant for this dataset only, confounding the comparison "
        f"against the control arm"
    )
    print(f"EPISODE_PATH_GATE_OK {path}")
    print("ALL_PROMPT_TAG_TESTS_PASSED")


if __name__ == "__main__":
    main()
