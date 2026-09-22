import numpy as np

from scripts.mechanism_causal_roles import intervene_rgb, matched_background_masks


def _fixture():
    h = w = 48
    yy, xx = np.mgrid[:h, :w]
    image = np.stack([xx * 5, yy * 5, (xx + yy) * 2], axis=-1).clip(0, 255).astype(np.uint8)
    anchors = np.stack([image, image[:, ::-1]])
    target = np.zeros((2, h, w), bool)
    target[:, 18:27, 16:25] = True
    background = np.ones_like(target)
    background[:, 12:33, 10:31] = False
    zeros = np.zeros_like(target)
    masks = {"target": target, "goal": zeros, "fixture": zeros, "robot": zeros,
             "task": target, "structure": target, "background": background}
    return anchors, masks


def test_matched_background_has_exact_area_and_is_background():
    _, masks = _fixture()
    control = matched_background_masks(masks["target"], masks["background"])
    assert np.array_equal(control.sum((1, 2)), masks["target"].sum((1, 2)))
    assert np.all(~control | masks["background"])


def test_clean_is_identity_and_delete_is_local():
    anchors, masks = _fixture()
    clean, clean_mask = intervene_rgb(anchors, masks, "clean")
    assert np.array_equal(clean, anchors)
    assert not clean_mask.any()

    edited, applied = intervene_rgb(anchors, masks, "delete_target", dilate_px=1)
    assert applied.sum() > masks["target"].sum()
    assert np.any(edited[applied] != anchors[applied])
    assert np.array_equal(edited[~applied], anchors[~applied])


def test_control_matches_dilated_delete_area():
    anchors, masks = _fixture()
    _, deleted = intervene_rgb(anchors, masks, "delete_target", dilate_px=2)
    _, control = intervene_rgb(anchors, masks, "control_target", dilate_px=2)
    assert np.array_equal(deleted.sum((1, 2)), control.sum((1, 2)))
    assert not np.any(control & masks["target"])
