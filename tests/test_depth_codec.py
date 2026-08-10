"""Acceptance test for depth_codec (plan/core/10-vision_banana_codec_refactor.md §6.1):
Barron power transform math, RGB-cube path bijection, quantized roundtrip (per depth region),
invalid-value handling.

Run: python /workspace/ttdu/ActionImages-Cogen/tests/test_depth_codec.py
"""
import sys
from pathlib import Path

import numpy as np

from training.percep.depth_codec import (
    barron_transform, barron_inverse, path_encode, path_decode, encode_depth, decode_depth,
    roundtrip_absrel, CUBE_PATH, INVALID_SENTINEL, INVALID_PROJ_DIST2_THRESHOLD,
    INVALID_PROJ_DIST_THRESHOLD, MIN_VALID, MAX_VALID, LAMBDA, C,
)


def must_raise(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")

# --- 1. math properties ---
d = np.linspace(0.05, 20.0, 5000).astype(np.float32)
y = barron_transform(d)
assert np.all(np.diff(y) > 0), "barron_transform must be strictly monotonic"
d_rec = barron_inverse(y)
max_err = np.max(np.abs(d_rec - d))
assert max_err < 1e-3, f"barron_transform/inverse roundtrip max err {max_err}"
print(f"BARRON_BIJECTION_OK max_err={max_err:.2e}")

y_grid = np.linspace(0, 1, 2000).astype(np.float32)
y_rec = path_decode(path_encode(y_grid))[0]
max_err = np.max(np.abs(y_rec - y_grid))
assert max_err < 1e-3, f"path_encode/decode roundtrip max err {max_err}"
print(f"CUBE_PATH_BIJECTION_OK max_err={max_err:.2e}")

edge_lens = np.sqrt(((CUBE_PATH[1:] - CUBE_PATH[:-1]) ** 2).sum(-1))
assert np.allclose(edge_lens, 1.0), f"expected unit edges, got {edge_lens}"
assert len(set(map(tuple, CUBE_PATH.tolist()))) == 8, "CUBE_PATH must visit 8 distinct vertices"
figure5_path = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1],
], dtype=np.float32)
assert np.array_equal(CUBE_PATH, figure5_path), "CUBE_PATH must follow the paper's Figure 5"
print("CUBE_PATH_HAMILTON_OK")

sentinel01 = INVALID_SENTINEL.astype(np.float32) / 255.0
_, d2 = path_decode(sentinel01)
assert d2 > INVALID_PROJ_DIST_THRESHOLD, (
    f"INVALID_SENTINEL must be far from CUBE_PATH (dist2={d2}), else decode can't flag it")
print(f"INVALID_SENTINEL_OFF_PATH_OK dist2={float(d2):.3f}")
assert INVALID_PROJ_DIST_THRESHOLD == INVALID_PROJ_DIST2_THRESHOLD

# --- 2. quantized roundtrip, per depth region ---
regions = [("near", 0.1, 1.0), ("mid", 1.0, 3.0), ("far", 3.0, 4.5), ("wide", 0.1, 9.9)]
for name, lo, hi in regions:
    dd = np.linspace(lo, hi, 5000).astype(np.float32).reshape(1, -1)
    dec = decode_depth(encode_depth(dd))
    absrel = np.abs(dec - dd) / dd
    print(f"REGION[{name}] range=({lo},{hi}) AbsRel mean={absrel.mean()*100:.3f}% max={absrel.max()*100:.3f}%")
    assert absrel.mean() < 0.02, f"{name}: mean AbsRel {absrel.mean()*100:.3f}% exceeds 2% threshold"
print("QUANTIZED_ROUNDTRIP_OK")

# --- 3. invalid values ---
invalid = np.array([0.0, -1.0, np.nan, np.inf, MAX_VALID + 5.0], dtype=np.float32).reshape(1, -1)
enc = encode_depth(invalid)
assert np.all(enc == INVALID_SENTINEL), f"invalid depth must encode to INVALID_SENTINEL, got {enc}"
dec = decode_depth(enc)
assert np.all(np.isnan(dec)), f"invalid-encoded pixels must decode to NaN, got {dec}"
print("INVALID_VALUE_HANDLING_OK")

# boundary: values just inside/outside MIN_VALID/MAX_VALID
just_valid = np.array([MIN_VALID + 1e-3, MAX_VALID - 1e-3], dtype=np.float32).reshape(1, -1)
enc = encode_depth(just_valid)
assert not np.any(np.all(enc == INVALID_SENTINEL, axis=-1)), "just-inside-range values must not be sentinel"
exact_boundary = np.array([MIN_VALID, MAX_VALID], dtype=np.float32).reshape(1, -1)
assert np.all(encode_depth(exact_boundary) == INVALID_SENTINEL), "exact boundaries are excluded"
print("BOUNDARY_VALID_OK")

# --- 4. strict API and engineering-domain behavior ---
for bad_lam, bad_c in [(-1.0, C), (0.0, C), (-3.0, 0.0), (-3.0, -1.0), (np.nan, C)]:
    must_raise(ValueError, barron_transform, np.array([1.0], np.float32), bad_lam, bad_c)
must_raise(ValueError, barron_transform, np.array([-1.0], np.float32))
must_raise(ValueError, barron_inverse, np.array([1.0], np.float32))

must_raise(TypeError, decode_depth, np.zeros((2, 2, 3), dtype=np.float32))
must_raise(ValueError, decode_depth, np.zeros((2, 2), dtype=np.uint8))
must_raise(ValueError, decode_depth, np.zeros((2, 2, 4), dtype=np.uint8))
must_raise(ValueError, path_decode, np.zeros((2, 2, 4), dtype=np.float32))
must_raise(ValueError, path_decode, np.full((2, 2, 3), np.nan, dtype=np.float32))
must_raise(ValueError, encode_depth, np.array(1.0, dtype=np.float32))
must_raise(TypeError, encode_depth, np.array([["1.0"]]))
must_raise(ValueError, roundtrip_absrel, np.zeros((2, 2), dtype=np.float32))

# Black is d=0 and white is d=infinity in the paper; both lie outside the strict RLBench range.
endpoints = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
assert np.all(np.isnan(decode_depth(endpoints))), "black/white path endpoints must decode invalid"

# A legal path color beyond MAX_VALID must also decode invalid, making encode/decode domains symmetric.
far_y = barron_transform(np.array([20.0], dtype=np.float32))
far_rgb = np.round(path_encode(far_y).reshape(1, 1, 3) * 255).astype(np.uint8)
assert np.isnan(decode_depth(far_rgb)[0, 0]), "decoded depth outside MAX_VALID must be invalid"

# Small on-path quantization remains valid; a cube-center color is too far from every used edge.
assert np.isfinite(decode_depth(encode_depth(np.full((2, 2), 2.0, np.float32)))).all()
cube_center = np.full((1, 1, 3), 128, dtype=np.uint8)
assert np.isnan(decode_depth(cube_center)[0, 0]), "far-off-path RGB must decode invalid"

# Small perpendicular RGB noise stays decodable, while small sentinel perturbations remain invalid.
noisy_valid = encode_depth(np.full((1, 1), 2.0, np.float32)).astype(np.int16)
noisy_valid[..., 2] = np.clip(noisy_valid[..., 2] + 5, 0, 255)
assert np.isfinite(decode_depth(noisy_valid.astype(np.uint8))[0, 0])
for delta in (-5, 5):
    noisy_sentinel = np.clip(INVALID_SENTINEL.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    assert np.isnan(decode_depth(noisy_sentinel.reshape(1, 1, 3))[0, 0])

# Single-frame, video, and batched-video shapes preserve all leading dimensions.
for shape in [(4,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    sample = np.full(shape, 2.0, dtype=np.float32)
    encoded = encode_depth(sample)
    decoded = decode_depth(encoded)
    assert encoded.shape == shape + (3,)
    assert decoded.shape == shape
print("STRICT_API_AND_DOMAIN_OK")

# --- 5. real selfgen episode (if present) ---
import glob
# repo_root/data/rlbench_selfgen is the symlink to the selfgen tree (see FORK_CHANGES.md)
data_root = Path(__file__).resolve().parents[1] / "data" / "rlbench_selfgen"
paths = sorted(glob.glob(str(data_root / "*" / "variation0" / "episodes" / "episode0" / "view1" / "depth.npz")))
if paths:
    errs = []
    for p in paths[:4]:
        depth = np.load(p)["depth"].astype(np.float32)
        valid = np.isfinite(depth) & (depth > MIN_VALID) & (depth < MAX_VALID)
        dec = decode_depth(encode_depth(depth))
        absrel = np.abs(dec[valid] - depth[valid]) / depth[valid]
        errs.append(float(absrel.mean()))
        print(f"{p.split('/')[-5]}: frames={depth.shape[0]} range=({depth.min():.2f},{depth.max():.2f})m "
              f"roundtrip AbsRel={errs[-1]*100:.3f}%")
    m = float(np.mean(errs))
    print(f"REAL_EPISODE_{'OK' if m < 0.02 else 'FAIL'} mean AbsRel={m*100:.3f}%")
else:
    print("REAL_EPISODE_SKIPPED (no selfgen_v2 depth.npz found)")

print("ALL_DEPTH_CODEC_TESTS_PASSED")
