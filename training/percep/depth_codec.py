"""Depth <-> RGB codec: Barron power transform + RGB-cube false-color path (Vision Banana,
arXiv:2604.20329, metric depth section + Eq.1). See plan/core/10-vision_banana_codec_refactor.md
for the derivation. The RGB-cube vertex order below follows the channel axes and color progression
shown in the paper's Figure 5.

Encoding: metric depth -> Barron transform y in [0,1) -> piecewise-linear RGB-cube path
(8-vertex Hamilton path from black to white, 7 unit edges) -> uint8 RGB.
Decode: nearest-line-segment projection onto the path -> invert Barron transform.

Invalid depth (<=0, NaN, Inf, out of sanity range) encodes to a fixed sentinel color that is
provably off the cube path, so decode can flag it instead of silently returning a bogus depth.

Pure functions + acceptance test. All uint8 RGB in [0,255].
"""
import numpy as np

LAMBDA, C = -3.0, 10.0 / 3.0

# Sanity range: RLBench external-camera working range (measured 0.88-4.16m). Barron transform's
# domain is [0,inf) but decode is numerically unstable near y->1 (see plan §2.4), so depth outside
# this range is treated as invalid rather than silently encoded.
MIN_VALID, MAX_VALID = 0.05, 10.0

# Figure-5 cube Hamilton path: black -> red -> yellow -> green -> cyan -> blue -> magenta ->
# white. It has 8 vertices and 7 unit edges; each step flips one coordinate (Gray-code order).
CUBE_PATH = np.array([
    [0, 0, 0],  # black
    [1, 0, 0],  # red
    [1, 1, 0],  # yellow
    [0, 1, 0],  # green
    [0, 1, 1],  # cyan
    [0, 0, 1],  # blue
    [1, 0, 1],  # magenta
    [1, 1, 1],  # white
], dtype=np.float32)
N_SEGMENTS = len(CUBE_PATH) - 1
PATH_STARTS = CUBE_PATH[:-1]
PATH_VECTORS = CUBE_PATH[1:] - CUBE_PATH[:-1]
PATH_LENGTHS2 = np.sum(PATH_VECTORS * PATH_VECTORS, axis=-1)

# Invalid-depth sentinel color: not on any CUBE_PATH edge (verified by INVALID_SENTINEL_MIN_DIST2
# test in tests/test_depth_codec.py), so decode can distinguish it from any valid encoding.
INVALID_SENTINEL = np.array([128, 0, 128], dtype=np.uint8)
INVALID_PROJ_DIST2_THRESHOLD = 0.15  # squared distance in normalized [0,1]^3 RGB space
# Backward-compatible alias; new code should use the name that explicitly says this is distance².
INVALID_PROJ_DIST_THRESHOLD = INVALID_PROJ_DIST2_THRESHOLD


def _validate_barron_params(lam: float, c: float) -> tuple[float, float]:
    """Validate parameters for a monotone mapping d>=0 -> y in [0,1)."""
    try:
        lam, c = float(lam), float(c)
    except (TypeError, ValueError) as exc:
        raise ValueError("lam and c must be real finite scalars") from exc
    if not np.isfinite(lam) or not np.isfinite(c):
        raise ValueError("lam and c must be finite")
    if lam >= -1.0:
        raise ValueError("lam must be less than -1 for d>=0 to map monotonically into [0,1)")
    if c <= 0.0 or lam * c >= 0.0:
        raise ValueError("c must be positive and lam*c must be negative")
    return lam, c


def _numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array, got dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be real, got complex dtype {array.dtype}")
    return array


_validate_barron_params(LAMBDA, C)


def barron_transform(d_m: np.ndarray, lam: float = LAMBDA, c: float = C) -> np.ndarray:
    """metric depth (d>0) -> normalized distance y in [0,1)."""
    lam, c = _validate_barron_params(lam, c)
    d = _numeric_array(d_m, "d_m")
    if np.any(~np.isfinite(d)) or np.any(d < 0):
        raise ValueError("barron_transform requires finite, non-negative depth")
    return 1.0 - (1.0 - d / (lam * c)) ** (lam + 1.0)


def barron_inverse(y: np.ndarray, lam: float = LAMBDA, c: float = C) -> np.ndarray:
    """inverse of barron_transform: y in [0,1) -> metric depth."""
    lam, c = _validate_barron_params(lam, c)
    y = _numeric_array(y, "y")
    if np.any(~np.isfinite(y)) or np.any(y < 0.0) or np.any(y >= 1.0):
        raise ValueError("barron_inverse requires finite y in [0,1)")
    return lam * c * (1.0 - (1.0 - y) ** (1.0 / (lam + 1.0)))


def path_encode(y: np.ndarray) -> np.ndarray:
    """y in [0,1] -> RGB in [0,1]^3, linear interpolation along CUBE_PATH."""
    y = _numeric_array(y, "y")
    if np.any(~np.isfinite(y)):
        raise ValueError("path_encode requires finite y")
    t = np.clip(y, 0.0, 1.0) * N_SEGMENTS               # position along path, [0, N_SEGMENTS]
    seg = np.clip(t.astype(np.int64), 0, N_SEGMENTS - 1)  # segment index
    frac = t - seg
    a, b = CUBE_PATH[seg], CUBE_PATH[seg + 1]
    return a + frac[..., None] * (b - a)


def path_decode(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB in [0,1]^3 -> (y in [0,1], min squared distance to the nearest path segment).

    Nearest-line-segment projection: for each of the 7 edges, project onto the segment
    (clamped to the segment), keep the closest one across all 7 (paper: "project onto the
    nearest line segment").
    """
    rgb01 = _numeric_array(rgb01, "rgb01")
    if rgb01.ndim < 1 or rgb01.shape[-1] != 3:
        raise ValueError(f"path_decode expects shape [..., 3], got {rgb01.shape}")
    if np.any(~np.isfinite(rgb01)) or np.any(rgb01 < 0.0) or np.any(rgb01 > 1.0):
        raise ValueError("path_decode requires finite RGB values in [0,1]")
    rgb01 = rgb01.astype(np.float32, copy=False)
    best_t, best_d2 = None, None
    for i in range(N_SEGMENTS):
        a, ab = PATH_STARTS[i], PATH_VECTORS[i]
        denom = float(PATH_LENGTHS2[i])
        frac = np.clip(((rgb01 - a) * ab).sum(-1) / denom, 0.0, 1.0)
        proj = a + frac[..., None] * ab
        d2 = ((rgb01 - proj) ** 2).sum(-1)
        t = i + frac
        if best_d2 is None:
            best_d2, best_t = d2, t
        else:
            better = d2 < best_d2
            best_d2 = np.where(better, d2, best_d2)
            best_t = np.where(better, t, best_t)
    return best_t / N_SEGMENTS, best_d2


def encode_depth(depth_m: np.ndarray, lam: float = LAMBDA, c: float = C) -> np.ndarray:
    """depth array [...] meters -> uint8 RGB [..., 3] (normally [H,W] -> [H,W,3]).

    Invalid pixels (d<=MIN_VALID, d>=MAX_VALID, NaN, Inf) are encoded as INVALID_SENTINEL.
    MIN_VALID and MAX_VALID are strict, excluded boundaries.
    """
    _validate_barron_params(lam, c)
    d = _numeric_array(depth_m, "depth_m")
    if d.ndim < 1:
        raise ValueError("encode_depth expects an array with at least one dimension")
    d = d.astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > MIN_VALID) & (d < MAX_VALID)
    y = np.zeros_like(d)
    y[valid] = barron_transform(d[valid], lam, c)
    rgb01 = path_encode(y)
    rgb = np.round(np.clip(rgb01, 0, 1) * 255.0).astype(np.uint8)
    rgb[~valid] = INVALID_SENTINEL
    return rgb


def decode_depth(rgb: np.ndarray, lam: float = LAMBDA, c: float = C) -> np.ndarray:
    """uint8 RGB [..., H, W, 3] -> depth [..., H, W] meters.

    Only uint8 input is accepted. Pixels far from the path, or whose decoded depth lies outside
    the strict engineering range (MIN_VALID, MAX_VALID), decode to NaN.
    """
    _validate_barron_params(lam, c)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        raise TypeError(f"decode_depth expects uint8 RGB in [0,255], got dtype {rgb.dtype}")
    if rgb.ndim < 2 or rgb.shape[-1] != 3:
        raise ValueError(f"decode_depth expects shape [..., 3], normally [..., H, W, 3], got {rgb.shape}")
    rgb01 = rgb.astype(np.float32) / 255.0
    y, d2 = path_decode(rgb01)
    depth = barron_inverse(np.clip(y, 0.0, 1.0 - 1e-6), lam, c)
    valid = ((d2 <= INVALID_PROJ_DIST2_THRESHOLD) & np.isfinite(depth)
             & (depth > MIN_VALID) & (depth < MAX_VALID))
    return np.where(valid, depth, np.nan).astype(np.float32, copy=False)


def roundtrip_absrel(depth_m: np.ndarray, **kw) -> float:
    d = _numeric_array(depth_m, "depth_m").astype(np.float32, copy=False)
    valid = np.isfinite(d) & (d > MIN_VALID) & (d < MAX_VALID)
    if not np.any(valid):
        raise ValueError("roundtrip_absrel requires at least one valid depth pixel")
    dec = decode_depth(encode_depth(depth_m, **kw), **kw)
    err = np.abs(dec[valid] - d[valid]) / d[valid]
    return float(np.mean(err))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run depth-codec acceptance on real depth files")
    # repo_root/data/rlbench_selfgen is the symlink to the selfgen tree (see FORK_CHANGES.md);
    # parents[2] is this repo's root now that the module lives at training/percep/.
    default_glob = str(Path(__file__).resolve().parents[2] / "data" / "rlbench_selfgen" / "*"
                       / "variation0" / "episodes" / "episode0" / "view1" / "depth.npz")
    parser.add_argument("--input-glob", default=default_glob)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    if args.max_files <= 0 or args.threshold <= 0:
        parser.error("--max-files and --threshold must be positive")

    import glob
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        parser.error(f"no depth files matched: {args.input_glob}")
    errs = []
    for p in paths[:args.max_files]:
        depth = np.load(p)["depth"].astype(np.float32)   # [T,H,W] f16 saved
        errs.append(roundtrip_absrel(depth))
        task = Path(p).relative_to(Path(p).parents[5]).parts[0]
        print(f"{task}: frames={depth.shape[0]} range=({depth.min():.2f},{depth.max():.2f})m "
              f"roundtrip AbsRel={errs[-1]*100:.3f}%")
    m = float(np.mean(errs))
    passed = m < args.threshold
    print(f"DEPTH_CODEC_{'PASS' if passed else 'FAIL'} mean AbsRel={m*100:.3f}% "
          f"(threshold {args.threshold*100:g}%)")
    if not passed:
        raise SystemExit(1)
