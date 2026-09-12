"""Surface-normal <-> RGB codec, derived from the metric depth we already have.

Ported from /workspace/ttdu/ttd/src/percep/normal_codec.py with two corrections (below). No new
data collection: RLBench selfgen stores metric depth in `view*/depth.npz` and the pinhole
intrinsics in `view*/camera_params.json`, which is everything a camera-frame normal needs. The
(n+1)/2 -> RGB visualisation is the same convention as the "Surface Normal" panel of Argus
(Zhuang et al., CVPR 2025), so the stream is directly comparable to that literature.

CORRECTION 1 -- intrinsics. The ttd version took `np.gradient(depth)` and used it directly, which
is dz/dpixel, not dz/dmetre. A pixel subtends z/fx metres at depth z, so that version's normals
are wrong by a factor of fx/z ~ 200-750 on this data and systematically flattened: every surface
tilts toward the camera as it recedes. Correct form (right-handed camera frame, +z forward):

    n ~ ( -(dz/du) * fx / z ,  -(dz/dv) * fy / z ,  1 ),  normalised

RLBench writes fx, fy NEGATIVE in camera_params.json (its image y axis points the other way), so
callers pass magnitudes and `depth_to_normal` takes abs() defensively -- a sign error here flips
the red and green channels and looks almost plausible, which is the worst kind of bug to ship.

CORRECTION 2 -- no invalid sentinel. The sky above the room walls is rendered at a CONSTANT far
clip plane (measured 3.3008 m, 23% of some views), which is inside depth_codec's [MIN_VALID,
MAX_VALID] and therefore already encoded by `<depth>` as an ordinary surface. Constant depth has
zero gradient, so it lands on (0,0,1) -- a flat plane facing the camera, pale blue. That is the
consistent choice: `<depth>` and `<normal>` must not disagree about what the sky is. Pinned by
tests/test_normal_codec.py::test_constant_depth_is_a_facing_plane.

Pure functions. encode_normal_from_depth([T,H,W] depth, fx, fy) -> [T,H,W,3] uint8 RGB.
"""
import numpy as np

# Guard for the 1/z factor. Depth this small never occurs in RLBench (measured working range
# 0.88-4.16 m) and would only arise from a corrupt frame; clamping keeps the gradient finite
# instead of producing NaNs that silently propagate into the training target.
MIN_DEPTH = 1e-3


def depth_to_normal(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """[H,W] metric depth + pinhole focal lengths (pixels) -> [H,W,3] unit normal, camera frame.

    Central differences via np.gradient; the border uses one-sided differences, which is what
    np.gradient does and is fine here because RLBench's frame border is always wall or floor.
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth_to_normal expects [H,W], got {d.shape}")
    z = np.clip(d, MIN_DEPTH, None)
    # np.gradient returns (d/drow, d/dcol) = (d/dv, d/du).
    dzdv, dzdu = np.gradient(d)
    nx = -dzdu * (abs(float(fx)) / z)
    ny = -dzdv * (abs(float(fy)) / z)
    n = np.stack([nx, ny, np.ones_like(d)], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-6, None)


def encode_normal_from_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """[T,H,W] metric depth -> [T,H,W,3] uint8 RGB, normal shown as (n+1)/2."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 3:
        raise ValueError(f"encode_normal_from_depth expects [T,H,W], got {d.shape}")
    n = np.stack([depth_to_normal(d[t], fx, fy) for t in range(d.shape[0])], axis=0)
    # np.round, not a bare cast: depth_codec.encode_depth quantises the same way
    # (`np.round(clip * 255)`), and truncation would send the commonest normal of all -- (0,0,1),
    # every flat surface facing the camera -- to 127 instead of 128, i.e. off-centre in every
    # channel for no reason.
    return np.clip(np.round((n + 1.0) * 127.5), 0, 255).astype(np.uint8)


def decode_normal(rgb: np.ndarray) -> np.ndarray:
    """[...,3] uint8 RGB -> unit normal in [-1,1]. Inverse of the (n+1)/2 mapping.

    Renormalises rather than trusting the decoded triple to be unit length: uint8 quantisation
    alone perturbs it by up to ~0.4%, and a VAE roundtrip by far more.
    """
    n = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-6, None)


def normal_cos(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean cosine similarity between two unit-normal fields [...,3]. 1.0 = identical."""
    return float((np.asarray(pred, np.float32) * np.asarray(gt, np.float32)).sum(-1).mean())


def roundtrip_cos(depth: np.ndarray, fx: float, fy: float) -> float:
    """Mean cosine between the GT normal and decode(encode(GT normal)) -- the codec's own loss.

    This measures uint8 quantisation only. The number that decides whether the stream is
    trainable is the VAE roundtrip, which is scripts/vae_roundtrip_normal.py.
    """
    d = np.asarray(depth, dtype=np.float32)
    gt = np.stack([depth_to_normal(d[t], fx, fy) for t in range(d.shape[0])], axis=0)
    return normal_cos(decode_normal(encode_normal_from_depth(d, fx, fy)), gt)
