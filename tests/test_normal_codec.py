"""Acceptance test for normal_codec: geometry, the two ported corrections, quantised roundtrip.

Run: python tests/test_normal_codec.py   (prints NORMAL_CODEC_OK on success)

CPU-only, mostly analytic. The load-bearing tests are the two that pin the corrections made when
porting from ttd (see the module docstring of training/percep/normal_codec.py):
test_slanted_plane_recovers_its_true_normal -- without the fx/z factor the recovered normal is
wrong by a factor of ~200-750 and no synthetic test that only checks "is a unit vector" notices;
and test_constant_depth_is_a_facing_plane -- the sky must agree with what depth_codec does with it.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from training.percep.normal_codec import (  # noqa: E402
    decode_normal,
    depth_to_normal,
    encode_normal_from_depth,
    normal_cos,
    roundtrip_cos,
)

FAILURES = []
FX = FY = 703.3542416031569  # measured from rlbench_selfgen_512_aug camera_params.json


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def _plane_depth(h, w, fx, fy, normal, z0=2.0):
    """Render the depth map of a plane through (0,0,z0) with the given camera-frame normal.

    Plane: n . X = n_z * z0. With X = (z*(u-cx)/fx, z*(v-cy)/fy, z) this solves for z per pixel,
    which is the exact inverse of what depth_to_normal is supposed to recover.
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy
    nx, ny, nz = normal
    return (nz * z0 / (nx * x_over_z + ny * y_over_z + nz)).astype(np.float32)


def test_constant_depth_is_a_facing_plane():
    """Correction 2: the sky sits at a constant far clip plane and must read as facing the camera.

    depth_codec encodes it as an ordinary surface (3.3008 m is inside [MIN_VALID, MAX_VALID]),
    so <normal> disagreeing -- e.g. by emitting an invalid sentinel -- would make the two streams
    describe different scenes from the same GT.
    """
    n = depth_to_normal(np.full((32, 32), 3.3008, np.float32), FX, FY)
    check(np.allclose(n, np.array([0.0, 0.0, 1.0]), atol=1e-6), "constant depth -> (0,0,1) everywhere")
    rgb = encode_normal_from_depth(np.full((2, 16, 16), 3.3008, np.float32), FX, FY)
    check(set(map(tuple, rgb.reshape(-1, 3).tolist())) == {(128, 128, 255)},
          f"and encodes to a single pale-blue value (got {set(map(tuple, rgb.reshape(-1, 3).tolist()))})")


def test_slanted_plane_recovers_its_true_normal():
    """Correction 1: with the fx/z factor the recovered normal matches the plane's analytically.

    Without it the gradient is dz/dpixel and the recovered normal is off by fx/z ~ 350 at 2 m,
    which drives cos to ~0. The margin between the two cases is enormous, which is exactly why
    the un-corrected version could have shipped looking fine.
    """
    for normal in ([0.0, 0.0, 1.0], [0.3, 0.0, 0.9539], [-0.4, 0.5, 0.7681], [0.0, -0.6, 0.8]):
        n = np.array(normal, np.float32)
        n /= np.linalg.norm(n)
        d = _plane_depth(96, 96, FX, FY, n)
        got = depth_to_normal(d, FX, FY)[8:-8, 8:-8]  # drop the one-sided-difference border
        cos = normal_cos(got, np.broadcast_to(n, got.shape))
        check(cos > 0.999, f"plane n={np.round(n, 3).tolist()} recovered (cos={cos:.6f})")


def test_gradient_scaling_actually_uses_intrinsics():
    """A different fx must give a different normal -- i.e. the argument is not ignored."""
    d = _plane_depth(64, 64, FX, FY, np.array([0.3, 0.0, 0.9539], np.float32))
    a = depth_to_normal(d, FX, FY)
    b = depth_to_normal(d, FX * 2.0, FY * 2.0)
    check(not np.allclose(a, b), "doubling fx changes the normal (intrinsics are used)")
    facing = np.array([0.0, 0.0, 1.0])
    # nx scales WITH fx, so a larger fx makes the same dz/dpixel a steeper metric slope and tilts
    # the normal further AWAY from facing the camera.
    check(normal_cos(b, np.broadcast_to(facing, b.shape))
          < normal_cos(a, np.broadcast_to(facing, a.shape)),
          "and a larger fx tilts the normal further from facing (nx scales with fx)")


def test_sign_convention_is_stable_under_negative_focal_lengths():
    """RLBench writes fx, fy negative. abs() inside the codec must make that a no-op."""
    d = _plane_depth(64, 64, FX, FY, np.array([0.3, -0.4, 0.866], np.float32))
    check(np.array_equal(depth_to_normal(d, FX, FY), depth_to_normal(d, -FX, -FY)),
          "depth_to_normal(fx) == depth_to_normal(-fx)")


def test_outputs_are_unit_vectors_and_in_range():
    rng = np.random.default_rng(0)
    d = (2.0 + 0.4 * rng.standard_normal((3, 48, 48))).astype(np.float32)
    n = np.stack([depth_to_normal(d[t], FX, FY) for t in range(3)])
    check(np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 1e-5, "every normal is unit length")
    rgb = encode_normal_from_depth(d, FX, FY)
    check(rgb.dtype == np.uint8 and rgb.shape == (3, 48, 48, 3), f"encode shape/dtype {rgb.shape}/{rgb.dtype}")
    check(decode_normal(rgb).shape == n.shape, "decode restores [T,H,W,3]")


def test_quantised_roundtrip_gate():
    """The codec's own loss must be negligible; anything worse than 0.99 is an encoding bug.

    Real depth, not synthetic: RLBench depth is stored float16, so it carries quantisation of its
    own that a smooth synthetic field would hide.
    """
    import glob
    import json

    eps = sorted(glob.glob(os.path.join(
        REPO, "data", "rlbench_selfgen_512_aug", "*", "variation0", "episodes", "episode*")))
    if not eps:
        sys.exit("FAIL: no data tree at data/rlbench_selfgen_512_aug")
    worst = (1.0, None)
    for ep in eps[:: max(1, len(eps) // 8)][:8]:
        vp = os.path.join(ep, "view1")
        d = np.load(os.path.join(vp, "depth.npz"))["depth"].astype(np.float32)[:6]
        with open(os.path.join(vp, "camera_params.json")) as f:
            k = json.load(f)["0"]["intrinsics"]
        cos = roundtrip_cos(d, k[0][0], k[1][1])
        if cos < worst[0]:
            worst = (cos, os.path.basename(ep))
    check(worst[0] >= 0.99, f"uint8 roundtrip cos >= 0.99 on real depth (worst {worst[0]:.6f} at {worst[1]})")


def main():
    for fn in [
        test_constant_depth_is_a_facing_plane,
        test_slanted_plane_recovers_its_true_normal,
        test_gradient_scaling_actually_uses_intrinsics,
        test_sign_convention_is_stable_under_negative_focal_lengths,
        test_outputs_are_unit_vectors_and_in_range,
        test_quantised_roundtrip_gate,
    ]:
        print(f"\n{fn.__name__}:")
        fn()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("\nNORMAL_CODEC_OK")


if __name__ == "__main__":
    main()
