"""Colosseum-style domain randomisation for the self-gen RLBench tree.

Reproduces what the official ActionImages RLBench release does, which the paper describes
only as "Robot-Colosseum background augmentation" (p.10) but which measurement on
`ActionImages/data/rlbench` shows to be considerably more:

  measured on the official tree            selfgen v2 / 512 (no aug)
  camera position std   0.39-0.52 m        0  (fixed at (1.350, 0, 1.580))
  optical-axis hit point std  0.07-0.13 m  --
  table colour/texture  randomised         never changed
  light colour          randomised         fixed
  grasp height std      0.1-1.3 mm         0.2 mm

WHAT IS AND IS NOT RANDOMISED HERE

Enabled: camera pose, table colour, table texture, background texture, light colour.
All five are scene-level -- they target objects that exist in EVERY stock RLBench scene
(`diningTable_visible`, `Floor`, `Wall1..4`, `DefaultLightA/B/D`, `cam_*`), so they work on
all 16 self-gen tasks without per-task configuration. Colosseum only ships configs for 20 of
RLBench's 100 tasks and only 5 of ours, so anything needing per-task `targets` would have
been the expensive path.

Deliberately NOT enabled:

* `object_color` / `object_texture` -- colour IS the instruction semantics for most of our
  tasks ("close the RED jar", "push the MAROON button", "stack 2 RED blocks"). Recolouring
  objects silently breaks language-vision correspondence. Colosseum avoids this by hand-listing
  `targets` per task, and the official release's 5 tasks have no colour words at all. If this
  is wanted later, restrict it to objects NOT referenced by `seg_targets.json`.
* `distractor_object` -- targets `spawn_boundary0`, which exists in Colosseum's own task .ttm
  scenes but not in stock RLBench.
* `object_size` / `object_mass` / `object_friction` -- the official release does not use them:
  grasp height std there is 0.1-1.3 mm, while Colosseum's `scale_range` of [0.75, 1.15] would
  move it by centimetres. The paper mentions no physics randomisation either.

WHY THE CAMERA IS IMPLEMENTED HERE RATHER THAN TAKEN FROM COLOSSEUM

`CameraPoseVariation` adds a position delta and a small euler delta to the initial pose and
never re-aims. That cannot produce the official distribution: there the camera positions have
std ~0.4 m yet 97-100% of optical axes still cross the table plane within 0.07-0.13 m of a
common point, i.e. the cameras are re-aimed at the workspace. Measured in spherical
coordinates about that point, the radius is nearly CONSTANT per view (view1 1.32+-0.03 m,
view3 1.74+-0.02 m) while elevation spans -15..85 deg and azimuth is close to uniform. That is
an orbit on a sphere with a look-at, which is what `_randomize_cameras` below does.
"""
from __future__ import annotations

import glob
import os

import numpy as np

# Where each task's manipulation actually happens: the mean gripper position over every episode
# of that task in the un-augmented 512 tree. Aiming here rather than at a single global point is
# load-bearing. The official tree's optical axes converge within 0.07-0.13 m of (0.33, 0.09) ON
# THE z=0.78 PLANE, but that is a plane INTERSECTION, not the aim point -- the manipulation sits
# 0.2-0.45 m above the table (z = 0.94..1.24 below). Aiming at the table surface put the objects
# out of the top of frame at high elevations: measured on a 3-episode probe, 25% of views had the
# task objects covering <0.05% of pixels and the median coverage fell from 2.15% to 0.53%.
TASK_CENTERS = {
    "close_jar": (0.263, 0.065, 0.991),
    "insert_onto_square_peg": (0.256, 0.008, 1.022),
    "light_bulb_in": (0.26, -0.02, 1.127),
    "meat_off_grill": (0.244, 0.005, 1.221),
    "open_drawer": (0.25, 0.105, 1.097),
    "place_shape_in_shape_sorter": (0.251, -0.018, 0.995),
    "push_buttons": (0.264, -0.009, 1.093),
    "put_groceries_in_cupboard": (0.249, 0.014, 1.243),
    "put_item_in_drawer": (0.245, 0.115, 1.174),
    "put_money_in_safe": (0.135, 0.048, 1.211),
    "reach_and_drag": (0.249, 0.02, 1.061),
    "slide_block_to_target": (0.268, -0.006, 1.04),
    "stack_blocks": (0.24, 0.014, 0.944),
    "stack_wine": (0.298, -0.144, 1.057),
    "sweep_to_dustpan": (0.257, 0.037, 1.177),
    "turn_tap": (0.251, 0.035, 1.057),
}
# Tasks not in the table (or a typo) fall back to the mean of the 16 above.
DEFAULT_CENTER = np.array([0.25, 0.02, 1.09])
LOOKAT_JITTER = 0.08          # m, 1-sigma; official hit-point std is 0.07-0.13 m
RADIUS_JITTER = (0.97, 1.03)  # official radius std is 2-3% of the radius
ELEV_SIGMA_DEG = 15.0         # official: std 11-19 deg about a per-view mean
ELEV_CLIP_DEG = (10.0, 85.0)  # official range: -15 .. +85 deg, clipped low to keep the
                              # camera above the table plane
# Azimuth is NOT uniform. A plain std on the raw angles reads 63-127 deg and looks uniform, but
# that statistic is meaningless on a circular variable. Measured properly against each view's own
# default azimuth, the mean resultant length is R_bar = 0.56-0.71 (uniform would be ~0), i.e. a
# circular std of 48-62 deg. Sampling uniformly instead throws the cameras behind walls and under
# the table, and the manipulation leaves frame -- verified by generating 3 episodes that way.
AZIM_SIGMA_DEG = 55.0
CAMERAS = ("cam_front", "cam_overhead", "cam_over_shoulder_left", "cam_over_shoulder_right")

# PyRep 4.1.0.3 reports ObjectType.LIGHT as "not supported", so `get_objects_in_tree` returns
# nothing and Colosseum's LightColorVariation finds no targets. The lights are still there and
# still settable through the raw sim API, so they are driven by handle instead.
LIGHT_NAMES = ("DefaultLightA", "DefaultLightB", "DefaultLightC", "DefaultLightD")
# Colosseum's close_box.yaml uses [[0,0,0],[0.5,0.5,0.5]] for light_color, i.e. it only ever
# DARKENS. Keeping that range reproduces the dim, colour-cast episodes visible in the official
# tree; a range centred on white would not.
LIGHT_DIFFUSE_RANGE = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
LIGHT_SPECULAR = (0.1, 0.1, 0.1)

TABLE_COLOR_RANGE = ((0.25, 0.25, 0.25), (1.0, 1.0, 1.0))  # from close_box.yaml
COLOSSEUM_TEXTURES = "/workspace/ttdu/robot-colosseum/colosseum/assets/textures"


class ColosseumAug:
    """Applies the enabled factors once per episode and reports what it did.

    One instance per worker; call `randomize(task_name, variation, seed)` before `get_demos`.
    The RNG is seeded from (task, variation, seed) with crc32 rather than `hash`, so a tree can
    be regenerated pixel-for-pixel -- see the note in gen_dataset.py.
    """

    def __init__(self, pyrep, textures_folder: str = COLOSSEUM_TEXTURES):
        self._pr = pyrep
        self._textures = textures_folder
        self._initial_cam_pose = {}
        self._light_handles = {}
        self._table_variation = None
        self._table_tex_variation = None
        self._bg_tex_variation = None
        self._n_textures = len(glob.glob(os.path.join(textures_folder, "*.png")))

    # -- setup ---------------------------------------------------------------------------
    def bind(self) -> None:
        """Resolve scene objects. Call once after `env.launch()`, before the first episode."""
        from pyrep.backend import sim
        from pyrep.objects.vision_sensor import VisionSensor

        for name in CAMERAS:
            try:
                cam = VisionSensor(name)
                self._initial_cam_pose[name] = (
                    np.array(cam.get_position()), np.array(cam.get_orientation())
                )
            except Exception:
                pass
        for name in LIGHT_NAMES:
            try:
                self._light_handles[name] = sim.simGetObjectHandle(name)
            except Exception:
                pass

        from colosseum.variations.background_texture import BackgroundTextureVariation
        from colosseum.variations.table_color import TableColorVariation
        from colosseum.variations.table_texture import TableTextureVariation

        # Colosseum's variations own their own RNG, seeded at construction. They are rebuilt
        # per episode in `randomize` so the per-episode seed actually takes effect.
        self._ctors = {
            "table_color": lambda s: TableColorVariation(
                self._pr, None, [], color_range=[list(TABLE_COLOR_RANGE[0]), list(TABLE_COLOR_RANGE[1])], seed=s
            ),
            "table_texture": lambda s: TableTextureVariation(self._pr, None, self._textures, seed=s),
            "background_texture": lambda s: BackgroundTextureVariation(self._pr, None, self._textures, seed=s),
        }

    # -- per episode ---------------------------------------------------------------------
    def randomize(self, rng: np.random.Generator, task_name: str = "") -> dict:
        center = np.array(TASK_CENTERS.get(task_name, DEFAULT_CENTER), dtype=float)
        meta = {"applied": True, "factors": {}, "center": [round(float(x), 3) for x in center]}
        meta["factors"]["cameras"] = self._randomize_cameras(rng, center)
        meta["factors"]["light"] = self._randomize_lights(rng)
        for name, ctor in self._ctors.items():
            try:
                ctor(int(rng.integers(0, 2**31 - 1))).randomize()
                meta["factors"][name] = True
            except Exception as e:  # a factor failing must not lose the episode
                meta["factors"][name] = f"FAILED: {type(e).__name__}"
        return meta

    def _randomize_cameras(self, rng: np.random.Generator, center: np.ndarray) -> dict:
        from scipy.spatial.transform import Rotation
        from pyrep.objects.vision_sensor import VisionSensor

        out = {}
        for name, (p0, _) in self._initial_cam_pose.items():
            d0 = p0 - center
            r0 = float(np.linalg.norm(d0))
            elev0 = float(np.degrees(np.arcsin(np.clip(d0[2] / r0, -1, 1))))

            # Radius is held near its original value: it is what sets how much of the scene a
            # view frames, and the official tree keeps it to ~2%. Elevation walks about the
            # camera's own default rather than a global mean, because the four views sit at
            # genuinely different heights (official view4 is +38 deg vs view1's +57 deg).
            r = r0 * float(rng.uniform(*RADIUS_JITTER))
            elev = np.clip(rng.normal(elev0, ELEV_SIGMA_DEG), *ELEV_CLIP_DEG)
            azim0 = float(np.arctan2(d0[1], d0[0]))
            azim = azim0 + np.radians(rng.normal(0.0, AZIM_SIGMA_DEG))
            e = np.radians(elev)
            pos = center + r * np.array(
                [np.cos(e) * np.cos(azim), np.cos(e) * np.sin(azim), np.sin(e)]
            )

            target = center + rng.normal(0.0, LOOKAT_JITTER, 3)
            f = target - pos
            f /= np.linalg.norm(f)
            # Degenerate only if the camera looks straight down the world z axis; the elevation
            # clip at 85 deg keeps it away from that.
            right = np.cross(np.array([0.0, 0.0, 1.0]), f)
            right /= np.linalg.norm(right)
            up = np.cross(f, right)
            # CoppeliaSim vision sensors look along +z of their own frame (verified against the
            # stock front camera: its +z column points at the table).
            rot = Rotation.from_matrix(np.stack([right, up, f], axis=1))

            cam = VisionSensor(name)
            cam.set_position(pos.tolist())
            # "XYZ" (intrinsic), NOT "xyz" (extrinsic). CoppeliaSim's Euler convention is
            # intrinsic: verified by round-trip -- from_matrix(M).as_euler("XYZ") reproduces
            # get_orientation() exactly while "xyz" does not, and setting the "xyz" angles back
            # does not restore the original matrix. Getting this wrong silently mis-aims every
            # camera, which is what made the task objects leave frame in 50% of views.
            cam.set_orientation(rot.as_euler("XYZ").tolist())
            out[name] = {"radius": round(r, 4), "elev_deg": round(float(elev), 1),
                         "azim_deg": round(float(np.degrees(azim)), 1)}
        return out

    def _randomize_lights(self, rng: np.random.Generator) -> dict:
        from pyrep.backend import sim

        lo, hi = np.array(LIGHT_DIFFUSE_RANGE[0]), np.array(LIGHT_DIFFUSE_RANGE[1])
        diffuse = rng.uniform(lo, hi)
        for handle in self._light_handles.values():
            try:
                sim.simSetLightParameters(handle, 1, diffuse.tolist(), list(LIGHT_SPECULAR))
            except Exception:
                return {"error": "simSetLightParameters failed"}
        return {"diffuse": [round(float(c), 3) for c in diffuse]}

    def restore_cameras(self) -> None:
        """Put the cameras back before the next episode's randomisation.

        Not strictly required (randomize() recomputes from the stored initial position, not
        from the current one), but it keeps the scene sane if a demo attempt fails and the
        episode is retried.
        """
        from pyrep.objects.vision_sensor import VisionSensor

        for name, (p0, o0) in self._initial_cam_pose.items():
            try:
                cam = VisionSensor(name)
                cam.set_position(p0.tolist())
                cam.set_orientation(o0.tolist())
            except Exception:
                pass
