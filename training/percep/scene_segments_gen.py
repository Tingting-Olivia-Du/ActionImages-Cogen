"""Offline generator for `<episode>/scene_segments.json` (Scene Role Segmentation v1).

    python -m training.percep.scene_segments_gen --root data/rlbench_selfgen_512_aug [--write]

PROVENANCE: migrated verbatim from ttd/src/percep/scene_segments_gen.py on 2026-08-30 so this
repository is self-contained (it was previously reached through a `sys.path.insert` into the ttd
checkout, which cannot survive a release). The only edits since are the close_box / close_drawer
entries in TASK_ROLES, added for the unseen-task perception evaluation.

Writes, per episode, the INSTRUCTION-INDEPENDENT half of the scene-role annotation:

    {"protocol": "rlbench-scene-role-v1", "task": ..., "instances": {name: {handles, base_role}}}

It deliberately does NOT store target/goal-by-instruction. seg_targets.json + build_referring_spec
is already the single source of truth for "which object does this instruction refer to"
(push_buttons and put_groceries_in_cupboard both pick their target from the text), and a second
copy here is how the two drift apart while every acceptance check keeps passing. The dataset
overlays `target` at load time -- see rlbench_selfgen._scene_role_lut.

Roles are resolved from the OBJECT NAME in handles.json, never from the handle id: measured on
this tree, handle 44 is `Panda_link2_visual` in most tasks but a different object in
put_groceries_in_cupboard, so id-based rules are wrong across tasks and unstable across episodes.

Name rules are EXPLICIT per task rather than substring-guessed. seg_targets_gen.py's audit
already established that same-prefix names can denote different instances and that a handle whose
name looks like part of an object can render another part -- so a generic "contains 'target' =>
target" rule is not safe. Anything a rule does not cover stays unmapped, becomes `unknown` at
encode time, and is reported here; it is never silently folded into background.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

# --- global rules, applied to every task ------------------------------------------------
# Panda_link{0..7}_visual. The gripper/finger links are NOT arm links -- they get their own role
# so the model can distinguish "the arm is nearby" from "the gripper is at the object".
ROBOT_ARM_NAMES = tuple(f"Panda_link{i}_visual" for i in range(8))
GRIPPER_NAMES = ("Panda_gripper_visual", "Panda_leftfinger_visual", "Panda_rightfinger_visual")
# Static environment. `workspace` and `spawn_boundary` are invisible helper planes in RLBench but
# they DO occupy handle-map pixels, so they must be classified rather than left unmapped.
BACKGROUND_NAMES = (
    "ResizableFloor_5_25_visibleElement",
    "Wall1", "Wall2", "Wall3", "Wall4",
    # The room ceiling. Same class as the walls and the floor, and listed for the same reason --
    # but it took a 5x bigger tree to find it. `rlbench_selfgen_512_aug` (70 episodes/task) has
    # Roof in handles.json for ZERO episodes; `rlbench_selfgen_512_aug_wide` (270/task) has it in
    # exactly 3, because Colosseum camera randomisation occasionally draws an elevation that looks
    # up past the wall tops. The handle-union then propagates those 3 to all 540 episodes of the
    # two affected tasks, so 3 rare renders turned into "UNMAPPED: Roof x270" and would have made
    # `_check_scene_roles_ready` refuse the whole tree.
    #
    # Adding it cannot change any existing checkpoint's targets: the 512_aug tree Roof never
    # appears in, so regenerating it would produce byte-identical output. Verified by grepping
    # every handles.json in both trees before making this change.
    "Roof",
    "diningTable_visible",
    "workspace",
    "spawn_boundary",
    "dollar_stack_boundary",
)

# RLBench renders "no object" (the sky above the room walls) as CoppeliaSim's -1. That reaches
# handles.json as 16777215 = 0xFFFFFF with an EMPTY name (simGetObjectName finds nothing), and
# reaches mask.npz as 65535 = 0xFFFF because gen_dataset.py stores the map as uint16. It is empty
# space, not an annotation gap.
#
# It is deliberately NOT emitted into scene_segments.json: seg_codec.build_role_lut allocates a
# 65536-wide LUT, so 16777215 is not a storable key, and the codec already maps its own
# NO_OBJECT_HANDLE (0xFFFF) to background. Recording it here would only create a second, wrong
# source of truth. What this constant is for is keeping the coverage gate honest -- before it
# existed, every episode reported an unmapped "" and main() returned COVERAGE INCOMPLETE, which
# is why the 1028 scene_segments.json files on disk were written past a failing gate.
NO_OBJECT_HANDLES = frozenset((0xFFFFFF, 0xFFFF))

# --- per-task rules ----------------------------------------------------------------------
# {task: {object_name: base_role}}. `target` never appears here: it is an instruction-dependent
# overlay applied at load time. An object that CAN be the target gets `distractor`, which is what
# it is on any sample where the instruction names a different one.
TASK_ROLES = {
    "close_jar": {
        "jar0": "goal", "jar1": "goal",          # the jar body receives the lid
        "jar_lid0": "distractor",
    },
    "insert_onto_square_peg": {
        "square_base": "fixture",
        "pillar0": "goal", "pillar1": "goal", "pillar2": "goal",
        "square_ring": "distractor",
    },
    "light_bulb_in": {
        "bulb0": "goal", "bulb1": "goal",        # the holders the bulb screws into
        "light_bulb0": "distractor", "light_bulb1": "distractor",
    },
    "meat_off_grill": {
        "grill_visual": "fixture",
        "chicken_visual": "distractor", "steak_visual": "distractor",
    },
    "open_drawer": {
        # The three boxes are the MANIPULANDA here (the instruction picks which one to open),
        # so they must be promotable to `target`; only the carcass is a fixture. Contrast
        # put_item_in_drawer below, where the same three boxes are the DESTINATION and so are
        # goals. Same geometry, different role, decided by the task -- which is why these tables
        # are per task rather than one shared name->role map.
        "drawer_frame": "fixture", "drawer_legs": "fixture",
        "drawer_top": "distractor", "drawer_middle": "distractor", "drawer_bottom": "distractor",
    },
    # ---- tasks below are NOT in the training tree -----------------------------------------
    # close_box and close_drawer are evaluated but never trained on (see
    # paper/EXPERIMENT_OUTLINE.md E12): they exist in RLBench and therefore in the closed-loop
    # harness, which builds scenes from the simulator rather than from disk, but they were never
    # rendered into data/rlbench_selfgen_512_aug. Their episodes live in a SEPARATE tree,
    # data/rlbench_unseen_tasks_512_aug, so that no training run can pick them up.
    "close_box": {
        # Same shape as close_jar: a body that receives a lid, plus the lid itself. The lid is
        # the manipulandum and so is `distractor` (promotable to `target` by the instruction
        # overlay), the carcass is the thing it closes onto.
        "box_base": "goal",
        "box_lid": "distractor",
    },
    "close_drawer": {
        # Identical geometry and identical semantics to open_drawer above -- the instruction
        # picks which of the three boxes to close, so all three are manipulanda and only the
        # carcass is fixture. Copied deliberately rather than shared: put_item_in_drawer proves
        # the same three names take a different role under a different task.
        "drawer_frame": "fixture", "drawer_legs": "fixture",
        "drawer_top": "distractor", "drawer_middle": "distractor", "drawer_bottom": "distractor",
    },
    "close_microwave": {
        # Same shape as close_drawer: an articulated part the robot moves, on a static carcass.
        "microwave_door": "distractor",
        "microwave_frame_vis": "fixture",
    },
    "take_lid_off_saucepan": {
        # Inverse of close_box -- the lid is removed rather than placed, but the role assignment
        # is unchanged: the lid is the manipulandum, the pan is what it comes off.
        "saucepan_lid_visual": "distractor",
        "saucepan_visual": "goal",
    },
    "toilet_seat_down": {
        # `toilet_seat_up_seat` is the hinged part that gets lowered; the other two names render
        # the static bowl. Kept as separate instances because role, not instance, drives colour.
        "toilet_seat_up_seat": "distractor",
        "toilet_seat_up_toilet": "fixture",
        "toilet": "fixture",
    },
    "basketball_in_hoop": {
        # Free-space placement rather than articulation: the ball is carried, the hoop receives it.
        "ball": "distractor",
        "basket_ball_hoop_visual": "goal",
    },
    "place_shape_in_shape_sorter": {
        "shape_sorter": "goal", "shape_sorter_visual": "goal",
        "cube": "distractor", "cylinder": "distractor", "moon_visual": "distractor",
        "star_visual": "distractor", "triangular_prism": "distractor",
    },
    "push_buttons": {
        # Every button is a candidate target; the instruction picks which. topPlate/wrap are the
        # same physical button split into several handles -- same role, so they merge visually.
        **{f"push_buttons_target{i}": "distractor" for i in range(4)},
        **{f"target_button_topPlate{i}": "distractor" for i in range(4)},
        **{f"target_button_wrap{i}": "distractor" for i in range(4)},
    },
    "put_groceries_in_cupboard": {
        "cupboard": "goal",
        **{f"{n}_visual": "distractor" for n in (
            "chocolate_jello", "coffee", "crackers", "mustard", "soup", "spam",
            "strawberry_jello", "sugar", "tuna",
        )},
    },
    "put_item_in_drawer": {
        "drawer_frame": "fixture", "drawer_legs": "fixture",
        "drawer_top": "goal", "drawer_middle": "goal", "drawer_bottom": "goal",
        "item": "distractor",
    },
    "put_money_in_safe": {
        "safe_body": "goal", "safe_door": "fixture",
        "dollar_stack": "distractor",
    },
    "reach_and_drag": {
        "stick": "tool",                          # the implement, not the thing being acted on
        "cube": "distractor",
        # The drag destination. Visible in only 14 of 2818 episodes -- consistent with the ~94%
        # occlusion rate already documented for this task's "target" group in seg_targets.json,
        # which is why it is absent from most handles.json files rather than a coverage bug.
        "target0": "goal",
    },
    "slide_block_to_target": {
        "target": "goal", "block": "distractor",
    },
    "stack_blocks": {
        "stack_blocks_target_plane": "goal",
        **{f"stack_blocks_target{i}": "distractor" for i in range(5)},
        **{f"stack_blocks_distractor{i}": "distractor" for i in range(5)},
    },
    "stack_wine": {
        "rack_bottom_visual": "goal", "rack_top_visual": "goal",
        "wine_bottle_visual": "distractor",
    },
    "sweep_to_dustpan": {
        **{f"Dustpan_{i}": "goal" for i in range(3, 7)},
        "sweep_to_dustpan_broom_visual": "tool",
        "broom_holder": "fixture",
        **{f"dirt{i}": "distractor" for i in range(10)},
    },
    "turn_tap": {
        "tap_main_visual": "fixture",
        "tap_left_visual": "distractor", "tap_right_visual": "distractor",
    },
}


def resolve_roles(task: str, handles: dict):
    """{handle_id_str: object_name} -> ({name: {handles, base_role}}, [unmapped (id, name)]).

    Handles sharing a role AND a logical object are merged into one instance; the instance name
    is the object name, so `shape_sorter` and `shape_sorter_visual` stay separate entries with
    the same role rather than being guessed into one object. They render as one colour anyway
    (role, not instance, drives colour), which is the point of §6.1.
    """
    task_map = TASK_ROLES.get(task, {})
    instances, unmapped = {}, []
    for hid, name in handles.items():
        if name in ROBOT_ARM_NAMES:
            role, inst = "robot_arm", "robot_arm"
        elif name in GRIPPER_NAMES:
            role, inst = "gripper", "robot_gripper"
        elif name in BACKGROUND_NAMES:
            role, inst = "background", "environment"
        elif name in task_map:
            role, inst = task_map[name], name
        elif int(hid) in NO_OBJECT_HANDLES and name == "":
            # Empty space, handled by seg_codec.NO_OBJECT_HANDLE. Neither an instance nor a gap.
            continue
        else:
            unmapped.append((int(hid), name))
            continue
        entry = instances.setdefault(inst, {"handles": [], "base_role": role})
        if entry["base_role"] != role:
            raise ValueError(f"{task}: instance {inst!r} got two roles: {entry['base_role']} vs {role}")
        entry["handles"].append(int(hid))
    for e in instances.values():
        e["handles"].sort()
    return instances, unmapped


def task_handle_union(eps, root):
    """[episode paths] -> ({task: {handle_id_str: name}}, [conflict strings]).

    WHY a union rather than each episode's own handles.json. gen_dataset.py:dump_handle_names
    only queries simGetObjectName on THREE frames (`idxs = [0, len(demo)//2, len(demo)-1]`), so an
    object occluded in all three never enters that episode's handles.json -- even though it is
    rendered, and therefore occupies handle-map pixels, in the frames in between. Those pixels then
    have no base_role and decode as `unknown` orange in the training target.

    Measured on rlbench_selfgen_512_aug: Panda_link0_visual is missing from 15 of 1028 episodes,
    Panda_link1_visual from 9, Panda_rightfinger_visual from 1, put_item_in_drawer's drawer_legs
    from 1 -- 66832 orange pixels across 10 (episode, view) pairs, all of them robot links seen
    from view3/view4 or a fixture.

    The union is safe because CoppeliaSim assigns handles when the scene loads and a task's scene
    is identical across its episodes: measured over all 1028 episodes, ZERO handle maps to two
    different names within a task (across tasks they collide constantly, e.g. 92 is drawer_legs,
    steak_visual, dirt4 and four others -- which is why this is keyed by task and merging globally
    would be wrong). Conflicts are returned rather than silently resolved; a non-empty list means
    the assumption broke and the caller must stop.

    The converse is NOT true and does not need to be: one object can hold different handle ids in
    different episodes (stack_wine's wine_bottle_visual is 82 in some, 102 in others -- CoppeliaSim
    numbers by load order and the Colosseum augmentation changes it). The union therefore contains
    entries for handles a given episode never renders. Those cost one unused LUT slot each and
    cannot mis-colour anything, precisely because handle -> name is unique within the task: a
    handle is either that object or absent, never something else. It does inflate the
    "recovered from the task union" counts below, which are per (task, object), not per rendered
    pixel.
    """
    union, conflicts = collections.defaultdict(dict), []
    for ep in eps:
        task = ep[len(root):].strip(os.sep).split(os.sep)[0]
        hp = os.path.join(ep, "handles.json")
        if not os.path.exists(hp):
            continue
        with open(hp) as f:
            handles = json.load(f)
        for hid, name in handles.items():
            prev = union[task].get(hid)
            if prev is not None and prev != name:
                conflicts.append(f"{task}: handle {hid} is {prev!r} and {name!r} (at {ep})")
            union[task][hid] = name
    return union, conflicts


def episode_task(ep: str) -> str:
    return os.path.relpath(ep, os.path.dirname(os.path.dirname(os.path.dirname(ep)))).split(os.sep)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/ttdu/ttd/data/rlbench_selfgen_v2")
    ap.add_argument("--write", action="store_true", help="write scene_segments.json (default: dry run)")
    ap.add_argument("--verify-masks", action="store_true",
                    help="also open every mask.npz and check each rendered handle is mapped "
                         "(slow: reads the whole tree)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.root, "*", "variation*", "episodes", "episode*")))
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        print(f"no episodes under {args.root}", file=sys.stderr)
        return 1

    union, conflicts = task_handle_union(eps, args.root)
    if conflicts:
        print("HANDLE/NAME CONFLICTS -- the per-task union assumption is broken:", file=sys.stderr)
        for c in conflicts[:20]:
            print(f"  {c}", file=sys.stderr)
        return 2
    filled = sum(len(union[t]) for t in union)
    print(f"per-task handle union: {len(union)} tasks, {filled} distinct (task, handle) pairs")

    unmapped_total = collections.Counter()
    role_counts = collections.Counter()
    n_written = 0
    missing_from_handles = collections.Counter()
    gap_filled = collections.Counter()

    for ep in eps:
        task = ep[len(args.root):].strip(os.sep).split(os.sep)[0]
        hp = os.path.join(ep, "handles.json")
        if not os.path.exists(hp):
            unmapped_total[(task, "<no handles.json>")] += 1
            continue
        with open(hp) as f:
            own = json.load(f)
        # Resolve against the task union, not this episode's own 3-frame sample. See
        # task_handle_union's docstring for why the two differ and why the union is the right one.
        handles = dict(union[task])
        for hid, name in own.items():
            handles[hid] = name
        for hid in set(handles) - set(own):
            gap_filled[(task, handles[hid])] += 1

        instances, unmapped = resolve_roles(task, handles)
        for hid, name in unmapped:
            unmapped_total[(task, name)] += 1
        for inst in instances.values():
            role_counts[inst["base_role"]] += len(inst["handles"])

        if args.verify_masks:
            known = {h for inst in instances.values() for h in inst["handles"]}
            for vd in sorted(glob.glob(os.path.join(ep, "view*"))):
                mp = os.path.join(vd, "mask.npz")
                if not os.path.exists(mp):
                    continue
                for u in np.unique(np.load(mp)["mask"]):
                    u = int(u)
                    if u in NO_OBJECT_HANDLES:
                        continue
                    if u and u not in known and str(u) not in handles:
                        missing_from_handles[(task, u)] += 1

        if args.write:
            doc = {"protocol": "rlbench-scene-role-v1", "task": task, "instances": instances}
            with open(os.path.join(ep, "scene_segments.json"), "w") as f:
                json.dump(doc, f, indent=1, sort_keys=True)
            n_written += 1

    print(f"episodes scanned : {len(eps)}")
    print(f"episodes written : {n_written}{'  (dry run, pass --write)' if not args.write else ''}")
    print("\nhandles per base_role (summed over episodes):")
    for role, n in role_counts.most_common():
        print(f"  {role:12s} {n}")

    if gap_filled:
        print(f"\nhandles recovered from the task union (absent from the episode's own "
              f"handles.json): {sum(gap_filled.values())} across {len(gap_filled)} (task, object)")
        for (task, name), n in gap_filled.most_common(20):
            print(f"  {task:32s} {name:40s} x{n}")

    print(f"\nUNMAPPED object names: {len(unmapped_total)} distinct")
    for (task, name), n in unmapped_total.most_common(30):
        print(f"  {task:32s} {name:40s} x{n}")
    if missing_from_handles:
        print(f"\nrendered handles ABSENT from handles.json: {sum(missing_from_handles.values())}")
        for (task, hid), n in missing_from_handles.most_common(20):
            print(f"  {task:32s} handle {hid:6d} x{n}")

    # Exit non-zero on any coverage gap: a partial annotation must not look like a success.
    # §12.1 requires unknown == 0 before training, and the only way that stays true is if the
    # generator refuses to report OK while gaps exist.
    if unmapped_total or missing_from_handles:
        print("\nCOVERAGE INCOMPLETE")
        return 1
    print("\nSCENE_SEGMENTS_COVERAGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
