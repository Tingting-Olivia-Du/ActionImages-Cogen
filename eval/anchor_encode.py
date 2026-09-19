"""把仿真器的实时观测编码成 arm7 训练时用的同一种锚定帧图像。

四条输入路径的锚定帧全都是 `[V, H, W, 3] uint8`（codec 的全部意义就是让每个模态都活在
RGB 空间里），所以 `policy._build_inputs` 一行不用改 —— 只是喂给它不同的数组。

**必须与训练时逐像素同源**，否则模型拿到的是它没见过的分布，而且不会报错：

| 模态 | 训练时（rlbench_selfgen.py:_encode_perception） | 这里 |
|---|---|---|
| depth | `encode_depth(depth_m)` | 同一个函数，`depth_in_meters=True` 的实时渲染 |
| normal | `encode_normal_from_depth(depth, fx*sx, fy*sy)` | 同一个函数，焦距按下面的理由**不**缩放 |
| segmentation | `encode_scene_roles(mask, role_lut)` | 同一个函数，LUT 由**实时** handle 名解析 |

焦距为什么不缩放：训练时 `_to_model_res` 把原生渲染分辨率的 depth 降采样到模型分辨率，
一个像素覆盖的场景变大，所以焦距（以像素计）要乘 `sx = width / raw_width`。
rollout 直接以模型分辨率渲染，`raw_width == width`，`sx = 1`。
断言而不是假设 —— 这一项错了只会让法向整体倾斜，图看着仍然正常。
"""
from __future__ import annotations

import numpy as np

from training.percep.depth_codec import encode_depth
from training.percep.normal_codec import encode_normal_from_depth
from training.percep.seg_codec import UNKNOWN_LABEL, build_role_lut, encode_scene_roles
from training.percep.scene_segments_gen import resolve_roles

# 模态 -> 该模态的 4 段模板。与 heldout_batch_eval.py 的 X+action 族一致。
ANCHOR_TEMPLATE = {
    "video": "video+action",
    "depth": "depth+action",
    "segmentation": "segmentation+action",
    "normal": "normal+action",
    # The 10-segment fusion canvas. Asked in `rgb_only` conditioning, which is what a rollout
    # actually has: one RGB frame per view, and NOTHING for depth/segmentation/normal -- the
    # simulator would happily render them, but a deployed robot cannot, and a fusion checkpoint
    # scored with them supplied is being flattered by exactly the anchor the F axis exists to
    # remove.
    #
    # Asking a fusion checkpoint through "video+action" instead would be off-distribution: it
    # never trains on a 4-segment canvas, so a bad number there would measure the format
    # mismatch, not the policy.
    "fusion": "video+depth+segmentation+normal+action",
    # Same 10-segment canvas, but asked under `full_anchor`: every modality gets its OWN real
    # anchor frame (rendered live from the simulator's ground truth), not just video. This is
    # off-deployment -- a real robot cannot supply a depth/segmentation/normal frame -- but it
    # answers a different question than `fusion` does: is closed-loop failure an action-
    # prediction problem, or a conditioning-starvation problem that rgb_only creates. See
    # HANDOFF_EVAL.md's four offline regimes; this extends `full_anchor` to closed loop, which
    # previously only had it offline.
    "fusion_full_anchor": "video+depth+segmentation+normal+action",
}

# Which conditioning each anchor is asked under. The 4-segment entries keep i2va (first frame of
# every segment, nothing given), which is what every previous arm was scored with.
ANCHOR_CONDITIONING = {
    "video": None, "depth": None, "segmentation": None, "normal": None,
    "fusion": "rgb_only",
    "fusion_full_anchor": "full_anchor",
}


def encode_depth_views(depth_views: np.ndarray) -> np.ndarray:
    """`[V,H,W]` 米 -> `[V,H,W,3]` uint8，与训练同一个 codec。"""
    if depth_views.ndim != 3:
        raise ValueError(f"expected [V,H,W] metric depth, got {depth_views.shape}")
    return np.stack([encode_depth(depth_views[v][None])[0] for v in range(depth_views.shape[0])])


def encode_normal_views(depth_views: np.ndarray, intr_views: np.ndarray) -> np.ndarray:
    """`[V,H,W]` 米 + `[V,3,3]` 内参 -> `[V,H,W,3]` uint8。

    焦距取绝对值：RLBench 把 fx/fy 写成负数（它的图像 y 轴与针孔约定相反），
    `normal_codec` 内部也取 abs，这里同样取，免得调用方继承这个符号陷阱
    （见 rlbench_selfgen.py:_focal_lengths 的同一条注释）。
    """
    out = []
    for v in range(depth_views.shape[0]):
        fx, fy = abs(float(intr_views[v][0, 0])), abs(float(intr_views[v][1, 1]))
        out.append(encode_normal_from_depth(depth_views[v][None], fx, fy)[0])
    return np.stack(out)


def build_live_role_lut(task: str, mask_views: np.ndarray, handle_names: dict):
    """(task, 实时 handle 名) -> (uint8 LUT, present_roles)。

    与训练时 `_scene_role_lut` 的差别：训练时读磁盘上的 scene_segments.json，
    这里用**当前场景**查出来的 handle 名现算 —— 因为 CoppeliaSim 按加载顺序编号，
    离线的映射对实时场景不成立（rollout_env.handle_names 的注释里有实例）。

    **不做 instruction 相关的 `target` 提升。** 训练时那一步靠 seg_targets.json，
    它是离线从 RLBench 任务源码生成的、按 episode 存的；rollout 时没有对应物。
    后果是实时角色图里本该是 `target` 的物体会保持它的 base_role（多半是 `distractor`）,
    这与训练分布**有出入**，必须在报告里写明，不能当成等价的输入。
    """
    instances, unmapped = resolve_roles(task, handle_names)
    handle_to_role = {}
    for inst in instances.values():
        role = inst.get("base_role")
        if role is None:
            continue
        for h in inst.get("handles", []):
            handle_to_role[int(h)] = role
    if not handle_to_role:
        raise RuntimeError(
            f"task {task!r}: 实时 handle 名解析不出任何角色（查到 {len(handle_names)} 个 handle）。"
            f"未映射: {unmapped[:5]}")
    lut = build_role_lut(handle_to_role)
    present = sorted(set(handle_to_role.values()) | {"background"})
    return lut, present, unmapped


def encode_seg_views(mask_views: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """`[V,H,W]` handle 图 + LUT -> `[V,H,W,3]` uint8 角色图。"""
    return np.stack([encode_scene_roles(mask_views[v][None].astype(np.uint16), lut)[0]
                     for v in range(mask_views.shape[0])])


def unknown_fraction(mask_views: np.ndarray, lut: np.ndarray) -> float:
    """被 LUT 判为 `unknown`（橙色）的像素占比。

    训练数据里这个值恒为 0（仓库的硬不变量，rlbench_selfgen._check_scene_roles_ready 在
    起训时就拦）。实时场景里它 > 0 就说明有 handle 没被映射到角色，模型会看到一个
    训练中**从未出现过**的颜色。这是 seg 这条路唯一会静默出错的地方，所以必须被测量。
    """
    labels = lut[mask_views.astype(np.intp)]
    return float((labels == UNKNOWN_LABEL).mean())
