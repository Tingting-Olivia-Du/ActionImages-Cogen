"""Diagnostic: is the baseline-derived scale wrong, or is VGGT depth itself unreliable?

Per sample the ideal scale is z_true / depth_raw. If those cluster tightly and merely differ
from the baseline-derived scale by a constant, the calibration recipe is fixable. If they
scatter, VGGT depth is not a metric signal here regardless of calibration.
Also takes a robust patch statistic around the gripper pixel, because the gripper is thin and
a 1-pixel miss lands on the table behind it.
"""
import json, os, sys
import numpy as np, torch, cv2
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

os.chdir("/workspace/ttdu/ActionImages")
DEV, N_EP, CROP, VG = sys.argv[1], int(sys.argv[2]), 512, 518
model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir="./checkpoints").to(DEV).eval()

per_sample_scale, base_scale_per_ep, patch_err, ctr_err = [], [], [], []
for eid in sorted(os.listdir("data/droid/processed"))[:N_EP]:
    d = f"data/droid/processed/{eid}"
    cam = json.load(open(f"{d}/camera.json")); act = np.load(f"{d}/action.npz")["action"]
    views = ["ext1", "ext2"]
    E = {v: np.array(cam[v]["extrinsics"], float) for v in views}
    K = {v: np.array(cam[v]["intrinsics"], float) for v in views}
    base_true = np.linalg.norm(E["ext1"][:3, 3] - E["ext2"][:3, 3])
    imgs, Kc = [], {}
    for v in views:
        cap = cv2.VideoCapture(f"{d}/video/{v}.mp4"); ok, fr = cap.read(); cap.release()
        if not ok: break
        H, W = fr.shape[:2]; x0, y0 = (W - CROP)//2, (H - CROP)//2
        imgs.append(cv2.cvtColor(fr[y0:y0+CROP, x0:x0+CROP], cv2.COLOR_BGR2RGB))
        k = K[v].copy(); k[0,2] -= x0; k[1,2] -= y0; Kc[v] = k
    if len(imgs) != 2: continue
    x = torch.from_numpy(np.stack(imgs)).permute(0,3,1,2).float()
    x = torch.nn.functional.interpolate(x, size=(VG,VG), mode="bilinear", align_corners=False)
    x = (x[None]/255.0).to(DEV)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tok, ps = model.aggregator(x); pose = model.camera_head(tok)[-1]
        dmap, _ = model.depth_head(tok, x, ps)
    extr,_ = pose_encoding_to_extri_intri(pose.float(), x.shape[-2:])
    extr = extr.squeeze(0).float().cpu().numpy(); depth = dmap.squeeze(0).float().cpu().numpy()[...,0]
    C = [-extr[i][:3,:3].T @ extr[i][:3,3] for i in range(2)]
    s_base = base_true / float(np.linalg.norm(C[0]-C[1]))
    base_scale_per_ep.append(s_base)
    for vi, v in enumerate(views):
        Ri = E[v][:3,:3].T; ti = -Ri @ E[v][:3,3]
        P = (Ri @ act[:,:3].T).T + ti
        uv = (Kc[v] @ P.T).T; uv = uv[:,:2]/uv[:,2:3]
        for t in range(0, len(act), max(1, len(act)//8)):
            u, vv = uv[t]; zt = P[t,2]
            if not (8 <= u < CROP-8 and 8 <= vv < CROP-8 and zt > 0.1): continue
            gu, gv = int(u*VG/CROP), int(vv*VG/CROP)
            raw_c = float(depth[vi, gv, gu])
            patch = depth[vi, max(0,gv-5):gv+6, max(0,gu-5):gu+6]
            raw_p = float(np.percentile(patch, 10))       # nearest surface in the patch
            if raw_c <= 1e-6: continue
            per_sample_scale.append(zt / raw_c)
            ctr_err.append(raw_c*s_base - zt)
            patch_err.append(raw_p*s_base - zt)

ps_ = np.array(per_sample_scale); bs = np.array(base_scale_per_ep)
ce, pe = np.array(ctr_err), np.array(patch_err)
print(f"samples={len(ps_)}  episodes={len(bs)}")
print(f"baseline-derived scale : mean {bs.mean():.4f}  std {bs.std():.4f}  ({bs.std()/bs.mean()*100:.1f}% CV)")
print(f"ideal per-sample scale : median {np.median(ps_):.4f}  IQR [{np.percentile(ps_,25):.3f},{np.percentile(ps_,75):.3f}]"
      f"  CV {ps_.std()/ps_.mean()*100:.0f}%")
print(f"  -> if VGGT depth were metric-consistent, these two would agree and the IQR would be tight")
print(f"centre-pixel err : median {np.median(np.abs(ce))*100:.1f} cm")
print(f"patch-p10   err : median {np.median(np.abs(pe))*100:.1f} cm   (thin-gripper mitigated)")
