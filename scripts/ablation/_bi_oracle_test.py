"""B-i checkpoint: does the SHIPPED GlobalPoseGraph path (incremental adds +
try_commit_correction guard, on-device) reach sub-10cm on fr1_desk when fed
GT-derived loop edges? Run inside the ROCm container.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/workspace/gradslam")
from scipy.spatial import cKDTree
from gradslam.slam.global_pose_graph import GlobalPoseGraph
from gradslam.evaluation.trajectory import load_tum_poses, associate_poses, compute_ate

EST = "/workspace/outputs/control_main__fr1desk/trajectory.txt"
GT = ("/workspace/datasets/public/TUM/tum_rgbd/freiburg1_desk/"
      "rgbd_dataset_freiburg1_desk/groundtruth.txt")

est = load_tum_poses(EST); gt = load_tum_poses(GT)
ets = sorted(est.keys()); E = np.stack([est[t] for t in ets]); N = len(ets)
ate0 = compute_ate(associate_poses(est, gt, max_dt=0.05))
print(f"baseline ATE: {ate0.rmse:.4f} m")

gts = np.array(sorted(gt.keys()))
def gt_at(t):
    i = int(np.abs(gts - t).argmin())
    return gt[gts[i]] if abs(gts[i] - t) < 0.05 else None

K = 10
kf = list(range(0, N, K)); kts = [ets[i] for i in kf]
kpos = np.array([E[i][:3, 3] for i in kf])

# Build graph INCREMENTALLY in keyframe order, injecting loop edges as they would
# be detected (when the current kf revisits an earlier one), each followed by the
# guarded commit — i.e. the shipped online path, not a batch solve.
pg = GlobalPoseGraph(n_iterations=25)
tree = cKDTree(kpos)
loop_for = {}  # current-node -> earliest revisited node
for j in range(len(kf)):
    for i in range(j):
        if np.linalg.norm(kpos[i] - kpos[j]) < 0.10 and abs(kts[i] - kts[j]) > 3.0:
            loop_for.setdefault(j, i)
            break

n_commits = 0
for j, fi in enumerate(kf):
    pg.add_keyframe(torch.tensor(E[fi], dtype=torch.float64), node_id=j)
    if j in loop_for:
        i = loop_for[j]
        Gi, Gj = gt_at(kts[i]), gt_at(kts[j])
        if Gi is None or Gj is None:
            continue
        M = np.linalg.inv(Gi) @ Gj
        if pg.add_loop_edge(i, j, torch.tensor(M, dtype=torch.float64), weight=20.0):
            res = pg.try_commit_correction()
            if res is None:
                pg.drop_last_edge()
            else:
                n_commits += 1
print(f"keyframes={len(kf)} loop_commits={n_commits}")

pg.finalize()  # B-i secondary fix: re-optimize the tail after the last loop commit
corr = {j: pg._poses[j].numpy() for j in range(len(kf))}
out = {}
for fi in range(N):
    kn = min(fi // K, len(kf) - 1)
    T_kf_frame = np.linalg.inv(E[kf[kn]]) @ E[fi]
    out[ets[fi]] = corr[kn] @ T_kf_frame
ate1 = compute_ate(associate_poses(out, gt, max_dt=0.05))
ok = "PASS (<0.10)" if ate1.rmse < 0.10 else "FAIL"
print(f"B-i shipped-path ATE: {ate1.rmse:.4f} m  [{ok}]  {ate0.rmse:.3f}->{ate1.rmse:.3f}")
