#!/usr/bin/env bash
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Verify the KITTI-360 principal-point-centred crop: rebuild runs of drive 0007 and compare the
# camera-frame motion pitch of lingbot vs GT (was 4.9-5.8 deg apart with the centre crop).
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull kitti360 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
O=/scratch/$USER/k360_ppcenter_verify
"$P" scripts/memory/pose_refine/build_seq_inputs.py kitti360 --seq 2013_05_28_drive_0007_sync --use-sdpa \
    --run-len 1000 --start-every 1000 --out $O
"$P" - $O <<'PY'
import sys, glob, numpy as np, torch
sys.path.insert(0, "scripts/memory/pose_refine")
import train_pose_refine as tp
def pitch(R, t, step=5):
    d = [np.degrees(np.arctan2(*(R[i].T @ (t[i + step] - t[i]))[[1, 2]])) for i in range(len(t) - step)
         if np.linalg.norm(t[i + step] - t[i]) > 0.5]
    return float(np.median(d)) if d else float("nan")
for f in sorted(glob.glob(f"{sys.argv[1]}/specials/*.npz")):
    z = np.load(f); hw = tuple(z["hw"])
    L = tp.pose_enc_to_c2w(torch.from_numpy(z["pose_enc"]).float(), hw).double().numpy()
    G = z["gt_c2w"].astype(np.float64)
    # lingbot is scale-free: rescale its translations to GT by the median step ratio before the 0.5 m gate
    sl = np.median(np.linalg.norm(np.diff(L[:, :3, 3], axis=0), axis=1)); sg = np.median(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1))
    a, b = pitch(G[:, :3, :3], G[:, :3, 3]), pitch(L[:, :3, :3], L[:, :3, 3] * sg / max(sl, 1e-9))
    m = tp.traj_metrics(torch.from_numpy(L), torch.from_numpy(G), 1.0)
    print(f"[verify] {f.split('/')[-1]}: K cx,cy {z['K'][0,0,2]:.1f},{z['K'][0,1,2]:.1f}; motion pitch GT {a:.2f} lingbot {b:.2f} diff {b-a:.2f} deg; "
          f"rel-rot residual {float(z['rel_rot_residual_deg']):.3f} deg/frame; ATE {m['ate']:.2f} m AUC@3 {m['auc3']:.1f}", flush=True)
PY
