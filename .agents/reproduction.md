# Reproduction results — lingbot-map

Paper vs. our runs. Local-only. Run outputs:
`/group/compact-3dmem/campaigns/lingbot_map/<dataset>/default/`.

## Table 2 — Oxford Spires (sparse, stride 12)

| Metric | Paper | Ours |
| --- | --- | --- |
| AUC@15 | 61.64 | 63.50 |
| AUC@30 | 75.16 | 76.50 |
| ATE | 6.42 | 6.19 |
| RPE-trans | 1.01 | 0.765 |
| RPE-Rot | 3.70 | 4.29 |

## Table 4 — Pose (AUC@3 / AUC@30 / ATE)

| Dataset | Paper | Ours (macro / micro) |
| --- | --- | --- |
| ETH3D | 37.22 / 81.09 / 0.43 | 36.98 / 81.01 / 0.442  ·  40.04 / 87.90 |
| 7-Scenes | 12.63 / 78.59 / 0.08 | 12.33 / 78.07 / 0.0789  ·  13.18 / 79.05 |
| Tanks & Temples | 45.80 / 92.80 / 0.20 | 47.99 / 93.17 / 0.2095  ·  44.87 / 92.33 |

## Table 5 — Reconstruction (Acc / Comp / F1)

| Dataset | Paper | Ours |
| --- | --- | --- |
| ETH3D | 0.16 / 0.08 / 86.79 | 0.168 / 0.090 / 86.72 |
| 7-Scenes | 0.02 / 0.07 / 80.39 | 0.0347 / 0.0431 / 81.78 — **not reproduced**, see `fixes.md` |
| NRGBD | 0.07 / 0.03 / 64.26 | 0.0737 / 0.0303 / 65.11 |

## Not paper benchmarks

- DROID-W: no baseline in the paper, pipeline-only test.
- VBR: no baseline in the paper, pipeline-only test. ATE 31.12, RPE-trans
  2.72, RPE-Rot 4.37 (7 scenes) — pipeline runs clean end to end.
- KITTI (Odometry): not in the paper either (repo calls it an ablation-only
  dataset). Ran on msp3 (11 GT scenes): ATE 24.07, RPE-trans 2.855,
  RPE-rot 0.856.
- TUM: no baseline in the paper either (no documented subset, downloaded
  full public benchmark). Ran on msp3 (66 scenes): ATE 0.1303,
  RPE-trans 0.0274, RPE-rot 0.919.

## Checklist

- [x] Oxford Spires (Table 2)
- [x] ETH3D (Tables 4/5)
- [x] Tanks & Temples (Table 4)
- [x] NRGBD (Table 5)
- [ ] 7-Scenes reconstruction gap — pose matches, Acc/Comp don't; root
      cause unresolved (see `fixes.md`)
- [x] VBR — pipeline validated, no paper baseline
- [x] TUM — ran on msp3, no paper baseline
- [x] KITTI (Odometry) — ran on msp3, no paper baseline
- [ ] Decide whether to PR bugs found this round upstream
