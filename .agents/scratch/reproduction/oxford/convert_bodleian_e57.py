"""Convert bodleian-library's TLS point cloud from .e57 to .pcd, matching
the filename preprocess/oxford.py's get_tls_pcd_path() already expects
(merged-cloud-1cm.pcd) -- so no code changes needed once this runs.

bodleian-library is the only Oxford Spires site shipped as .e57 rather
than .pcd on HuggingFace (open3d doesn't support .e57 at all -- confirmed
"unknown file extension"). Uses pye57 (wraps libE57Format) to read, then
open3d to write.
"""

import numpy as np
import open3d as o3d
import pye57

SRC = "/group/compact-3dmem/datasets/oxford_spires/ground_truth_map/bodleian-library/merged-cloud-1cm.e57"
DST = "/group/compact-3dmem/datasets/oxford_spires/ground_truth_map/bodleian-library/merged-cloud-1cm.pcd"

print(f"Reading {SRC} ...")
e57 = pye57.E57(SRC)
header = e57.get_header(0)
print(f"Point count: {header.point_count}, fields: {header.point_fields}")

data = e57.read_scan(0, ignore_missing_fields=True)

xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], axis=-1).astype(np.float64)
print(f"Loaded {xyz.shape[0]} points")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

if all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
    colors = np.stack([data["colorRed"], data["colorGreen"], data["colorBlue"]], axis=-1)
    # pye57 colors are typically 0-255 uint8 range even if dtype is float
    colors = (colors.astype(np.float64) / 255.0).clip(0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("Colors attached.")

print(f"Writing {DST} ...")
o3d.io.write_point_cloud(DST, pcd, write_ascii=False, compressed=False)
print("Done.")
