"""Download TAT training-set GT data (log/trans/json per scene, from
Google Drive) for benchmark/datasets/tnt.py. Images come separately from
tnt_toolbox/download_t2_dataset.py.

Google Drive rate-limits some files ("too many users..."), per-file not
per-account -- try each individually, skip/log whatever's still
rate-limited (see fixes.md for the full story if this needs retrying).
"""

import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/group/compact-3dmem/datasets/TAT")

# scene -> {file_type: google_drive_id}
FILE_IDS = {
    "Barn": {
        "ply": "1I9d8Vvi3uKESU1Da13nEO9nXdyrgyfS9",
        "log": "19baFLY8jrnPoRp2FX2pkuKZA3gcR6mn9",
        "trans": "1c3F2ZF46BFXzw7b4Kjc0eD5SoW4PhhXg",
        "json": "1ZMeO6gbenmfsb6kxWi-ZM1flOy9wd1-s",
    },
    "Caterpillar": {
        "ply": "18p9DjPj57QtTQ4u9e2WHHw-rUmetaouW",
        "log": "1qCYVKFKwCxOkbLXsvHf-q7u8zoE9Qm-i",
        "trans": "1VT3ZIsg20mZPvWZZHIgdK-Oxgd9BRtsu",
        "json": "16HspBIUg9rP9tSCczYgdUXVYH9QFClzq",
    },
    "Church": {
        "ply": "13IAhuGeL34bXvgvOIXfyhl1bqKwnoqBD",
        "log": "1tzEBpBVxXY-Rtq3qpEOfBmr1QB2nvB3B",
        "trans": "1OZoD-O0FPSDZEHPzH0SUkEawadtx4ofd",
        "json": "19CpnfWHtCUeQLfsKFiQa-a63JGF96fjX",
    },
    "Ignatius": {
        "ply": "1K4TFKLuD-lvJtU6iY-DsxTlp9y85nb6U",
        "log": "172dDxEcJyA6i2Ih3zy3QNWK1R-2sAIi_",
        "trans": "1wSCbCrOT7GsGVLDq0aXs4RHhs4FUzUeM",
        "json": "1_0fESbNxfNI5NWzQ4RBhh460a9_0J54q",
    },
    "Meetingroom": {
        "ply": "1zp9d0k7PAx2ErZYXOKtciTzxOhKehgyb",
        "log": "1iJT9LC4raXaL1oE13-DnacImVl1QLdz0",
        "trans": "1ZQ9g0g0VWrdmihj6PnU-KS3rYZxIf7zH",
        "json": "1Rsc5BPURtVvWhQIfZGQWh6M6QDfl76GW",
    },
    "Truck": {
        "ply": "1BG1A3__pJ0B4vsVKOdHdLXIv7j4PutTf",
        "log": "1uItQhBS5Uiwgdo5xLshHiTJZddX9c3Za",
        "trans": "148ahz6l5c5OV3-VnLJ4sF5gJfLdoIxCt",
        "json": "1syfQomvqAqV-5jKvhT2r6CIPZy9tQd5V",
    },
}

# GDrive "file type" -> destination filename suffix (per tnt.py's docstring)
DEST_SUFFIX = {
    "ply": ".ply",
    "log": "_COLMAP_SfM.log",
    "trans": "_trans.txt",
    "json": ".json",
}


def try_download(scene: str, file_type: str, file_id: str) -> bool:
    scene_dir = OUTPUT_DIR / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    dst = scene_dir / f"{scene}{DEST_SUFFIX[file_type]}"
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[{scene}/{file_type}] already present, skipping.")
        return True

    print(f"[{scene}/{file_type}] downloading ...")
    result = subprocess.run(
        ["gdown", file_id, "-O", str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        print(f"[{scene}/{file_type}] FAILED (likely rate-limited): "
              f"{result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown error'}")
        if dst.exists():
            dst.unlink()
        return False

    print(f"[{scene}/{file_type}] done ({dst.stat().st_size} bytes).")
    return True


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for scene, files in FILE_IDS.items():
        for file_type, file_id in files.items():
            if not try_download(scene, file_type, file_id):
                failed.append(f"{scene}/{file_type}")

    print(f"\nDone. {sum(len(f) for f in FILE_IDS.values()) - len(failed)}/"
          f"{sum(len(f) for f in FILE_IDS.values())} succeeded.")
    if failed:
        print("Still rate-limited / failed (retry later):", failed)
