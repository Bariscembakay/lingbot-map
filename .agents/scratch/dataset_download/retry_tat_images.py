"""Retry-loop downloader for TAT training-set image zips, working around
Google Drive's per-file view-quota wall ("too many users have viewed or
downloaded this file recently... may take up to 24 hours").

These are legacy (~2018) Drive file IDs needing a `resourcekey` query
param to resolve (Google's 2021 security change for old shared links) --
gdown (any version up to 6.1.0, and 4.7.1's --fuzzy) doesn't handle this
correctly, so this implements the browser's own confirm-token flow by
hand: GET the "virus scan warning" interstitial, extract the `uuid` from
its confirm form, then GET the real file from
drive.usercontent.google.com with confirm=t&uuid=<uuid>. If Google's
quota message appears instead, back off and retry later.

Usage: run once, it loops (with a long sleep between passes) until every
scene succeeds or `--max-hours` elapses.
"""

import argparse
import re
import time
from pathlib import Path

import requests

OUTPUT_DIR = Path("/group/compact-3dmem/datasets/TAT_images_incoming")

# scene -> (file_id, resourcekey), from tanksandtemples.org/download's
# current Training Data image-set links.
IMAGE_ZIPS = {
    "Barn": ("0B-ePgl6HF260NzQySklGdXZyQzA", "0-luQ7Jaym5BQL6IjxsgXY9A"),
    "Caterpillar": ("0B-ePgl6HF260b2JNbnZYYjczU2s", "0-pO8ilXjCCCEkSTVuePIe5g"),
    "Church": ("0B-ePgl6HF260SmhXM0czaHJ3SU0", "0-Gsw54gkOI-Crg5S-9wJZpg"),
    "Ignatius": ("0B-ePgl6HF260d0l0ZDNSZ3ZxREk", "0-DoBcm2nIvBpxqMUPgmcmXQ"),
    "Meetingroom": ("0B-ePgl6HF260cV9lNmlZZGp6aUU", "0-AvrSVlLY3Q6HP3oVVzSvsw"),
    "Truck": ("0B-ePgl6HF260NEw3OGN4ckF0dnM", "0-uYzL1Ga_EW1Ck0o-msT7Sg"),
}

UUID_RE = re.compile(r'name="uuid" value="([a-f0-9-]{36})"')


def try_download(scene: str, file_id: str, resourcekey: str) -> str:
    """Returns 'done', 'quota', or 'error'."""
    dst = OUTPUT_DIR / f"{scene}_images.zip"
    if dst.exists() and dst.stat().st_size > 10_000_000:  # sanity: real zips are big
        return "done"

    session = requests.Session()
    r1 = session.get(
        "https://drive.google.com/uc",
        params={"id": file_id, "resourcekey": resourcekey, "export": "download"},
        timeout=60,
    )
    if "Quota exceeded" in r1.text or "cannot view or download" in r1.text:
        return "quota"

    m = UUID_RE.search(r1.text)
    if not m:
        # Small file with no interstitial -- content-type check.
        if r1.headers.get("content-type", "").startswith("application/zip"):
            dst.write_bytes(r1.content)
            return "done"
        return "error"

    r2 = session.get(
        "https://drive.usercontent.google.com/download",
        params={"id": file_id, "resourcekey": resourcekey, "export": "download",
                "confirm": "t", "uuid": m.group(1)},
        timeout=300,
    )
    if "Quota exceeded" in r2.text[:2000] if len(r2.content) < 100_000 else False:
        return "quota"

    dst.write_bytes(r2.content)
    if dst.stat().st_size < 10_000_000:
        # Got another HTML error page, not a real zip.
        dst.unlink()
        return "quota"
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=26.0)
    ap.add_argument("--retry-interval-min", type=float, default=30.0)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    remaining = dict(IMAGE_ZIPS)

    attempt = 0
    while remaining and time.time() < deadline:
        attempt += 1
        print(f"\n=== Attempt {attempt} ===", flush=True)
        for scene in list(remaining):
            file_id, key = remaining[scene]
            try:
                status = try_download(scene, file_id, key)
            except Exception as e:
                print(f"[{scene}] error: {e}", flush=True)
                status = "error"

            if status == "done":
                print(f"[{scene}] SUCCESS", flush=True)
                del remaining[scene]
            elif status == "quota":
                print(f"[{scene}] still quota-limited", flush=True)
            else:
                print(f"[{scene}] unexpected error, will retry", flush=True)

        if remaining:
            print(f"{len(remaining)} scene(s) remaining: {list(remaining)}. "
                  f"Sleeping {args.retry_interval_min} min ...", flush=True)
            time.sleep(args.retry_interval_min * 60)

    if remaining:
        print(f"\nGave up after {args.max_hours}h. Still missing: {list(remaining)}")
    else:
        print("\nAll TAT image zips downloaded.")


if __name__ == "__main__":
    main()
