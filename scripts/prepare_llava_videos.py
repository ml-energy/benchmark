#!/usr/bin/env python
"""
prepare_llava_videos.py
---------------------------------------------------------
Download every *.tar.gz shard from lmms‑lab/LLaVA‑Video‑178K into the standard
HF cache, figure out the snapshot path automatically, then extract each shard
to <dest_root>.

Example:
    python prepare_llava_videos.py /turbo/llava_video_178k --jobs 12
"""

import argparse, tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import snapshot_download

REPO_ID    = "lmms-lab/LLaVA-Video-178K"
TAR_SUFFIX = ".tar.gz"

def untar(
    tar_path: Path,
    dest_root: Path,
    pbar: tqdm | None = None
) -> bool:
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest_root)
        if pbar:
            pbar.update(1)
        return True
    except Exception as e:
        print(f"Failed on {tar_path.name}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest_root", help="where the reconstructed tree is written")
    ap.add_argument("--jobs", "-j", type=int, default=8,
                    help="parallel extraction workers (default 8)")
    args = ap.parse_args()

    dest_root = Path(args.dest_root).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Download all tarballs into the HF cache and get the snapshot path
    # ---------------------------------------------------------------------
    snapshot_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="*.tar.gz",
        )
    )
    # ---------------------------------------------------------------------
    # Extract each tar.gz in parallel, then delete it
    # ---------------------------------------------------------------------
    tar_files = list(snapshot_dir.rglob(f"*{TAR_SUFFIX}"))
    if not tar_files:
        raise SystemExit("No tarballs found – repo structure may have changed")
    pbar = tqdm(total=len(tar_files), desc="Extracting shards", unit="shard")

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        tasks = {ex.submit(untar, t, dest_root, pbar): t for t in tar_files}
        done  = 0
        for fut in as_completed(tasks):
            done += fut.result()

    print(f"Extracted {done}/{len(tar_files)} shards into {dest_root}")

if __name__ == "__main__":
    main()

