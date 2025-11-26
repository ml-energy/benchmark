"""Download and prepare the FSD50K audio dataset.

Download the FSD50K dataset audio files using either the HuggingFace mirror
(default, faster) or the original Zenodo source.

The dataset contains development and evaluation audio clips used for the
audio-chat workload. This script downloads only the dev_audio clips (~3GB).

Example:
    # Using HuggingFace mirror (recommended)
    python prepare_fsd50k.py /data/fsd50k

    # Using Zenodo (original source)
    python prepare_fsd50k.py /data/fsd50k --source zenodo
"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "Fhrozen/FSD50k"
ZENODO_DOI = "10.5281/zenodo.4060432"


def download_from_huggingface(dest_root: Path) -> None:
    """Download FSD50K from HuggingFace mirror."""
    print(f"Downloading FSD50K from HuggingFace to {dest_root}...")
    print("If you hit rate limits, consider setting $HF_TOKEN.")

    # Download only the dev clips
    snapshot_dir = Path(
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns="clips/dev/*",
        )
    )

    # Copy the clips directory to destination
    source_clips = snapshot_dir / "clips"
    dest_clips = dest_root / "clips"

    if source_clips.exists():
        print(f"Copying audio clips to {dest_clips}...")
        shutil.copytree(source_clips, dest_clips, dirs_exist_ok=True)

        # Count the files
        dev_files = list((dest_clips / "dev").glob("*.wav"))
        print(
            f"Successfully downloaded {len(dev_files)} audio files to {dest_clips}/dev/"
        )

        # Remove the snapshot directory
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    else:
        raise RuntimeError(f"Expected clips directory not found in {snapshot_dir}")


def download_from_zenodo(dest_root: Path) -> None:
    """Download FSD50K from Zenodo (original source)."""
    print(f"Downloading FSD50K from Zenodo to {dest_root}...")

    # Check if zenodo_get is installed
    if not shutil.which("zenodo_get"):
        raise RuntimeError(
            "zenodo_get is not installed. Install it with:\n"
            "  pip install zenodo-get\n"
            "or use uv run:\n"
            "  uv run --with zenodo-get prepare_fsd50k.py --source zenodo /data/fsd50k"
        )

    # Download using zenodo_get
    print("Downloading archive from Zenodo (this may take a while)...")
    subprocess.run(
        ["zenodo_get", ZENODO_DOI, "-g", "FSD50K.dev_audio.z*"],
        cwd=dest_root,
        check=True,
    )

    # Repair the split zip file
    zip_file = dest_root / "FSD50K.dev_audio.zip"
    if not zip_file.exists():
        raise RuntimeError(f"Expected zip file {zip_file} not found after download")

    print("Repairing split zip archive...")
    full_zip = dest_root / "FSD50K.dev_audio_full.zip"
    subprocess.run(
        ["zip", "-F", str(zip_file), "--out", str(full_zip)],
        check=True,
    )

    # Extract the archive
    print(f"Extracting archive to {dest_root}...")
    with zipfile.ZipFile(full_zip, "r") as zf:
        zf.extractall(dest_root)

    # Count the files
    dev_files = list((dest_root / "FSD50K.dev_audio").glob("*.wav"))
    print(f"Successfully extracted {len(dev_files)} audio files")

    # Clean up intermediate files
    print("Cleaning up temporary files...")
    full_zip.unlink(missing_ok=True)
    print(f"Removed {full_zip}")
    for zip_part in dest_root.glob("FSD50K.dev_audio.z*"):
        zip_part.unlink()
        print(f"Removed {zip_part}")
    for md5_file in dest_root.glob("md5sums.txt"):
        md5_file.unlink()
        print(f"Removed {md5_file}")

    print()
    print(f"export AUDIO_DATA_DIR={dest_root / 'FSD50K.dev_audio'}")


def main():
    ap = argparse.ArgumentParser(
        description="Download FSD50K audio dataset for audio-chat workload"
    )
    ap.add_argument(
        "dest_root",
        help="Destination directory for the dataset",
    )
    ap.add_argument(
        "--source",
        choices=["huggingface", "zenodo"],
        default="zenodo",
        help="Download source (default: zenodo)",
    )
    args = ap.parse_args()

    dest_root = Path(args.dest_root).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        download_from_huggingface(dest_root)
    else:
        download_from_zenodo(dest_root)


if __name__ == "__main__":
    main()
