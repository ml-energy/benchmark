"""Migrate existing benchmark results to the new num_request_repeats format.

This script transforms results from before the num_request_repeats implementation to
be compatible with the new format. It handles two cases:

1. Regular results: Equivalent to num_request_repeats=1
2. GPQA results with 396 or 594 requests: Equivalent to num_request_repeats=2 or 3
   (since GPQA dataset only has 198 unique requests)

Usage:
    # Dry run (preview changes)
    python scripts/migrate_results_to_repeats.py <results_dir> --dry-run

    # Actually perform migration
    python scripts/migrate_results_to_repeats.py <results_dir>

    # Migrate with backup
    python scripts/migrate_results_to_repeats.py <results_dir> --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def calculate_repeats_and_unique(
    num_prompts: int, endpoint_type: str | None, workload_type: str | None
) -> tuple[int, int]:
    """Calculate num_request_repeats and num_unique_prompts from num_prompts.

    Args:
        num_prompts: Total number of prompts in the result
        endpoint_type: The endpoint type (if available in result)
        workload_type: The workload type inferred from path

    Returns:
        Tuple of (num_request_repeats, num_unique_prompts)
    """
    # GPQA special case: dataset only has 198 unique requests
    if workload_type == "gpqa":
        if num_prompts == 396:
            return 2, 198
        elif num_prompts == 594:
            return 3, 198
        elif num_prompts == 198:
            return 1, 198
        else:
            # Unknown GPQA case, assume no repeats
            print(
                f"  Warning: GPQA result with unexpected num_prompts={num_prompts}, "
                f"assuming num_request_repeats=1"
            )
            return 1, num_prompts

    # All other workloads: no repeats in old results
    return 1, num_prompts


def infer_workload_from_path(result_path: Path) -> str | None:
    """Infer workload type from result file path.

    Args:
        result_path: Path to results.json

    Returns:
        Workload name (e.g., "gpqa", "lm-arena-chat") or None if cannot infer
    """
    # Look for workload name in path components
    # Typical path: run/llm/gpqa/results/model/gpu/params/results.json
    parts = result_path.parts
    known_workloads = {
        "gpqa",
        "lm-arena-chat",
        "sourcegraph-fim",
        "image-chat",
        "video-chat",
        "audio-chat",
        "omni-chat",
        "length-control",
    }

    for part in parts:
        if part in known_workloads:
            return part

    return None


def should_rename_directory(
    result_dir: Path, num_request_repeats: int, num_unique_prompts: int
) -> tuple[bool, Path | None]:
    """Check if directory should be renamed to include num_request_repeats.

    Args:
        result_dir: Current results directory
        num_request_repeats: Number of request repeats
        num_unique_prompts: Number of unique prompts

    Returns:
        Tuple of (should_rename, new_path)
    """
    # Only rename if num_request_repeats > 1
    if num_request_repeats <= 1:
        return False, None

    # Check if path already contains num_request_repeats
    dir_name = result_dir.name
    if f"num_request_repeats+{num_request_repeats}" in dir_name:
        # Already has the correct parameter
        return False, None

    # Need to rename: insert num_request_repeats before num_requests parameter
    # Current: ...+num_requests+396+...
    # New:     ...+num_request_repeats+2+num_requests+198+...

    # Parse directory name into parameters
    parts = dir_name.split("+")
    new_parts = []
    i = 0
    inserted = False

    while i < len(parts):
        # Look for num_requests parameter
        if i + 1 < len(parts) and parts[i] == "num_requests":
            # Insert num_request_repeats before num_requests
            if not inserted:
                new_parts.extend(["num_request_repeats", str(num_request_repeats)])
                inserted = True
            # Update num_requests value to unique count
            new_parts.extend([parts[i], str(num_unique_prompts)])
            i += 2
        else:
            new_parts.append(parts[i])
            i += 1

    new_dir_name = "+".join(new_parts)
    new_path = result_dir.parent / new_dir_name

    return True, new_path


def migrate_result_file(
    result_path: Path, dry_run: bool = False, backup: bool = False
) -> dict[str, Any]:
    """Migrate a single result file to the new format.

    Args:
        result_path: Path to results.json file
        dry_run: If True, only print what would be done
        backup: If True, create backup before modifying

    Returns:
        Dictionary with migration statistics
    """
    stats = {
        "processed": False,
        "modified": False,
        "renamed": False,
        "error": None,
    }

    try:
        # Read existing result
        with open(result_path) as f:
            result = json.load(f)

        # Get current values
        num_prompts = result.get("num_prompts")
        if num_prompts is None:
            stats["error"] = "Missing num_prompts field"
            return stats

        endpoint_type = result.get("endpoint_type")
        workload_type = infer_workload_from_path(result_path)

        # Calculate new values
        num_request_repeats, num_unique_prompts = calculate_repeats_and_unique(
            num_prompts, endpoint_type, workload_type
        )

        # Check if already migrated
        if "num_request_repeats" in result and "num_unique_prompts" in result:
            # Already has new fields, check if they're correct
            if (
                result["num_request_repeats"] == num_request_repeats
                and result["num_unique_prompts"] == num_unique_prompts
            ):
                stats["processed"] = True
                return stats
            else:
                print(
                    "  Warning: File has num_request_repeats/num_unique_prompts but values don't match expected"
                )

        # Prepare modified result
        modified_result = result.copy()
        modified_result["num_request_repeats"] = num_request_repeats
        modified_result["num_unique_prompts"] = num_unique_prompts
        # num_prompts stays the same (total requests sent)

        # Check if directory needs renaming
        result_dir = result_path.parent
        should_rename, new_dir = should_rename_directory(
            result_dir, num_request_repeats, num_unique_prompts
        )

        # Print what will be done
        print(f"  Workload: {workload_type or 'unknown'}")
        print(f"  num_prompts: {num_prompts}")
        print(f"  -> num_request_repeats: {num_request_repeats}")
        print(f"  -> num_unique_prompts: {num_unique_prompts}")

        if should_rename and new_dir:
            print("  Directory rename:")
            print(f"    From: {result_dir.name}")
            print(f"    To:   {new_dir.name}")

        if dry_run:
            print("  [DRY RUN] Would modify result file")
            if should_rename:
                print("  [DRY RUN] Would rename directory")
            stats["processed"] = True
            stats["modified"] = True
            stats["renamed"] = should_rename
            return stats

        # Create backup if requested
        if backup:
            backup_path = result_path.with_suffix(".json.bak")
            shutil.copy2(result_path, backup_path)
            print(f"  Created backup: {backup_path.name}")

        # Write modified result
        with open(result_path, "w") as f:
            json.dump(modified_result, f, indent=2)
        print("  Modified result file")
        stats["modified"] = True

        # Rename directory if needed
        if should_rename and new_dir:
            if new_dir.exists():
                print(
                    f"  Warning: Target directory already exists: {new_dir.name}, skipping rename"
                )
            else:
                shutil.move(str(result_dir), str(new_dir))
                print("  Renamed directory")
                stats["renamed"] = True

        stats["processed"] = True

    except Exception as e:
        stats["error"] = str(e)
        print(f"  Error: {e}")

    return stats


def migrate_results_directory(
    results_base_dir: Path, dry_run: bool = False, backup: bool = False
) -> None:
    """Migrate all result files in a directory tree.

    Args:
        results_base_dir: Base directory containing result files
        dry_run: If True, only print what would be done
        backup: If True, create backups before modifying
    """
    # Find all results.json files
    result_files = list(results_base_dir.rglob("results.json"))

    if not result_files:
        print(f"No results.json files found in {results_base_dir}")
        return

    print(f"Found {len(result_files)} result files")
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60)
    print()

    # Track statistics
    total_processed = 0
    total_modified = 0
    total_renamed = 0
    total_errors = 0

    # Process each file
    for i, result_path in enumerate(result_files, 1):
        print(f"[{i}/{len(result_files)}] {result_path.relative_to(results_base_dir)}")

        stats = migrate_result_file(result_path, dry_run=dry_run, backup=backup)

        if stats["processed"]:
            total_processed += 1
        if stats["modified"]:
            total_modified += 1
        if stats["renamed"]:
            total_renamed += 1
        if stats["error"]:
            total_errors += 1

        print()

    # Print summary
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total files found:       {len(result_files)}")
    print(f"Successfully processed:  {total_processed}")
    print(f"Files modified:          {total_modified}")
    print(f"Directories renamed:     {total_renamed}")
    print(f"Errors:                  {total_errors}")

    if dry_run:
        print()
        print(
            "This was a DRY RUN. To actually perform migration, run without --dry-run"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing benchmark results to num_request_repeats format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Base directory containing result files to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backups before modifying files",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Directory does not exist: {args.results_dir}")
        return 1

    migrate_results_directory(
        args.results_dir, dry_run=args.dry_run, backup=args.backup
    )

    return 0


if __name__ == "__main__":
    exit(main())
