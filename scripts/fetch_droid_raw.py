"""
Fetch or stream raw DROID trajectory data for a given Robo2VLM-1 dataset entry ID.

Requires the instruction-based index built by build_droid_index.py, which maps
language instructions (embedded in entry IDs) to GCS file paths.

HOW IT WORKS
------------
1. Extract the sanitized instruction from the entry ID (strip droid_ prefix,
   strip _{loop_index}_q{n} suffix).
2. Look it up in the instruction index (build_droid_index.py --build-full).
3. Derive the GCS episode folder from the matched file_path.
4. Download or stream the raw data via gsutil.

The raw DROID episode folder on GCS contains:
    trajectory.h5              — robot states and actions, ~1–5 MB
    {metadata}.json            — episode metadata
    recordings/
        MP4/  {serial}-stereo.mp4  — pre-rendered stereo videos, ~20–50 MB each
        SVO/  {serial}.svo         — raw ZED recordings, ~500 MB–2 GB each

ESTIMATED DOWNLOAD SIZE PER EPISODE
-------------------------------------
    MP4s + trajectory.h5 only:  ~60–150 MB   (default)
    Full incl. SVO:             ~1.5–6 GB

STREAMING
---------
    --stream pipes an MP4 directly from GCS into ffplay (requires ffmpeg).
    No disk space used.

REQUIREMENTS
------------
    gsutil  — https://cloud.google.com/sdk/docs/install
    ffplay  — only for --stream (part of ffmpeg)
    Index   — build with: python scripts/build_droid_index.py --build-full

USAGE
-----
    # Download MP4s + trajectory.h5 for an entry
    python scripts/fetch_droid_raw.py --id droid_pick_up_the_block_3_q1

    # Check size before downloading
    python scripts/fetch_droid_raw.py --id droid_pick_up_the_block_3_q1 --size-only

    # Stream first camera video directly (no download, requires ffplay)
    python scripts/fetch_droid_raw.py --id droid_pick_up_the_block_3_q1 --stream

    # Download everything including large SVO files
    python scripts/fetch_droid_raw.py --id droid_pick_up_the_block_3_q1 --include-svo
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


META_KEY = "__meta__"


# --------------------------------------------------------------------------- #
# Sanitization (must mirror build_droid_index.py)                             #
# --------------------------------------------------------------------------- #

def sanitize_instruction(instruction: str) -> str:
    step1 = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in instruction).rstrip()
    return re.sub(r"[^\w\-_]", "_", step1)


def extract_sanitized_instruction(entry_id: str) -> str | None:
    base = re.sub(r"_q\d+$", "", entry_id)
    m = re.match(r"^droid_(.+)_\d+$", base)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Index loading and lookup                                                     #
# --------------------------------------------------------------------------- #

def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Build it first with:\n"
            "  python scripts/build_droid_index.py --build-full"
        )
    with open(index_path) as f:
        return json.load(f)


def resolve_gcs_paths(entry_id: str, index: dict) -> list[str]:
    """
    Return a list of GCS file paths matching the instruction in this entry_id.

    Raises KeyError if no match is found.
    Returns multiple paths if the instruction is shared by several episodes.
    """
    sanitized = extract_sanitized_instruction(entry_id)
    if sanitized is None:
        raise KeyError(f"Cannot parse instruction from entry ID: {entry_id!r}")

    # Build sanitized -> raw_instruction lookup on the fly
    # (index is usually loaded once per script run, so this is fine)
    matches: list[str] = []
    matched_instrs: list[str] = []
    for raw_instr, paths in index.items():
        if raw_instr == META_KEY:
            continue
        if sanitize_instruction(raw_instr) == sanitized:
            matches.extend(paths)
            matched_instrs.append(raw_instr)

    if not matches:
        raise KeyError(
            f"Instruction {sanitized!r} not found in index.\n"
            f"If the index is incomplete, rebuild with:\n"
            f"  python scripts/build_droid_index.py --build-full --resume"
        )

    return matches, matched_instrs


# --------------------------------------------------------------------------- #
# GCS operations                                                               #
# --------------------------------------------------------------------------- #

def get_size(gcs_episode_path: str) -> str:
    result = subprocess.run(
        ["gsutil", "du", "-sh", f"{gcs_episode_path}/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"(size unknown: {result.stderr.strip()})"
    return result.stdout.split()[0] if result.stdout else "(size unknown)"


def list_gcs_mp4s(gcs_episode_path: str) -> list[str]:
    result = subprocess.run(
        ["gsutil", "ls", f"{gcs_episode_path}/recordings/MP4/"],
        capture_output=True, text=True,
    )
    return [l.strip() for l in result.stdout.splitlines() if l.strip().endswith(".mp4")]


def download(gcs_episode_path: str, local_dir: Path, include_svo: bool, dry_run: bool) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)

    if include_svo:
        cmds = [["gsutil", "-m", "cp", "-r", f"{gcs_episode_path}/", str(local_dir)]]
    else:
        cmds = [
            ["gsutil", "-m", "cp", f"{gcs_episode_path}/trajectory.h5", str(local_dir)],
            ["gsutil", "-m", "cp", "-r", f"{gcs_episode_path}/recordings/MP4/", str(local_dir)],
            ["gsutil", "-m", "cp", f"{gcs_episode_path}/*.json", str(local_dir)],
        ]

    for cmd in cmds:
        print(f"  {'[DRY RUN] ' if dry_run else ''}{' '.join(cmd)}")
        if not dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise RuntimeError(
                    f"gsutil failed (return code {result.returncode}): {' '.join(cmd)}"
                )


def stream_mp4(gcs_mp4_path: str) -> None:
    print(f"  Streaming: {gcs_mp4_path}")
    gsutil = subprocess.Popen(["gsutil", "cat", gcs_mp4_path], stdout=subprocess.PIPE)
    ffplay  = subprocess.Popen(["ffplay", "-autoexit", "-"], stdin=gsutil.stdout)
    gsutil.stdout.close()
    ffplay.wait()
    gsutil.wait()


# --------------------------------------------------------------------------- #
# Main fetch logic                                                             #
# --------------------------------------------------------------------------- #

def fetch(
    entry_id: str,
    index: dict,
    output_dir: Path,
    include_svo: bool,
    size_only: bool,
    do_stream: bool,
    dry_run: bool,
    pick: int | None,
) -> None:
    if not entry_id.startswith("droid_"):
        print(f"  SKIP {entry_id}: not a DROID entry.")
        print("  OXE entries cannot be traced back to raw data — see README Dataset Notes.")
        return

    print(f"\n--- {entry_id} ---")

    gcs_file_paths, matched_instrs = resolve_gcs_paths(entry_id, index)

    if len(matched_instrs) == 1:
        print(f"  instruction: {matched_instrs[0]}")
    else:
        print(f"  instruction (multiple variants): {matched_instrs}")

    if len(gcs_file_paths) > 1:
        print(f"  WARNING: {len(gcs_file_paths)} episodes share this instruction.")
        for i, p in enumerate(gcs_file_paths):
            print(f"    [{i}] {p}")
        if pick is None:
            print("  Use --pick N to select one, or proceed to operate on ALL of them.")
        else:
            gcs_file_paths = [gcs_file_paths[pick]]
            print(f"  Using episode [{pick}]: {gcs_file_paths[0]}")

    for gcs_file_path in gcs_file_paths:
        gcs_path = gcs_file_path.replace("/trajectory.h5", "")
        print(f"\n  GCS path: {gcs_path}")

        if size_only:
            print(f"  total size: {get_size(gcs_path)}")
            continue

        if do_stream:
            mp4s = list_gcs_mp4s(gcs_path)
            if not mp4s:
                print("  No MP4 files found — may be SVO-only recording (requires ZED SDK).")
                continue
            print(f"  Available cameras ({len(mp4s)}): {[p.split('/')[-1] for p in mp4s]}")
            print(f"  Streaming: {mp4s[0].split('/')[-1]}")
            stream_mp4(mp4s[0])
            continue

        # Build local directory name from entry_id (strip _q{n} suffix)
        episode_dir = re.sub(r"_q\d+$", "", entry_id)
        local_dir = output_dir / episode_dir
        download(gcs_path, local_dir, include_svo, dry_run)

        if not dry_run:
            mp4s = sorted(local_dir.rglob("*.mp4"))
            if mp4s:
                print(f"\n  Videos ({len(mp4s)}):")
                for v in mp4s:
                    print(f"    {v}")
            else:
                print("  No MP4 files found (SVO-only recording — requires ZED SDK).")
            print(f"  -> raw data at {local_dir}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch or stream raw DROID data for a Robo2VLM-1 entry ID."
    )
    parser.add_argument(
        "--id", nargs="+", required=True, metavar="ENTRY_ID",
        help="One or more entry IDs (e.g. droid_pick_up_the_block_3_q1)"
    )
    parser.add_argument(
        "--index", default="dataset_analysis/droid_instruction_index.json", metavar="FILE",
        help="Path to instruction index built by build_droid_index.py "
             "(default: dataset_analysis/droid_instruction_index.json)"
    )
    parser.add_argument(
        "--output", default="./droid_raw", metavar="DIR",
        help="Local directory for downloaded data (default: ./droid_raw)"
    )
    parser.add_argument(
        "--include-svo", action="store_true",
        help="Also download raw SVO files (~500 MB–2 GB per camera). Excluded by default."
    )
    parser.add_argument(
        "--size-only", action="store_true",
        help="Show GCS episode size without downloading"
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream first camera MP4 via ffplay without downloading (requires ffmpeg)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print gsutil commands without executing them"
    )
    parser.add_argument(
        "--pick", type=int, default=None, metavar="N",
        help="When multiple episodes match (ambiguous instruction), pick episode #N (0-based)"
    )
    args = parser.parse_args()

    index = load_index(Path(args.index))
    meta = index.get(META_KEY, {})
    n_shards = len(meta.get("processed_shards", []))
    print(f"Loaded index: {len(index)-1} instructions, "
          f"{n_shards}/{2048} shards, {meta.get('total_episodes', '?')} episodes")

    errors = []
    for entry_id in args.id:
        try:
            fetch(
                entry_id, index, Path(args.output),
                include_svo=args.include_svo,
                size_only=args.size_only,
                do_stream=args.stream,
                dry_run=args.dry_run,
                pick=args.pick,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((entry_id, e))

    if errors:
        print(f"\n{len(errors)} error(s):")
        for eid, e in errors:
            print(f"  {eid}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
