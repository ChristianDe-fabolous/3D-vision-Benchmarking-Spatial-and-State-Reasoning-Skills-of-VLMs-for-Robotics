"""
Build a local instruction-based index mapping DROID episodes to GCS raw data paths.

WHY INSTRUCTION-BASED
---------------------
Robo2VLM-1 entry IDs embed a loop counter (e.g. droid_..._12714_q28) that comes
from enumerating tfds.load("droid", split="train"). TFDS shuffles shards by default
for train splits without a fixed seed, so this ordering is NOT reproducible from
shard structure alone.

Instead, the instruction text is embedded (sanitized) in the entry ID itself and
is directly matchable against the TFDS records. For most entries the instruction
is unique or near-unique, giving an unambiguous mapping. Common instructions may
have multiple matches (reported as ambiguous).

HOW IT WORKS
------------
1. Read all 2048 TFDS shards from gs://gresearch/robotics/droid/1.0.1/
2. For each episode extract: language_instruction + episode_metadata/file_path
3. Build index: sanitized_instruction -> [gcs_file_paths]
   (sanitized the same way robo2vlm does, so entry_id -> instruction is exact)
4. Lookup: given an entry_id, extract its sanitized instruction, match in index.

SANITIZATION (reproduces robo2vlm's create_huggingface_dataset.py)
------------------------------------------------------------------
step1: keep alphanumeric / space / underscore, replace rest with '_', rstrip
step2: re.sub(r'[^\\w\\-_]', '_', step1)   <- spaces become '_'

REQUIREMENTS
------------
    pip install tensorflow
    gcloud auth application-default login

USAGE
-----
    # Build full index (reads all 2048 shards — plan for several hours on GCS)
    python scripts/build_droid_index.py --build-full

    # Resume an interrupted run
    python scripts/build_droid_index.py --build-full --resume

    # Look up one or more entry IDs in the built index
    python scripts/build_droid_index.py --lookup droid_pick_up_the_block_3_q1

    # Look up all DROID entries in a results file
    python scripts/build_droid_index.py --lookup-results outputs/my_run/results.jsonl

OUTPUT FORMAT
-------------
    {
      "__meta__": {
        "processed_shards": [0, 1, 2, ...],
        "total_episodes": 95658
      },
      "remove the black lid from the grey pot on the stove and place it inside the sink": [
        "gs://xembodiment_data/r2d2/r2d2-data-full/ILIAD/success/.../trajectory.h5",
        ...
      ],
      ...
    }

    The key is the RAW instruction text (not sanitized) — sanitization is applied
    at query time for matching. gcs_raw_path is derived by stripping /trajectory.h5.
"""

import argparse
import json
import re
import sys
from pathlib import Path


GCS_DROID_TFDS   = "gs://gresearch/robotics/droid/1.0.1"
SHARD_TEMPLATE   = "droid_101-train.tfrecord-{shard:05d}-of-02048"
TOTAL_SHARDS     = 2048
META_KEY         = "__meta__"


# --------------------------------------------------------------------------- #
# Sanitization — must match robo2vlm's create_huggingface_dataset.py exactly  #
# --------------------------------------------------------------------------- #

def sanitize_instruction(instruction: str) -> str:
    """
    Reproduce robo2vlm's two-step instruction sanitization.

    Step 1 (load_from_tfds.py line 115):
        keep alphanumeric, space, underscore; replace everything else with '_'; rstrip
    Step 2 (create_huggingface_dataset.py line 58):
        re.sub(r'[^\\w\\-_]', '_', traj_id)  — turns spaces into '_'
    """
    step1 = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in instruction).rstrip()
    step2 = re.sub(r"[^\w\-_]", "_", step1)
    return step2


def extract_sanitized_instruction(entry_id: str) -> str | None:
    """
    Extract the sanitized instruction portion from a DROID entry ID.

    entry_id format: droid_{sanitized_instruction}_{loop_index}_q{question_index}
    Returns the sanitized_instruction string, or None if the format doesn't match.
    """
    # strip _q{N} suffix first
    base = re.sub(r"_q\d+$", "", entry_id)
    # then strip droid_ prefix and _{loop_index} suffix
    m = re.match(r"^droid_(.+)_\d+$", base)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Index building                                                               #
# --------------------------------------------------------------------------- #

def build_full_index(
    output_path: Path,
    checkpoint_every: int = 20,
    resume: bool = False,
) -> None:
    """
    Stream all TOTAL_SHARDS TFRecord shards from GCS and build the instruction index.
    Saves a checkpoint every `checkpoint_every` shards.
    """
    import tensorflow as tf

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing index if resuming
    index: dict = {META_KEY: {"processed_shards": [], "total_episodes": 0}}
    if resume and output_path.exists():
        with open(output_path) as f:
            index = json.load(f)
        print(f"Resuming from {output_path}")

    processed: set[int] = set(index.get(META_KEY, {}).get("processed_shards", []))
    total_episodes: int = index.get(META_KEY, {}).get("total_episodes", 0)
    remaining = [s for s in range(TOTAL_SHARDS) if s not in processed]
    print(f"Shards to process: {len(remaining)} / {TOTAL_SHARDS}  "
          f"(already done: {len(processed)}, episodes so far: {total_episodes})")

    errors: list[str] = []

    for shard_num in remaining:
        shard_name = SHARD_TEMPLATE.format(shard=shard_num)
        shard_url  = f"{GCS_DROID_TFDS}/{shard_name}"
        print(f"Shard {shard_num:4d}/{TOTAL_SHARDS-1} ... ", end="", flush=True)

        shard_count = 0
        try:
            ds = tf.data.TFRecordDataset(shard_url, buffer_size=32 * 1024 * 1024)
            for raw in ds:
                try:
                    ctx = tf.io.parse_single_example(
                        raw,
                        features={
                            "episode_metadata/file_path": tf.io.FixedLenFeature(
                                [], tf.string, default_value=b""
                            ),
                            "steps/language_instruction": tf.io.VarLenFeature(tf.string),
                        },
                    )
                    file_path = ctx["episode_metadata/file_path"].numpy().decode("utf-8")
                    lang      = tf.sparse.to_dense(ctx["steps/language_instruction"]).numpy()
                    instruction = lang[0].decode("utf-8") if len(lang) > 0 else ""

                    if instruction and file_path:
                        bucket = index.setdefault(instruction, [])
                        if file_path not in bucket:
                            bucket.append(file_path)

                    shard_count += 1
                except Exception as e:
                    errors.append(f"shard {shard_num} record: {e}")

            total_episodes += shard_count
            index[META_KEY]["processed_shards"].append(shard_num)
            index[META_KEY]["total_episodes"] = total_episodes
            print(f"{shard_count} episodes  (total {total_episodes})")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(f"shard {shard_num}: {e}")
            continue

        # Periodic checkpoint (without pretty-printing to keep I/O fast)
        if len(index[META_KEY]["processed_shards"]) % checkpoint_every == 0:
            with open(output_path, "w") as f:
                json.dump(index, f)
            unique = len(index) - 1
            print(f"  -> checkpoint saved  ({unique} unique instructions)")

    # Final save (pretty-printed)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    unique = len(index) - 1
    print(f"\nDone: {total_episodes} episodes, {unique} unique instructions -> {output_path}")
    if errors:
        print(f"{len(errors)} error(s) (first 10):")
        for e in errors[:10]:
            print(f"  {e}")


# --------------------------------------------------------------------------- #
# Lookup                                                                       #
# --------------------------------------------------------------------------- #

def lookup_entry_ids(entry_ids: list[str], index_path: Path) -> None:
    """Look up DROID entry IDs in the instruction-based index."""
    if not index_path.exists():
        print(f"Index not found: {index_path}")
        print("Build it first with:  python scripts/build_droid_index.py --build-full")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    meta = index.get(META_KEY, {})
    n_shards = len(meta.get("processed_shards", []))
    n_episodes = meta.get("total_episodes", "?")
    n_instructions = len(index) - 1
    print(f"Index: {n_instructions} unique instructions, "
          f"{n_shards}/{TOTAL_SHARDS} shards, {n_episodes} episodes total\n")

    # Pre-build sanitized -> raw instruction map (do once)
    sanitized_to_raw: dict[str, list[str]] = {}
    for raw_instr in index:
        if raw_instr == META_KEY:
            continue
        s = sanitize_instruction(raw_instr)
        sanitized_to_raw.setdefault(s, []).append(raw_instr)

    found = 0
    ambiguous = 0
    missing = 0

    for entry_id in entry_ids:
        if not entry_id.startswith("droid_"):
            print(f"{entry_id}: SKIP — not a DROID entry")
            continue

        sanitized = extract_sanitized_instruction(entry_id)
        if sanitized is None:
            print(f"{entry_id}: cannot parse instruction from ID")
            missing += 1
            continue

        raw_instrs = sanitized_to_raw.get(sanitized, [])

        print(f"{entry_id}")
        if not raw_instrs:
            print(f"  NOT FOUND  (sanitized key: {sanitized!r})")
            missing += 1
        elif len(raw_instrs) == 1:
            raw = raw_instrs[0]
            paths = index[raw]
            gcs_raw_paths = [p.replace("/trajectory.h5", "/") for p in paths]
            if len(paths) == 1:
                print(f"  instruction : {raw!r}")
                print(f"  gcs_raw_path: {gcs_raw_paths[0]}")
                found += 1
            else:
                print(f"  instruction : {raw!r}")
                print(f"  AMBIGUOUS   : {len(paths)} episodes share this instruction:")
                for p in paths:
                    print(f"    {p}")
                ambiguous += 1
        else:
            # Multiple raw instructions map to the same sanitized key (rare)
            print(f"  AMBIGUOUS sanitized key {sanitized!r} matches {len(raw_instrs)} instructions:")
            for r in raw_instrs:
                for p in index[r]:
                    print(f"    [{r!r}] {p}")
            ambiguous += 1
        print()

    total = found + ambiguous + missing
    print(f"Summary: {found}/{total} resolved, {ambiguous} ambiguous, {missing} not found")


def parse_entry_ids_from_results(results_path: Path) -> list[str]:
    """Extract unique DROID entry IDs from a results.jsonl file."""
    ids = set()
    with open(results_path) as f:
        for line in f:
            entry = json.loads(line)
            eid = entry.get("entry_id", "")
            if eid.startswith("droid_"):
                ids.add(eid)
    return sorted(ids)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or query the DROID instruction-based GCS index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build-full", action="store_true",
        help="Stream all 2048 TFDS shards and build the full instruction index."
    )
    group.add_argument(
        "--lookup", nargs="+", metavar="ENTRY_ID",
        help="Look up one or more DROID entry IDs in the built index."
    )
    group.add_argument(
        "--lookup-results", metavar="FILE",
        help="Look up all DROID entry IDs found in a results.jsonl file."
    )

    parser.add_argument(
        "--output", default="dataset_analysis/droid_instruction_index.json",
        metavar="FILE",
        help="Index file path (default: dataset_analysis/droid_instruction_index.json)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted --build-full run."
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=20, metavar="N",
        help="Save checkpoint every N shards during --build-full (default: 20)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.build_full:
        build_full_index(
            output_path=output_path,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )

    elif args.lookup:
        lookup_entry_ids(args.lookup, output_path)

    elif args.lookup_results:
        entry_ids = parse_entry_ids_from_results(Path(args.lookup_results))
        print(f"Found {len(entry_ids)} unique DROID entry IDs in {args.lookup_results}")
        lookup_entry_ids(entry_ids, output_path)


if __name__ == "__main__":
    main()
