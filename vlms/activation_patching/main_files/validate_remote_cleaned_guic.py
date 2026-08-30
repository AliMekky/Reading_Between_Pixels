#!/usr/bin/env python3
"""Download and pixel-validate the committed cleaned GUIC dataset."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from build_and_push_cleaned_guic import pixel_validation


def log(section: str, message: str) -> None:
    print("[{}] {}".format(section, message), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="AHAAM/GUIC")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--revision",
        default="27b45899d1154ef1f08ce5c40d45d2468e4ea3e2",
    )
    parser.add_argument(
        "--cache_dir",
        default="../hf_remote_validation_cache/27b45899",
    )
    parser.add_argument(
        "--output",
        default="../hf_dataset_GUIC_cleaned/remote_validation.json",
    )
    args = parser.parse_args()

    info = HfApi().dataset_info(args.repo_id, revision=args.revision)
    log("REMOTE", "requested_repo={}".format(args.repo_id))
    log("REMOTE", "canonical_repo={}".format(info.id))
    log("REMOTE", "commit_sha={}".format(info.sha))
    if info.sha != args.revision:
        raise RuntimeError(
            "Remote revision mismatch: expected {}, received {}".format(
                args.revision, info.sha
            )
        )

    cache_dir = Path(args.cache_dir).resolve()
    log("DOWNLOAD", "split={} cache={}".format(args.split, cache_dir))
    dataset = load_dataset(
        args.repo_id,
        split=args.split,
        revision=args.revision,
        cache_dir=str(cache_dir),
        download_mode="force_redownload",
    )
    log("DOWNLOAD", "remote_rows={}".format(len(dataset)))

    validation = pixel_validation(dataset)
    validation.update(
        {
            "requested_repo": args.repo_id,
            "canonical_repo": info.id,
            "commit_sha": info.sha,
            "split": args.split,
            "source": "remote_huggingface_commit",
        }
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    log("RESULT", json.dumps(validation, sort_keys=True))
    log("OUTPUT", str(output))
    if not validation["passed"]:
        raise RuntimeError("Remote pixel validation failed")
    log("PASS", "All remote cleaned images passed pixel-level validation")


if __name__ == "__main__":
    main()
