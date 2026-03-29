#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


VALID_VARIANTS = [
    "notext",
    "correct_answer",
    "misleading_groundable",
    "misleading_ungroundable",
    "irrelevant_word",
]


def iter_ig_jobs(
    ig_root: Path,
    mode: str,
    variants: List[str],
) -> Iterable[Tuple[str, str, Path]]:
    """
    Finds jobs of the form:
      ig_root/<something>/<variant>/<qid>/ig_<mode>.npz
    or
      ig_root/<variant>/<qid>/ig_<mode>.npz

    Returns:
      (variant, qid, npz_path)
    """
    target_name = f"ig_{mode}.npz"

    for npz_path in ig_root.rglob(target_name):
        parts = npz_path.parts
        if len(parts) < 3:
            continue

        qid = npz_path.parent.name
        variant = npz_path.parent.parent.name

        if variant not in variants:
            continue

        yield variant, qid, npz_path


def load_ids_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--ids_file not found: {p}")
    ids = set()
    with p.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


def build_cmd(
    region_script: Path,
    npz_path: Path,
    qid: str,
    variant: str,
    out_dir: Path,
    hf_dataset: str,
    split: str,
    hf_cache_dir: str,
    grid_source: str,
    smooth_sigma: float,
    alpha: float,
    edge_tau: float,
    floor_percentile: float,
    min_region_size: int,
    expand_ratio: float,
    expand_min_pixels: int,
    dilate_iters: int,
    topk_global: int,
) -> List[str]:
    return [
        sys.executable,
        str(region_script),
        "--question_id", str(qid),
        "--npz", str(npz_path),
        "--variant", str(variant),
        "--hf_dataset", str(hf_dataset),
        "--split", str(split),
        "--hf_cache_dir", str(hf_cache_dir),
        "--grid_source", str(grid_source),
        "--smooth_sigma", str(smooth_sigma),
        "--alpha", str(alpha),
        "--edge_tau", str(edge_tau),
        "--floor_percentile", str(floor_percentile),
        "--min_region_size", str(min_region_size),
        "--expand_ratio", str(expand_ratio),
        "--expand_min_pixels", str(expand_min_pixels),
        "--dilate_iters", str(dilate_iters),
        "--topk_global", str(topk_global),
        "--out_dir", str(out_dir),
    ]


def already_done(out_dir: Path, qid: str, variant: str) -> bool:
    """
    Matches the output layout of your region script:
      out_dir/<qid>/<variant>/report.json
    """
    return (out_dir / str(qid) / str(variant) / "report.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the strict-sign region script for all variants/qids found under an IG output root."
    )

    ap.add_argument(
        "--region_script",
        type=str,
        required=True,
        help="Path to the region-determination script (your latest strict-sign script).",
    )
    ap.add_argument(
        "--ig_root",
        type=str,
        required=True,
        help="Root directory containing ig_<mode>.npz files.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output root for the region script.",
    )

    ap.add_argument(
        "--mode",
        type=str,
        default="prefill_next_token",
        choices=["teacher_forced", "prefill_next_token"],
        help="Matches ig_<mode>.npz filename.",
    )

    ap.add_argument(
        "--variants",
        type=str,
        default="notext,correct_answer,misleading_groundable,misleading_ungroundable,irrelevant_word",
        help="Comma-separated variants to process.",
    )
    ap.add_argument(
        "--ids_file",
        type=str,
        default="",
        help="Optional file with qids to keep, one per line.",
    )

    ap.add_argument("--hf_dataset", type=str, default="AHAAM/GUIC")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--hf_cache_dir", type=str, default="./hf_dataset_GUIC")

    ap.add_argument("--grid_source", type=str, default="base", choices=["mosaic", "base"])
    ap.add_argument("--smooth_sigma", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--edge_tau", type=float, default=0.08)
    ap.add_argument("--floor_percentile", type=float, default=65.0)
    ap.add_argument("--min_region_size", type=int, default=3)
    ap.add_argument("--expand_ratio", type=float, default=0.20)
    ap.add_argument("--expand_min_pixels", type=int, default=10)
    ap.add_argument("--dilate_iters", type=int, default=1)
    ap.add_argument("--topk_global", type=int, default=20) ## 50 for llava next and 20 for qwen vl

    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip jobs whose output report.json already exists.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    ap.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one subprocess fails.",
    )

    args = ap.parse_args()

    region_script = Path(args.region_script).resolve()
    ig_root = Path(args.ig_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not region_script.exists():
        raise FileNotFoundError(f"--region_script not found: {region_script}")
    if not ig_root.exists():
        raise FileNotFoundError(f"--ig_root not found: {ig_root}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VALID_VARIANTS]
    if bad:
        raise ValueError(f"Unknown variants in --variants: {bad}")

    keep_ids = load_ids_file(args.ids_file)

    jobs = []
    seen = set()

    for variant, qid, npz_path in iter_ig_jobs(ig_root=ig_root, mode=args.mode, variants=variants):
        if keep_ids is not None and str(qid) not in keep_ids:
            continue

        key = (variant, qid, str(npz_path))
        if key in seen:
            continue
        seen.add(key)

        if args.skip_existing and already_done(out_dir, qid, variant):
            continue

        jobs.append((variant, qid, npz_path))

    print(f"Found {len(jobs)} jobs.")

    n_ok = 0
    n_fail = 0

    for i, (variant, qid, npz_path) in enumerate(jobs, start=1):
        cmd = build_cmd(
            region_script=region_script,
            npz_path=npz_path,
            qid=qid,
            variant=variant,
            out_dir=out_dir,
            hf_dataset=args.hf_dataset,
            split=args.split,
            hf_cache_dir=args.hf_cache_dir,
            grid_source=args.grid_source,
            smooth_sigma=args.smooth_sigma,
            alpha=args.alpha,
            edge_tau=args.edge_tau,
            floor_percentile=args.floor_percentile,
            min_region_size=args.min_region_size,
            expand_ratio=args.expand_ratio,
            expand_min_pixels=args.expand_min_pixels,
            dilate_iters=args.dilate_iters,
            topk_global=args.topk_global,
        )

        print(f"[{i}/{len(jobs)}] variant={variant} qid={qid}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd)

        if result.returncode == 0:
            n_ok += 1
        else:
            n_fail += 1
            print(f"[ERROR] Failed for qid={qid}, variant={variant}, returncode={result.returncode}")
            if args.stop_on_error:
                raise SystemExit(result.returncode)

    print(f"Done. success={n_ok} failed={n_fail} total={len(jobs)}")


if __name__ == "__main__":
    main()