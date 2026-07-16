# #!/usr/bin/env python3
# """
# analyze_winner_csv_v2.py

# Given a CSV with per-(qid,variant,layer) region winner labels (text/corr/mis) computed
# from your attention density_ratio metric, test whether the winner at each layer
# predicts the model's *semantic* MCQ choice.

# Why semantic matters:
# - If MCQ options are shuffled per question, the raw predicted letter (A/B/C/D)
#   is not comparable across samples.
# - The attention NPZ meta stores option_meta['label_to_key'] mapping letters -> option keys.

# We therefore:
# 1) Load NPZ meta for each (variant,qid) to recover predicted_key
# 2) Map winner -> expected_key:
#       corr -> correct_answer
#       mis  -> misleading_groundable
#       text -> <current variant>
# 3) Compute per-layer accuracy: P(predicted_key == expected_key)

# Outputs:
# - Prints per-layer accuracies
# - Saves ONE plot: accuracy vs layer (optionally per variant)

# Expected NPZ structure:
# attn_root/<variant>/<qid>/gen_attn_gen_token.npz
# """

# from __future__ import annotations
# import argparse, json, re
# from pathlib import Path
# from typing import Dict, Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def _first_present_col(df: pd.DataFrame, candidates) -> Optional[str]:
#     for c in candidates:
#         if c in df.columns:
#             return c
#     return None


# def parse_pred_letter(gen_token_text: str) -> Optional[str]:
#     if gen_token_text is None:
#         return None
#     s = str(gen_token_text)
#     m = re.search(r"\b([ABCD])\b", s)
#     if m:
#         return m.group(1)
#     m = re.search(r"([ABCD])", s)
#     return m.group(1) if m else None


# def load_npz_meta(npz_path: Path) -> Dict:
#     with np.load(str(npz_path), allow_pickle=False) as z:
#         meta_raw = z["meta"]
#         if isinstance(meta_raw, np.ndarray):
#             meta_raw = meta_raw.tolist()
#         return json.loads(meta_raw)


# def infer_pred_key_from_meta(meta: Dict) -> Tuple[Optional[str], Optional[str]]:
#     pred_letter = parse_pred_letter(meta.get("generated_token_text"))
#     label_to_key = meta.get("option_meta", {}).get("label_to_key", {})
#     pred_key = label_to_key.get(pred_letter) if pred_letter else None
#     return pred_letter, pred_key


# def expected_key_from_winner(variant: str, winner: str) -> Optional[str]:
#     if winner == "corr":
#         return "correct_answer"
#     if winner == "mis":
#         return "misleading_groundable"
#     if winner == "text":
#         return variant
#     return None


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", default="winners_by_layer.csv", help="Winner CSV (qid,variant,layer,winner)")
#     ap.add_argument("--attn_root", default="llava-next_attentions", help="Root: <variant>/<qid>/gen_attn_gen_token.npz")
#     ap.add_argument("--npz_name", default="gen_attn_gen_token.npz")
#     ap.add_argument("--variant", default="misleading_ungroundable", help="Filter to a single variant or ALL")
#     ap.add_argument("--group_by_variant", action="store_true", help="Plot one line per variant")
#     ap.add_argument("--plot_out", default="winner_predicts_choice.png")
#     ap.add_argument("--title", default=None)
#     args = ap.parse_args()

#     df = pd.read_csv(args.csv)

#     qid_col = _first_present_col(df, ["qid", "question_id", "questionid"])
#     if qid_col is None:
#         raise SystemExit("CSV missing qid/question_id")
#     if qid_col != "qid":
#         df = df.rename(columns={qid_col: "qid"})

#     for req in ["variant", "layer", "winner"]:
#         if req not in df.columns:
#             raise SystemExit(f"CSV missing required column: {req}")

#     if args.variant != "ALL":
#         df = df[df["variant"] == args.variant].copy()

#     attn_root = Path(args.attn_root)

#     # Load pred_key once per (variant,qid)
#     uniq = df[["variant", "qid"]].drop_duplicates().itertuples(index=False, name=None)
#     pred_letter_map: Dict[Tuple[str, str], Optional[str]] = {}
#     pred_key_map: Dict[Tuple[str, str], Optional[str]] = {}

#     missing = 0
#     for variant, qid in uniq:
#         npz_path = attn_root / str(variant) / str(variant) / str(qid) / args.npz_name
#         if not npz_path.exists():
#             missing += 1
#             pred_letter_map[(variant, qid)] = None
#             pred_key_map[(variant, qid)] = None
#             continue
#         meta = load_npz_meta(npz_path)
#         pred_letter, pred_key = infer_pred_key_from_meta(meta)
#         pred_letter_map[(variant, qid)] = pred_letter
#         pred_key_map[(variant, qid)] = pred_key

#     if missing:
#         print(f"WARNING: missing NPZ for {missing} (variant,qid) pairs under {attn_root}")

#     df["pred_letter"] = [pred_letter_map.get((v, q)) for v, q in zip(df["variant"], df["qid"])]
#     df["pred_key"] = [pred_key_map.get((v, q)) for v, q in zip(df["variant"], df["qid"])]

#     df = df[df["pred_key"].notna()].copy()

#     df["expected_key"] = [expected_key_from_winner(v, w) for v, w in zip(df["variant"], df["winner"])]
#     df["match"] = (df["pred_key"] == df["expected_key"]).astype(int)

#     layers = sorted(df["layer"].unique())

#     def curve(d: pd.DataFrame) -> pd.DataFrame:
#         rows = []
#         for L in layers:
#             dd = d[d["layer"] == L]
#             if len(dd) == 0:
#                 continue
#             rows.append({"layer": int(L), "n": int(len(dd)), "acc": float(dd["match"].mean())})
#         return pd.DataFrame(rows)

#     plt.figure(figsize=(12, 6))

#     if args.group_by_variant:
#         for variant, dvar in df.groupby("variant"):
#             c = curve(dvar)
#             if c.empty:
#                 continue
#             plt.plot(c["layer"], c["acc"], marker="o", label=str(variant))
#             # Print majority baseline
#             pk = dvar[["qid", "pred_key"]].drop_duplicates()["pred_key"]
#             print(f"Variant={variant}: majority(pred_key) baseline = {pk.value_counts(normalize=True).max():.3f}")
#     else:
#         print(df.columns)
#         c = curve(df)
#         print(c)
#         plt.plot(c["layer"], c["acc"], marker="o", label="semantic mapping")


#     plt.axhline(0.25, linestyle="--", linewidth=1)  # 4-way chance anchor
#     plt.ylim(0, 1)
#     plt.xlabel("Layer")
#     plt.ylabel("Accuracy: winner@layer -> predicted option key")

#     title = args.title
#     if title is None:
#         title = "Does region-winner at a layer predict the semantic MCQ choice?"
#         if args.variant != "ALL":
#             title += f" (variant={args.variant})"
#     plt.title(title)

#     plt.grid(True, alpha=0.3)
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(args.plot_out, dpi=200)
#     plt.close()

#     print(f"Saved plot to: {Path(args.plot_out).resolve()}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
analyze_winner_csv_v2.py

Given a CSV with per-(qid,variant,layer) region winner labels (text/corr/mis)
computed from your attention density_ratio metric, test whether the winner at
each layer predicts the model's *semantic* MCQ choice.

Semantic target:
- MCQ options can be shuffled, so predicted letter is not stable across samples.
- We recover predicted_key using NPZ meta['option_meta']['label_to_key'] and
  meta['generated_token_text'].

Mapping:
  winner -> expected_key
    corr -> correct_answer
    mis  -> misleading_groundable
    text -> <current variant>   (correct_answer / misleading_groundable / misleading_ungroundable / irrelevant_word)

Metric:
  acc(layer) = P( predicted_key == expected_key(winner@layer) )

Baselines:
  - chance baseline = 0.25
  - majority predicted-key baseline (per variant):
        max_k P(predicted_key = k)
    (computed from the same NPZ-derived predicted_key values)

Expected NPZ structure:
  attn_root/<variant>/<qid>/gen_attn_gen_token.npz
  (some setups may have attn_root/<variant>/<variant>/<qid>/..., we support both)
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _first_present_col(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_pred_letter(gen_token_text: str) -> Optional[str]:
    if gen_token_text is None:
        return None
    s = str(gen_token_text)
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", s)
    return m.group(1) if m else None


def load_npz_meta(npz_path: Path) -> Dict:
    with np.load(str(npz_path), allow_pickle=False) as z:
        meta_raw = z["meta"]
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.tolist()
        return json.loads(meta_raw)


def infer_pred_key_from_meta(meta: Dict) -> Tuple[Optional[str], Optional[str]]:
    pred_letter = parse_pred_letter(meta.get("generated_token_text"))
    label_to_key = meta.get("option_meta", {}).get("label_to_key", {})
    pred_key = label_to_key.get(pred_letter) if pred_letter else None
    return pred_letter, pred_key


def expected_key_from_winner(variant: str, winner: str) -> Optional[str]:
    if winner == "corr":
        return "correct_answer"
    if winner == "mis":
        return "misleading_groundable"
    if winner == "text":
        return variant
    return None


def resolve_npz_path(attn_root: Path, variant: str, qid: str, npz_name: str) -> Optional[Path]:
    """
    Supports both:
      attn_root/<variant>/<qid>/<npz_name>
    and (legacy/buggy layouts):
      attn_root/<variant>/<variant>/<qid>/<npz_name>
    """
    p1 = attn_root / str(variant) / str(qid) / npz_name
    if p1.exists():
        return p1
    p2 = attn_root / str(variant) / str(variant) / str(qid) / npz_name
    if p2.exists():
        return p2
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="winners_by_layer.csv", help="Winner CSV (qid,variant,layer,winner)")
    ap.add_argument("--attn_root", default="llava-next_attentions", help="Root containing <variant>/<qid>/gen_attn_*.npz")
    ap.add_argument("--npz_name", default="gen_attn_gen_token.npz")
    ap.add_argument("--variant", default="misleading_ungroundable", help="Filter to a single variant or ALL")
    ap.add_argument("--group_by_variant", action="store_true", help="Plot one line per variant")
    ap.add_argument("--plot_out", default="winner_predicts_choice.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no_majority_baseline", action="store_true", help="Disable majority-pred_key baseline lines")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    qid_col = _first_present_col(df, ["qid", "question_id", "questionid"])
    if qid_col is None:
        raise SystemExit("CSV missing qid/question_id")
    if qid_col != "qid":
        df = df.rename(columns={qid_col: "qid"})

    for req in ["variant", "layer", "winner"]:
        if req not in df.columns:
            raise SystemExit(f"CSV missing required column: {req}")

    if args.variant != "ALL":
        df = df[df["variant"] == args.variant].copy()

    attn_root = Path(args.attn_root)

    # Load pred_key once per (variant,qid)
    uniq = df[["variant", "qid"]].drop_duplicates().itertuples(index=False, name=None)
    pred_letter_map: Dict[Tuple[str, str], Optional[str]] = {}
    pred_key_map: Dict[Tuple[str, str], Optional[str]] = {}

    missing = 0
    for variant, qid in uniq:
        npz_path = resolve_npz_path(attn_root, str(variant), str(qid), args.npz_name)
        if npz_path is None:
            missing += 1
            pred_letter_map[(variant, qid)] = None
            pred_key_map[(variant, qid)] = None
            continue
        meta = load_npz_meta(npz_path)
        pred_letter, pred_key = infer_pred_key_from_meta(meta)
        pred_letter_map[(variant, qid)] = pred_letter
        pred_key_map[(variant, qid)] = pred_key

    if missing:
        print(f"WARNING: missing NPZ for {missing} (variant,qid) pairs under {attn_root}")

    df["pred_letter"] = [pred_letter_map.get((v, q)) for v, q in zip(df["variant"], df["qid"])]
    df["pred_key"] = [pred_key_map.get((v, q)) for v, q in zip(df["variant"], df["qid"])]

    # Keep only rows where we could infer pred_key
    df = df[df["pred_key"].notna()].copy()

    df["expected_key"] = [expected_key_from_winner(v, w) for v, w in zip(df["variant"], df["winner"])]
    df["match"] = (df["pred_key"] == df["expected_key"]).astype(int)

    layers = sorted(df["layer"].unique())

    def curve(d: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for L in layers:
            dd = d[d["layer"] == L]
            if len(dd) == 0:
                continue
            rows.append({"layer": int(L), "n": int(len(dd)), "acc": float(dd["match"].mean())})
        return pd.DataFrame(rows)

    # ---- Plot ----
    plt.figure(figsize=(12, 6))

    # chance baseline
    plt.axhline(0.25, linestyle="--", linewidth=1, label="chance (0.25)")

    if args.group_by_variant:
        for variant, dvar in df.groupby("variant"):
            c = curve(dvar)
            if c.empty:
                continue
            plt.plot(c["layer"], c["acc"], marker="o", label=str(variant))

            # majority predicted-key baseline (per variant)
            if not args.no_majority_baseline:
                pk = dvar[["qid", "pred_key"]].drop_duplicates()["pred_key"]
                maj = float(pk.value_counts(normalize=True).max()) if len(pk) else float("nan")
                if np.isfinite(maj):
                    plt.axhline(maj, linestyle=":", linewidth=1, alpha=0.9,
                                label=f"majority(pred_key) baseline ({variant}) = {maj:.3f}")
                    print(f"Variant={variant}: majority(pred_key) baseline = {maj:.3f}")

    else:
        c = curve(df)
        if c.empty:
            raise SystemExit("No data left after filtering / NPZ loading. Check paths and CSV.")
        plt.plot(c["layer"], c["acc"], marker="o", label="winner→pred_key accuracy")

        if not args.no_majority_baseline:
            pk = df[["variant", "qid", "pred_key"]].drop_duplicates()["pred_key"]
            maj = float(pk.value_counts(normalize=True).max()) if len(pk) else float("nan")
            if np.isfinite(maj):
                plt.axhline(maj, linestyle=":", linewidth=1, alpha=0.9,
                            label=f"majority(pred_key) baseline = {maj:.3f}")
                print(f"ALL: majority(pred_key) baseline = {maj:.3f}")

    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy: winner@layer -> predicted option key")

    title = args.title
    if title is None:
        title = "Does region-winner at a layer predict the semantic MCQ choice?"
        if args.variant != "ALL":
            title += f" (variant={args.variant})"
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=200)
    plt.close()

    print(f"Saved plot to: {Path(args.plot_out).resolve()}")


if __name__ == "__main__":
    main()