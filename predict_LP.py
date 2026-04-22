"""
Generate per-scan predictions for all diseases using the best LP checkpoints.
Output format (long): filename, label, prediction, logit_class_0, logit_class_1, prob_class_0, prob_class_1, disease_name
"""
import os
import glob
import argparse
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


def load_best_checkpoint(results_dir):
    ckpts = glob.glob(os.path.join(results_dir, "best_overall_balanced_acc*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    ckpts.sort(key=lambda p: float(os.path.basename(p).split("balanced_acc")[1].split("_")[0]),
               reverse=True)
    return ckpts[0]


def load_embeddings(embeds_dir):
    embeddings = {}
    for path in sorted(glob.glob(os.path.join(embeds_dir, "*.h5"))):
        case_id = os.path.basename(path).replace(".h5", "")
        with h5py.File(path, "r") as hf:
            embeddings[case_id] = torch.tensor(hf["y_hat"][:]).float()
    return embeddings


def load_labels(labels_root, disease, label_col=None):
    """Load GT labels for a disease, returns dict {case_id: label}."""
    csv_path = os.path.join(labels_root, f"{disease}.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    col = label_col if label_col else disease
    labels = {}
    for _, row in df.iterrows():
        case_id = str(row["case_id"]).replace(".nii.gz", "").replace(".h5", "")
        labels[case_id] = int(row[col])
    return labels


def predict_disease(embeddings, ckpt_path, labels, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    head_name = ckpt["head_name"]
    sd = ckpt["state_dict"]

    weight = sd[f"clfs.{head_name}.weight"]
    bias = sd[f"clfs.{head_name}.bias"]
    num_classes, in_dim = weight.shape

    clf = nn.Linear(in_dim, num_classes).to(device)
    clf.weight.data = weight.to(device)
    clf.bias.data = bias.to(device)
    clf.eval()

    rows = []
    with torch.no_grad():
        for case_id, emb in embeddings.items():
            logits = clf(emb.unsqueeze(0).to(device)).squeeze(0)
            probs = F.softmax(logits, dim=0).cpu().numpy()
            logits_np = logits.cpu().numpy()
            pred = int(probs.argmax())
            gt = labels.get(case_id, None)

            row = {"filename": case_id, "label": gt, "prediction": pred}
            for c in range(num_classes):
                row[f"logit_class_{c}"] = float(logits_np[c])
            for c in range(num_classes):
                row[f"prob_class_{c}"] = float(probs[c])

            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds_dir", required=True)
    ap.add_argument("--lp_results_dir", required=True)
    ap.add_argument("--labels_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--diseases", nargs="+", default=None)
    ap.add_argument("--label_col", type=str, default=None,
                    help="Column name in labels CSV (defaults to disease name)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.diseases:
        diseases = args.diseases
    else:
        diseases = sorted([
            d for d in os.listdir(args.lp_results_dir)
            if os.path.isdir(os.path.join(args.lp_results_dir, d))
        ])
    print(f"Diseases: {diseases}")

    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeds_dir)
    print(f"Loaded {len(embeddings)} embeddings")

    all_rows = []
    for disease in diseases:
        results_dir = os.path.join(args.lp_results_dir, disease, "results")
        if not os.path.isdir(results_dir):
            print(f"Skipping {disease} — results dir not found")
            continue
        try:
            ckpt_path = load_best_checkpoint(results_dir)
        except FileNotFoundError as e:
            print(f"Skipping {disease}: {e}")
            continue

        print(f"Predicting {disease} using {os.path.basename(ckpt_path)} ...")
        labels = load_labels(args.labels_root, disease, label_col=args.label_col)
        rows = predict_disease(embeddings, ckpt_path, labels, device)
        for row in rows:
            row["disease_name"] = disease
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # Reorder columns
    fixed_cols = ["filename", "label", "prediction"]
    logit_cols = sorted([c for c in df.columns if c.startswith("logit_")])
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_")])
    df = df[fixed_cols + logit_cols + prob_cols + ["disease_name"]]
    df = df.sort_values(["disease_name", "filename"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
