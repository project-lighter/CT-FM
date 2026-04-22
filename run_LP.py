import argparse, os, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from metrics.balanced_accuracy import BalancedAccuracy

import torch
import h5py
import wandb
import numpy as np
from torch.utils.data import WeightedRandomSampler, RandomSampler


from torchmetrics import (
    Accuracy,
    F1Score,
    AUROC,
    AveragePrecision,
    Recall,
    Specificity,
    MetricCollection,
)
import pandas as pd

class FeaturesDataset(Dataset):
    def __init__(self, embeds_dir, csv_path, split, target_column=None):
        # Load CSV and filter by split
        df = pd.read_csv(csv_path)
        split_df = df[df['split'] == split].copy()

        # Build paths and label mapping
        self.paths = []
        self.label_mapping = {}

        for _, row in split_df.iterrows():
            # Extract filename without extension
            case_id = row['case_id']
            filename = case_id.split('.nii.gz')[0] if '.nii.gz' in case_id else case_id
            filename_base = filename.replace('.h5', '')  # Base filename for mapping

            # Construct path with .h5 extension
            h5_filename = filename_base + '.h5'
            path = os.path.join(embeds_dir, h5_filename)

            # Only add if file exists
            if os.path.exists(path):
                self.paths.append(path)
                self.label_mapping[filename_base] = int(row[target_column])
            else:
                print(f"Warning: File not found, skipping: {path}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        # Extract filename without extension from path
        filename = os.path.basename(path).replace('.h5', '')

        # Get label from CSV mapping
        if filename not in self.label_mapping:
            raise ValueError(f"Filename {filename} not found in label mapping")
        y = torch.tensor(self.label_mapping[filename]).long()

        # Load features from h5 file
        with h5py.File(path, 'r') as hf:
            y_hat = torch.tensor(hf['y_hat'][:]).float() # torch.Size([2048])

        return y_hat, y

    def _get_num_classes(self):
        # Get unique labels from the CSV mapping
        all_labels = set(self.label_mapping.values())
        return len(all_labels)


# Removed: get_paths_in_split - now using 'split' column in CSV directly


class AllClassifiers(nn.Module):
    def __init__(self, in_dim, num_classes, lrs):
        super().__init__()
        self.clfs = nn.ModuleDict()
        self.param_groups = []
        for lr in lrs:
            #name = f"clf_lr_{lr:g}".replace(".", "_")
            name = f"clf_lr_{lr:.0e}".replace("-", "_").replace(".", "_")
            m = nn.Linear(in_dim, num_classes)
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.zeros_(m.bias)
            self.clfs[name] = m
            self.param_groups.append({"params": m.parameters(), "lr": lr})
    def forward(self, x): return {k: m(x) for k, m in self.clfs.items()}

def acc_top1(logits, y):  # "top-1"
    return (logits.argmax(1) == y).float().mean().item()

def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    tot_loss, n_batches = 0.0, 0
    tot_samples = 0
    per_clf_losses = {name: 0.0 for name in model.clfs.keys()}
    for xb, yb in loader: # xb: (B, D), yb: (B, 1)
        batch_size = xb.size(0)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = model(xb)
        #import pdb; pdb.set_trace()
        loss_heads = [crit(o, yb) for o in out.values()]
        per_clf_losses = {name: per_clf_losses[name] + l.item() * batch_size for name, l in zip(model.clfs.keys(), loss_heads)}
        loss = torch.stack(loss_heads).sum()  # sum over heads
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * batch_size
        tot_samples += batch_size
        n_batches += 1
    # compute per-clf name average loss
    for name in per_clf_losses.keys():
        per_clf_losses[name] /= tot_samples
    return tot_loss / tot_samples, per_clf_losses


@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes, monitor_metric):
    base_metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=num_classes, top_k=1),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "auroc": AUROC(task="multiclass", num_classes=num_classes),
        "ap": AveragePrecision(task="multiclass", num_classes=num_classes),
        "sensitivity": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        "specificity": Specificity(task="multiclass", num_classes=num_classes, average="macro"),
        "balanced_acc": BalancedAccuracy(num_classes=num_classes, task="multiclass"),
    })

    model.eval()
    names = list(model.clfs.keys())

    metrics = {name: base_metrics.clone().to(device) for name in names}
    losses = {name: 0.0 for name in names}
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        outputs = model(xb)  # {clf_name: logits}
        batch_size = xb.size(0)
        total_samples += batch_size

        for name, logits in outputs.items():
            losses[name] += crit(logits, yb).item() * batch_size
            metrics[name].update(logits, yb)

    for name in names:
        losses[name] /= total_samples

    # compute all metrics per classifier
    computed = {name: metrics[name].compute() for name in names}
    for name in names:
        for k, v in computed[name].items():
            computed[name][k] = v.item()
        computed[name]["loss"] = losses[name]

    # pick best classifier by accuracy
    best_name = max(names, key=lambda n: computed[n][monitor_metric])
    best_acc = computed[best_name][monitor_metric] #.item()

    #import pdb; pdb.set_trace()
    return best_name, best_acc, computed



def save_single_head(model, name, path, results):
    sd = {f"clfs.{name}."+k: v for k, v in model.clfs[name].state_dict().items()}
    torch.save({"state_dict": sd, "head_name": name, "val_metrics": results}, path)

def extract_labels_from_dataset(ds: FeaturesDataset):
    labels = []
    for p in ds.paths:
        fn = os.path.basename(p).replace(".h5", "")
        if fn not in ds.label_mapping:
            raise ValueError(f"{fn} missing from label mapping")
        labels.append(int(ds.label_mapping[fn]))
    return np.asarray(labels, dtype=int)


def compute_sample_weights(ds: FeaturesDataset):
    labels = extract_labels_from_dataset(ds)
    if labels.size == 0:
        raise RuntimeError("Empty label array when computing sample weights.")

    num_classes = int(labels.max()) + 1
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts[class_counts == 0] = 1  # avoid div by zero

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return torch.as_tensor(sample_weights, dtype=torch.float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds_root", default='/path/to/embeddings', help="Root directory containing subdirectories of embeddings for each target")
    ap.add_argument("--embeds_dir", type=str, default=None,
                    help="Direct path to embeddings dir (overrides embeds_root/target/embeddings)")
    ap.add_argument("--labels_root", type=str,
                    default='/path/to/labels',
                    help='Root directory containing CSV files for labels')
    ap.add_argument("--target", type=str, default='splenomegaly',
                    help='target name (used to construct CSV filename: target.csv)')
    ap.add_argument("--out_dir", type=str, default=None, help="Directory to save results (default: embeds_root/target/linear_probe_results)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs",     type=int, default=1000)
    ap.add_argument(
                        "--lrs",
                        type=float,
                        nargs="+",
                        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1], #[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5],
                        help="List of learning rates"
                    )
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--monitor_metric", type=str, default="balanced_acc")
    ap.add_argument("--monitor_metric_mode", type=str, default='max')
    ap.add_argument("--top_k_checkpoints", type=int, default=3)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="linear_probe_cvpr26")
    ap.add_argument("--patience", type=int, default=50,
                help="Early stopping patience based on best head")
    #ap.add_argument("--wandb_run_name", type=str, default=None)
    args = ap.parse_args()
    embeds_dir = args.embeds_dir if args.embeds_dir else os.path.join(args.embeds_root, args.target, "embeddings")

    use_wandb = args.use_wandb
    if use_wandb:
        wandb_dir = os.path.join(
            args.embeds_root,
            args.target,
            "wandb"
        )
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

        wandb_run_name = os.path.basename(embeds_dir)
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
        )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Construct CSV path from labels_root and target
    labels_csv = os.path.join(args.labels_root, args.target + '.csv')
    print(f'Loading labels from: {labels_csv}')

    # Create datasets using 'split' column in CSV
    ds_tr = FeaturesDataset(embeds_dir, labels_csv, split='train', target_column=args.target)
    embed_dim = ds_tr[0][0].shape[0]
    num_classes = ds_tr._get_num_classes()
    print(f'Feature dimension: {embed_dim}, num_classes: {num_classes}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_scaling_range = (0.1, 1.0)

    sample_weights = compute_sample_weights(ds_tr)

    print(
        f"Weight statistics - Before scaling: "
        f"Max: {sample_weights.max().item():.4f}, "
        f"Min: {sample_weights.min().item():.4f}, "
        f"Mean: {sample_weights.mean().item():.4f}"
    )

    if torch.isclose(sample_weights.max(), sample_weights.min()):
        print("Warning: All samples have the same weight. Using uniform sampling instead.")
        sampler = RandomSampler(
            ds_tr,
            replacement=True,
            num_samples=len(ds_tr),
        )
    else:
        min_w, max_w = sample_weights.min(), sample_weights.max()
        min_scale, max_scale = weight_scaling_range

        scaled_sample_weights = min_scale + (max_scale - min_scale) * (
            (sample_weights - min_w) / (max_w - min_w)
        )

        print(
            f"Weight statistics - After scaling: "
            f"Max: {scaled_sample_weights.max().item():.4f}, "
            f"Min: {scaled_sample_weights.min().item():.4f}"
        )

        sampler = WeightedRandomSampler(
            weights=scaled_sample_weights,
            num_samples=len(ds_tr),
            replacement=True,
        )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        sampler=sampler,        # IMPORTANT
        shuffle=False,          # MUST be False when sampler is used
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    ds_va = FeaturesDataset(embeds_dir, labels_csv, split='val', target_column=args.target)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=True)

    lrs = args.lrs
    model = AllClassifiers(embed_dim, num_classes, lrs).to(device)
    opt = torch.optim.AdamW(model.param_groups, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    #current_patience = 0
    monitor_metric = args.monitor_metric
    monitor_metric_mode = args.monitor_metric_mode

    # track best **per classifier**
    init_met_val = -1.0 if monitor_metric_mode == "max" else np.inf
    best_score_per_head = {name: init_met_val for name in model.clfs}
    best_met_per_head   = {name: None for name in model.clfs}
    #best_met_per_head = {name: {} for name in model.clfs.keys()}
    best_met_overall = init_met_val
    best_head_overall = None

    head_names = list(model.clfs.keys())
    num_heads = len(head_names)

    train_losses_over_epochs = {name: [] for name in head_names}
    val_losses_over_epochs = {name: [] for name in head_names}
    monitor_metric_over_epochs = {name: [] for name in head_names}

    patience = args.patience
    epochs_no_improve = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, per_clf_losses = train_one_epoch(model, dl_tr, opt, crit, device)
        best_name, best_met, all_met = evaluate(model, dl_va, crit, device, num_classes, monitor_metric)
        # get balanced_acc, auroc
        best_bal_acc = all_met[best_name]['balanced_acc']
        best_auroc = all_met[best_name]['auroc']
        if use_wandb:
            log_dict = {
                "epoch": ep,
                "train/loss": tr_loss,
                "val/best_head": best_name,
                f"val/{monitor_metric}_best": best_met,
                "val/balanced_acc_best": best_bal_acc,
                "val/auroc_best": best_auroc,
            }

            # per-head metrics
            for name in head_names:
                log_dict[f"train/{name}/loss"] = per_clf_losses[name]
                log_dict[f"val/{name}/loss"] = all_met[name]["loss"]
                log_dict[f"val/{name}/{monitor_metric}"] = all_met[name][monitor_metric]

            wandb.log(log_dict, step=ep)

        # record losses for plotting
        for name in head_names:
            train_losses_over_epochs[name].append(per_clf_losses[name])
            val_losses_over_epochs[name].append(all_met[name]["loss"])
            monitor_metric_over_epochs[name].append(all_met[name][monitor_metric])
            score = all_met[name][monitor_metric]
            improved = (score > best_score_per_head[name]) if monitor_metric_mode == "max" else (score < best_score_per_head[name])
            if improved:
                best_score_per_head[name] = score
                best_met_per_head[name] = dict(all_met[name])  # ensure it's a plain dict

        # for each head, plot train vs val loss & monitor metric in a single figure with multiple subplots
        if ep % args.log_interval == 0 or ep == args.epochs:
            fig, axs = plt.subplots(1, num_heads, figsize=(6 * num_heads, 12))
            if num_heads == 1:
                axs = [axs]
            for i, name in enumerate(head_names):
                axs[i].plot(range(1, ep + 1), train_losses_over_epochs[name], label="Train Loss")
                axs[i].plot(range(1, ep + 1), val_losses_over_epochs[name], label="Val Loss")
                axs[i].plot(range(1, ep + 1), monitor_metric_over_epochs[name], label=f"Val {monitor_metric}")
                axs[i].set_xlabel("Epoch")
                axs[i].set_title(f"Metrics for {name}")
                axs[i].set_ylabel("Value")
                axs[i].legend()
                axs[i].set_ylim(0, 1.5)  # limit y-axis to max=2
                axs[i].set_yticks([round(x, 1) for x in np.arange(0, 2.1, 0.1)])  # grid every 0.1
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"progress.png"))
            plt.close(fig)

        # save best overall classifier and prune checkpoints
        if monitor_metric_mode == 'max' and best_met > best_met_overall:
            best_met_overall = best_met
            best_head_overall = best_name

            # get corresponding results
            results_best = all_met[best_name]
            save_single_head(model, best_name, os.path.join(out_dir, f"best_overall_{monitor_metric}{best_met:.4f}_{best_name}_ep{ep}.pth"), results_best)
            # prune to top-k checkpoints
            saved_ckpts = [f for f in os.listdir(out_dir) if f.startswith(f"best_overall_{monitor_metric}")]
            if len(saved_ckpts) > args.top_k_checkpoints:
                saved_ckpts.sort(key=lambda fn: float(fn.split(monitor_metric)[-1].split('_')[0]), reverse=True)
                for ckpt_to_remove in saved_ckpts[args.top_k_checkpoints:]:
                    os.remove(os.path.join(out_dir, ckpt_to_remove))

            epochs_no_improve = 0
        elif monitor_metric_mode == 'min' and best_met < best_met_overall:
            best_met_overall = best_met
            best_head_overall = best_name
            # get corresponding results
            results_best = all_met[best_name]
            save_single_head(model, best_name, os.path.join(out_dir, f"best_{monitor_metric}{best_met:.4f}_{best_name}_ep{ep}.pth"), results_best)
            # prune to top-k checkpoints
            saved_ckpts = [f for f in os.listdir(out_dir) if f.startswith(f"best_{monitor_metric}")]
            if len(saved_ckpts) > args.top_k_checkpoints:
                # sort by metric value extracted from filename
                saved_ckpts.sort(key=lambda fn: float(fn.split(monitor_metric)[-1].split('_')[0]), reverse=True)
                for ckpt_to_remove in saved_ckpts[args.top_k_checkpoints:]:
                    os.remove(os.path.join(out_dir, ckpt_to_remove))

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"epoch {ep} | train_loss {tr_loss:.4f} | "
              f"val_best {best_name}={best_met:.4f}")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {ep} "
                f"(no improvement in {patience} epochs)")
            break
    # final summary of the overall best head name, monitor_metric value, train & val loss
    print("Overall best classifier:", {
        "name": best_head_overall,
        'val_' + monitor_metric: best_met_overall,
    })
    if use_wandb:
        wandb.finish()

    best_met_per_head_clean = {k: v for k, v in best_met_per_head.items() if v is not None}
    df_report = pd.DataFrame.from_dict(best_met_per_head_clean, orient="index")
    df_report.index = [f"{idx} (best)" if idx == best_head_overall else idx for idx in df_report.index]
    df_report.to_csv(os.path.join(out_dir, "val_report.csv"))

if __name__ == "__main__":
    main()
