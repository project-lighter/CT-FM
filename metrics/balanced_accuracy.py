import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores

import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores
from typing import Optional


class MultiLabelBalancedAccuracy(Metric):
    def __init__(self, num_labels, threshold=0.5):
        super().__init__()
        self.num_labels = num_labels
        self.threshold = threshold
        self.add_state("tp", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_labels), dist_reduce_fx="sum")

    def update(self, probs, target):
        preds = (probs >= self.threshold).long()
        target = target.long()

        for i in range(self.num_labels):
            mask = target[:, i] != -1
            if mask.sum() == 0:
                continue

            p = preds[mask, i]
            t = target[mask, i]

            self.tp[i] += ((p == 1) & (t == 1)).sum()
            self.fp[i] += ((p == 1) & (t == 0)).sum()
            self.tn[i] += ((p == 0) & (t == 0)).sum()
            self.fn[i] += ((p == 0) & (t == 1)).sum()

    def compute(self):
        recall = self.tp / (self.tp + self.fn + 1e-8)
        spec = self.tn / (self.tn + self.fp + 1e-8)
        return ((recall + spec) / 2).mean()

class BalancedAccuracy(Metric):
    def __init__(
        self,
        task: str = "multiclass",
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        assert task in {
            "binary",
            "multiclass",
            "multilabel",
        }, "Only 'binary', 'multiclass', and 'multilabel' tasks are supported."
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task = task
        self.threshold = threshold
        self.ignore_index = ignore_index

        # Determine the number of state elements needed
        if task == "binary":
            num_state_elements = 1
            if num_classes is not None and num_classes != 2:
                pass
            self.num_classes = 2
        elif task == "multiclass":
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError(f"`num_classes` must be an integer >= 2 for task '{task}'.")
            num_state_elements = num_classes
            self.num_classes = num_classes
        elif task == "multilabel":
            if not isinstance(num_classes, int) or num_classes < 1:
                 raise ValueError(f"`num_labels` must be an integer >= 1 for task '{task}'.")
            num_state_elements = num_classes
            self.num_classes = num_classes
        else:
            raise ValueError(f"Task {task} not supported!")

        self.add_state("tp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.to(torch.long)
        stats = None

        if self.task == "binary":
            if preds.ndim > 1 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)

            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)

            hard_preds = (preds >= self.threshold).long()

            stats = stat_scores(
                preds=hard_preds,
                target=target,
                task="binary",
                threshold=self.threshold,
                average="none",
            )

        elif self.task == "multilabel":
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)

            hard_preds = (preds >= self.threshold).long()

            # Handle ignore_index by computing stats manually per label
            if self.ignore_index is not None:
                # Create mask for valid entries (not ignored)
                valid_mask = (target != self.ignore_index)  # (B, num_labels)

                # Compute stats per label, only counting valid entries
                tp = torch.zeros(self.num_classes, device=preds.device)
                fp = torch.zeros(self.num_classes, device=preds.device)
                tn = torch.zeros(self.num_classes, device=preds.device)
                fn = torch.zeros(self.num_classes, device=preds.device)

                for label_idx in range(self.num_classes):
                    mask = valid_mask[:, label_idx]  # (B,)
                    if mask.sum() == 0:
                        continue

                    pred_label = hard_preds[:, label_idx][mask]  # valid preds for this label
                    target_label = target[:, label_idx][mask]    # valid targets for this label

                    tp[label_idx] = ((pred_label == 1) & (target_label == 1)).sum()
                    fp[label_idx] = ((pred_label == 1) & (target_label == 0)).sum()
                    tn[label_idx] = ((pred_label == 0) & (target_label == 0)).sum()
                    fn[label_idx] = ((pred_label == 0) & (target_label == 1)).sum()

                self.tp += tp
                self.fp += fp
                self.tn += tn
                self.fn += fn
                return  # Early return since we've already updated states

            else:
                stats = stat_scores(
                    preds=hard_preds,
                    target=target,
                    task="multilabel",
                    num_labels=self.num_classes,
                    average=None,
                )

        elif self.task == "multiclass":
            if preds.ndim == 2 and preds.size(1) == self.num_classes:
                preds = torch.argmax(preds, dim=1)

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            )

        if stats.ndim == 1:
            stats = stats.unsqueeze(0)

        tp, fp, tn, fn, _ = stats.unbind(dim=1)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> torch.Tensor:
        """Compute the final Balanced Accuracy score."""
        recall = self.tp / (self.tp + self.fn + 1e-8)
        specificity = self.tn / (self.tn + self.fp + 1e-8)
        balanced_acc = (recall + specificity) / 2
        return balanced_acc.mean()
