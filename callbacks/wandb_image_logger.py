from typing import Any, Dict

import torch
from pytorch_lightning import Callback, Trainer
import wandb
from lighter import LighterSystem
from monai.visualize import blend_images

class WandbImageLogger(Callback):
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LighterSystem, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            x = outputs["input"].detach().cpu()
            y = outputs["target"].detach().cpu()
            y_pred = outputs["pred"].detach().cpu()

            assert x.shape == y.shape == y_pred.shape

            # Create overlay images for predictions and ground truth
            pred_overlay = blend_images(image=x, label=y_pred, alpha=0.3, cmap="hsv").squeeze()
            gt_overlay = blend_images(image=x, label=y, alpha=0.3, cmap="hsv").squeeze()

            views = ["axial", "coronal", "sagittal"]
            indices = [pred_overlay.shape[1] // 2, pred_overlay.shape[2] // 2, pred_overlay.shape[3] // 2]

            images = []
            for view, idx in zip(views, indices):
                if view == "axial":
                    pred_image = wandb.Image(pred_overlay[:, idx, :, :], caption=f"pred_{view}")
                    gt_image = wandb.Image(gt_overlay[:, idx, :, :], caption=f"gt_{view}")
                elif view == "coronal":
                    pred_image = wandb.Image(pred_overlay[:, :, idx, :], caption=f"pred_{view}")
                    gt_image = wandb.Image(gt_overlay[:, :, idx, :], caption=f"gt_{view}")
                else:  # sagittal
                    pred_image = wandb.Image(pred_overlay[:, :, :, idx], caption=f"pred_{view}")
                    gt_image = wandb.Image(gt_overlay[:, :, :, idx], caption=f"gt_{view}")

                images.extend([pred_image, gt_image])

            # Log images to WandB
            trainer.logger.log_image("images", images)
