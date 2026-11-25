import os

import cv2
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import mark_boundaries

from datasets.neophyte_dataset import get_normalize_params
from models.sagr import propagate_node_labels


def save_graph_predictions(
    model,
    data_module,
    class_names,
    class_colors,
    save_dir,
    splits=("val", "test"),
    num_tiles=24,
    max_plot_size=1024,
):
    """Draw qualitative prediction figures for the first val/test tiles
    where the model predicts a class other than the background."""

    device = next(model.parameters()).device
    num_classes = model.fc.out_features
    if len(class_names) != num_classes:
        raise ValueError(
            f"{len(class_names)} class names given but the model predicts "
            f"{num_classes} classes."
        )

    # undo the input normalization for display
    norm_mean, norm_std = get_normalize_params(
        data_module.transform_val or data_module.transform_test
    )
    norm_mean = np.asarray(norm_mean if norm_mean is not None else [0.0, 0.0, 0.0])
    norm_std = np.asarray(norm_std if norm_std is not None else [1.0, 1.0, 1.0])

    def class_overlay(image, mask):
        """Paint the class colors over a brightened grayscale of the tile."""
        gray = np.clip(np.dot(image, [0.2989, 0.5870, 0.1140]) * 1.15, 0, 1)
        overlay = np.repeat(gray[..., None], 3, axis=2)
        for c in range(1, num_classes):  # 0 = background stays gray
            overlay[mask == c] = mcolors.to_rgb(class_colors[c])
        return overlay

    def downscale(arr, interpolation):
        h, w = arr.shape[:2]
        scale = min(1.0, max_plot_size / max(h, w))
        if scale >= 1.0:
            return arr
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=interpolation)

    for split in splits:
        dataset = getattr(data_module, f"{split}_dataset", None)
        if dataset is None:
            raise ValueError(f"Split '{split}' was not set up.")

        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"[{split}] scanning {len(dataset)} tiles for up to {num_tiles} "
              f"with predicted non-background classes -> {split_dir}")

        order = np.random.default_rng(0).permutation(len(dataset))
        drawn = 0
        for idx in order:
            if drawn >= num_tiles:
                break
            vis = dataset.get_visualization_sample(idx)
            graph = vis["graph"].to(device)

            with torch.no_grad():
                logits = model(graph.x, graph.edge_index)
                probs = torch.softmax(logits, dim=1)
                node_pred = propagate_node_labels(
                    probs, graph.edge_index,
                    **model.prediction_node_label_propagation_args,
                )

            pred_mask = node_pred.cpu().numpy()[vis["segments"]]
            if not np.any(pred_mask):  # only class 0 (ground) predicted
                continue
            drawn += 1
            image = np.clip(vis["image"] * norm_std + norm_mean, 0.0, 1.0)

            panels = [
                (image, "Image", cv2.INTER_AREA),
                (class_overlay(image, vis["mask"]), "Reference", cv2.INTER_NEAREST),
                (mark_boundaries(image, vis["segments"], color=(1, 1, 0), mode="outer"),
                 "Segments", cv2.INTER_AREA),
                (class_overlay(image, pred_mask), "Prediction", cv2.INTER_NEAREST),
            ]
            panels = [
                (downscale(panel, interp), title) for panel, title, interp in panels
            ]
            pan_h, pan_w = panels[0][0].shape[:2]
            fig, axes = plt.subplots(
                1, len(panels), figsize=(4 * pan_w / 100, pan_h / 100 + 1.4), dpi=100
            )
            for ax, (panel, title) in zip(np.atleast_1d(axes), panels):
                ax.imshow(panel)
                ax.set_title(title, fontsize=16)
                ax.axis("off")

            stem = os.path.splitext(os.path.basename(str(vis["name"])))[0]
            fig.suptitle(f"{split}  {stem}", fontsize=14)
            fig.legend(
                handles=[
                    mpatches.Patch(color=class_colors[c], label=class_names[c])
                    for c in range(1, num_classes)
                ],
                loc="lower center", ncol=min(num_classes - 1, 6), frameon=False,
            )
            out_path = os.path.join(split_dir, f"{split}_{idx:04d}_{stem}.png")
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_path}")

        if drawn == 0:
            print(f"[{split}] no tiles with predicted non-background classes")
