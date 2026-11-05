
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from samDataSet import SamNeophyteDataset
from sam3_model import Sam3EnsembleModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 4

NUM_CLASSES = 2  # Binary: other/background vs prompted species
MAX_PROPOSALS = 32
IGNORE_INDEX = -1

# Adapt these to your dataset configuration
TRAIN_DIRS = ["path/to/train"]
VAL_DIRS = ["path/to/val"]

CLASSES = classes = {
    0: {"name": "Background",     "color": "#000000", "stats_name": "background"},
    1: {"name": "B. davidii",     "color": "#dce14c", "stats_name": "B_davidii"},
    2: {"name": "R. typhina",     "color": "#b76e44", "stats_name": "R_typhina"},
    3: {"name": "R. japonica",    "color": "#305dc6", "stats_name": "R_japonica"},
    4: {"name": "B. orientalis",  "color": "#e2256a", "stats_name": "B_orientalis"},
    5: {"name": "S. inaequidens", "color": "#50e3c2", "stats_name": "S_inaequidens"},
    6: {"name": "A. altissima",   "color": "#dd60f6", "stats_name": "A_altissima"},
}

# --------------------------------------------------
# Loss: Cross Entropy + Dice
# --------------------------------------------------

def dice_cce_loss(logits, targets, ignore_index=-1):

    # logits:  [B, 2, H, W]
    # targets: [B, H, W]

    cce = F.cross_entropy(
        logits,
        targets.long(),
        ignore_index=ignore_index
    )

    probs = torch.softmax(logits, dim=1)

    valid = (targets != ignore_index).float()
    foreground = (targets == 1).float()

    pred_fg = probs[:, 1]

    intersection = (pred_fg * foreground * valid).sum()
    denominator = (
        (pred_fg * valid).sum()
        + (foreground * valid).sum()
    )

    dice = 1 - (
        (2 * intersection + 1e-6)
        / (denominator + 1e-6)
    )

    return cce + dice


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = SamNeophyteDataset(
    image_dirs=TRAIN_DIRS,
    classes=CLASSES,
    # Add your training transforms here
)

val_dataset = SamNeophyteDataset(
    image_dirs=VAL_DIRS,
    classes=CLASSES,
    # Add your validation transforms here
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = Sam3EnsembleModel(
    num_classes=NUM_CLASSES,
    max_proposals=MAX_PROPOSALS,
    ignore_index=IGNORE_INDEX
).to(DEVICE)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# --------------------------------------------------
# Training / validation epoch
# --------------------------------------------------

def run_epoch(loader, training=True):

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE).long()

        # Optional nDSM
        height_maps = batch.get("ndsm")

        if height_maps is not None:
            height_maps = height_maps.to(DEVICE)

            # CLIP image encoder expects 3 channels
            if height_maps.shape[1] == 1:
                height_maps = height_maps.repeat(1, 3, 1, 1)
        else:
            height_maps = torch.zeros_like(images)

        # Expand each tile into one sample per species
        (
            enhanced_images,
            height_maps,
            texts,
            labels,
            binary_masks
        ) = model.prepare_inputs(
            images,
            masks,
            height_maps=height_maps
        )

        enhanced_images = enhanced_images.to(DEVICE)
        height_maps = height_maps.to(DEVICE)
        targets = binary_masks.to(DEVICE).long()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            # Prompt generator -> SAM3 -> classifier
            logits = model(
                enhanced_images,
                texts,
                height_maps
            )

            loss = dice_cce_loss(
                logits,
                targets,
                ignore_index=IGNORE_INDEX
            )

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    train_loss = run_epoch(
        train_loader,
        training=True
    )

    val_loss = run_epoch(
        val_loader,
        training=False
    )

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS:03d}] "
        f"| Train Loss: {train_loss:.5f} "
        f"| Val Loss: {val_loss:.5f}"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        "sam3_checkpoint_last.pth"
    )
