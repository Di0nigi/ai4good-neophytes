import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.nn import GATConv, SAGEConv
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)


class HybridResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HybridResNetBlock, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)
        self.gat_conv = GATConv(out_channels, out_channels // 4, heads=4, concat=True)
        self.bn1 = BatchNorm1d(out_channels)
        self.bn2 = BatchNorm1d(out_channels)
        self.relu = ReLU(inplace=False)
        self.shortcut = (
            Linear(in_channels, out_channels) if in_channels != out_channels else None
        )

    def forward(self, x, edge_index):
        identity = x
        out = self.sage_conv(x, edge_index)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.gat_conv(out, edge_index)
        out = self.bn2(out)
        out = self.relu(out)
        if self.shortcut:
            identity = self.shortcut(identity)
        out = out + identity
        out = self.relu(out)
        return out


class HybridResNet18GNN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=64,
        class_weights=None,
        lr: float = 1e-3,
    ):
        super(HybridResNet18GNN, self).__init__()
        self.lr = lr
        self.initial_conv = SAGEConv(in_channels, hidden_channels)
        self.initial_bn = BatchNorm1d(hidden_channels)
        self.initial_relu = ReLU(inplace=False)

        self.block1 = HybridResNetBlock(hidden_channels, hidden_channels)
        self.block2 = HybridResNetBlock(hidden_channels, hidden_channels)
        self.block3 = HybridResNetBlock(hidden_channels, hidden_channels * 2)
        self.block4 = HybridResNetBlock(hidden_channels * 2, hidden_channels * 2)
        self.block5 = HybridResNetBlock(hidden_channels * 2, hidden_channels * 4)
        self.block6 = HybridResNetBlock(hidden_channels * 4, hidden_channels * 4)
        self.block7 = HybridResNetBlock(hidden_channels * 4, hidden_channels * 8)
        self.block8 = HybridResNetBlock(hidden_channels * 8, hidden_channels * 8)

        self.fc = Linear(hidden_channels * 8, num_classes)

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )

        for split in ["train", "val", "test"]:
            setattr(self, f"{split}_acc", MulticlassAccuracy(num_classes=num_classes, average="macro", ignore_index=-1))
            setattr(self, f"{split}_iou", MulticlassJaccardIndex(num_classes=num_classes, average="macro", ignore_index=-1))
            setattr(self, f"{split}_f1", MulticlassF1Score(num_classes=num_classes, average="macro", ignore_index=-1))
            setattr(self, f"{split}_auc", MulticlassAUROC(num_classes=num_classes, average="macro", ignore_index=-1, thresholds=5))
            setattr(self, f"{split}_f1_per_class", MulticlassF1Score(num_classes=num_classes, average=None, ignore_index=-1))

    def forward(self, x, edge_index):
        x = self.initial_conv(x, edge_index)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = self.block4(x, edge_index)
        x = self.block5(x, edge_index)
        x = self.block6(x, edge_index)
        x = self.block7(x, edge_index)
        x = self.block8(x, edge_index)
        x = self.fc(x)

        return x  # Return raw logits, not log_softmax

    def _shared_step(self, batch, stage):
        logits = self(batch.x, batch.edge_index)
        valid = batch.y >= 0

        if not valid.any():
            print(f"{stage} batch contains no labeled graph nodes.")
            return

        # Ensure labels are within valid range
        num_classes = self.fc.out_features
        y_valid = batch.y[valid]
        if (y_valid >= num_classes).any():
            self.log(f"Warning: Found labels >= num_classes ({num_classes}). Clamping.")
            y_valid = torch.clamp(y_valid, max=num_classes - 1)
        
        loss = F.cross_entropy(
            logits[valid],
            y_valid,
            weight=self.class_weights,
        )

        preds = torch.argmax(logits, dim=1)
        batch_size = int(valid.sum())

        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, batch_size=batch_size)
        getattr(self, f"{stage}_acc").update(preds[valid].detach(), y_valid.detach())
        getattr(self, f"{stage}_iou").update(preds[valid].detach(), y_valid.detach())
        getattr(self, f"{stage}_f1").update(preds[valid].detach(), y_valid.detach())
        getattr(self, f"{stage}_auc").update(logits[valid].detach(), y_valid.detach().long())
        getattr(self, f"{stage}_f1_per_class").update(preds[valid].detach(), y_valid.detach())
        return loss

    def _epoch_end(self, stage, prog_bar=True):
        self.log_dict({
            f"{stage}_acc": getattr(self, f"{stage}_acc").compute(),
            f"{stage}_iou": getattr(self, f"{stage}_iou").compute(),
            f"{stage}_f1": getattr(self, f"{stage}_f1").compute(),
            f"{stage}_auc": getattr(self, f"{stage}_auc").compute(),
        }, prog_bar=prog_bar)

        for i, f1 in enumerate(getattr(self, f"{stage}_f1_per_class").compute()):
            self.log(f"{stage}_f1_class_{i}", f1, prog_bar=False)

        for metric in ("acc", "iou", "f1", "auc", "f1_per_class"):
            getattr(self, f"{stage}_{metric}").reset()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        return optimizer
