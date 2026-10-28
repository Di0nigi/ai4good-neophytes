import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.nn import GATConv, SAGEConv
from torchmetrics.classification import MulticlassAccuracy


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
        self, in_channels, num_classes, hidden_channels=64, class_weights=None
    ):
        super(HybridResNet18GNN, self).__init__()
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

        self.train_acc = MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=-1,
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=num_classes,
            ignore_index=-1,
        )

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

        return F.log_softmax(x, dim=1)

    def _shared_step(self, batch, stage):
        logits = self(batch.x, batch.edge_index)
        valid = batch.y >= 0

        if not valid.any():
            print(f"{stage} batch contains no labeled graph nodes.")
            return 

        loss = F.cross_entropy(
            logits[valid],
            batch.y[valid],
            weight=self.class_weights,
        )

        probs = logits[valid].softmax(dim=-1)
        targets = batch.y[valid]

        if stage == "train":
            self.train_acc.update(probs, targets)
            self.log("train/loss", loss, on_step=False, on_epoch=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        else:
            self.val_acc.update(probs, targets)
            self.log("val/loss", loss, on_step=False, on_epoch=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        return optimizer
