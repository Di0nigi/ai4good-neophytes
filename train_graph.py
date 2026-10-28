import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from datasets.graph_data import TileGraphDataModule
from models.sagr import HybridResNet18GNN


def main():

    df = pd.read_csv("/zfs/ai4good/datasets/Neophytes/stats_imagewise.csv")

    df = df[df["site"] == "Aarau_1_1"]
    df["image"] = df["img_path"].str.replace(
        "../data/", "/zfs/ai4good/datasets/", regex=False
    )
    df["mask"] = df["mask_path"].str.replace(
        "../data/", "/zfs/ai4good/datasets/", regex=False
    )

    train = df[df["split"] == "train"][["image", "mask"]]
    test = df[df["split"] == "test"][["image", "mask"]]
    validation = df[df["split"] == "val"][["image", "mask"]]

    train_dict = train.to_dict(orient="records")
    test_dict = test.to_dict(orient="records")
    class_values = [1, 2, 3, 4, 5, 6]

    dm = TileGraphDataModule(
        train_records=train_dict,
        val_records=test_dict,
        class_values=class_values,
        cache_dir="./graph_cache",
        batch_size=32,
        num_workers=4,
        graph_kwargs={
            "image_bands": (1, 2, 3),
            "image_scale": 255.0,
            "felz_scale": 100,
            "felz_sigma": 0.3,
            "felz_min_size": 10,
        },
    )

    dm.setup()
    example = dm.train_dataset[0]
    print("Node feature shape:", example.x.shape)
    print("Edge index shape:", example.edge_index.shape)
    print("Number of labeled nodes:", int((example.y >= 0).sum()))

    num_classes = len(class_values)
    in_channels = example.x.shape[1]

    model = HybridResNet18GNN(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=64,
    )
    logger = CSVLogger("./", name="graph_log.csv")
    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
    )

    last_ckpt_path = os.path.join("./", "checkpoints", "last.ckpt")
    ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
