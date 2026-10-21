import numpy as np
from torch.utils.data import Dataset


class PromptDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = dataset.classes
        self.background_name = dataset.background_stats_name
        # index -> stats_name, ordered by class index
        self.index_to_name = {
            i: self.classes[i]["stats_name"] for i in self.classes.keys()
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        mask = sample["mask"].numpy()
        vals, counts = np.unique(mask, return_counts=True)
        # drop background and ignore
        candidates = [
            (int(v), int(c)) for v, c in zip(vals, counts)
            if self.index_to_name.get(int(v)) not in (None, self.background_name)
        ]
        if candidates:
            dominant = max(candidates, key=lambda x: x[1])[0]
            sample["prompt"] = self.index_to_name[dominant]
        else:
            sample["prompt"] = ""
        return sample


from torch.utils.data import DataLoader
from datasets import neophyte_datamodule as nd


dm = nd.NeophyteDataModule(cfg.data.image_dir,
                                    cfg.data.classes,
                                    train_transforms,
                                    val_transforms,
                                    test_transforms,
                                    dsm_transform=dsm_transforms,
                                    ndsm_transform=ndsm_transforms,
                                    dataset_stats=cfg.data.dataset_stats,
                                    weighted_sampling=cfg.data.weighted_sampling,
                                    cap_quantile=cfg.data.cap_quantile,
                                    lift_quantile=cfg.data.lift_quantile,
                                    bg_fraction=cfg.data.bg_fraction,
                                    min_strat_images=cfg.data.min_strat_images,
                                    value_mapping=cfg.data.value_mapping,
                                    ignore_index=cfg.data.ignore_index,
                                    load_auxiliary=cfg.data.load_auxiliary,
                                    load_dsm=cfg.data.load_dsm,
                                    load_dtm=cfg.data.load_dtm,
                                    load_ndsm=cfg.data.load_ndsm,
                                    few_shot_k=cfg.data.few_shot_k,
                                    few_shot_mode=cfg.data.few_shot_mode,
                                    few_shot_bg_k=cfg.data.few_shot_bg_k,
                                    max_nodata_frac=cfg.data.max_nodata_frac,
                                    val_few_shot_k=cfg.data.val_few_shot_k,
                                    val_few_shot_mode=cfg.data.val_few_shot_mode,
                                    val_few_shot_bg_k=cfg.data.val_few_shot_bg_k,
                                    load_into_ram=cfg.data.load_into_ram,
                                    batch_size_train=cfg.data.batch_size_train,
                                    batch_size_val=cfg.data.batch_size_val,
                                    batch_size_test=cfg.data.batch_size_test,
                                    num_workers=cfg.data.num_workers)
dm.setup(stage="fit")

train_ds = PromptDataset(dm.train_dataset)
val_ds   = PromptDataset(dm.val_dataset)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4,
                          persistent_workers=True, prefetch_factor=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4,
                          persistent_workers=True, prefetch_factor=4, pin_memory=True)


batch = next(iter(train_loader))
for k, v in batch.items():
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v.dtype)
    else:
        print(k, type(v).__name__, len(v), v[:1])



#model = YourModel()
#trainer = pl.Trainer(...)
#trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)