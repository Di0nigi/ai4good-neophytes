"""Lightning adapter for the binary SAM3 ensemble and six-species evaluation."""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.sam3_model import Sam3EnsembleModel
from SAM.samPromptsDescriptions import speciesDictById
from utils.scheduler_utils import get_scheduler


class Sam3SegmentationModel(pl.LightningModule):
    def __init__(self, model='Sam3Ensemble', encoder_name='sam3', img_size=512,
                 num_classes=7, learning_rate=1e-4, in_channels=None,
                 ignore_index=-1, optimizer='AdamW', lr_scheduler=None,
                 loss='DiceCE', weight=None, patch_2_img_size=True, fusion_mode='concat',
                 sam_checkpoint='facebook/sam3', max_proposals=8, height_max_m=10.,
                 threshold=0.5, rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225), height_mean=1.43, height_std=3.10):
        super().__init__()
        self.save_hyperparameters()
        if num_classes != 7 or loss != 'DiceCE':
            raise ValueError('SAM3 uses 6 species + nothing and binary DiceCE loss.')
        self.num_classes = num_classes
        self.in_channels = in_channels or {'image': 3, 'ndsm': 1}
        self.ensemble = Sam3EnsembleModel(2, sam_checkpoint=sam_checkpoint,
                                         max_proposals=max_proposals,
                                         height_max_m=height_max_m, ignore_index=ignore_index)
        self.register_buffer('rgb_mean', torch.tensor(list(rgb_mean)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('rgb_std', torch.tensor(list(rgb_std)).view(1, 3, 1, 1), persistent=False)
        for stage in ['val', 'test']:
            self.register_buffer(f'{stage}_confusion', torch.zeros(7, 7, dtype=torch.long), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.ensemble.sam_model.eval()
        self.ensemble.prompt_generator.clipModel.eval()
        return self

    def configure_optimizers(self):
        cls = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[self.hparams.optimizer]
        optimizer = cls((p for p in self.parameters() if p.requires_grad), lr=self.hparams.learning_rate)
        if self.hparams.lr_scheduler:
            scheduler = get_scheduler(optimizer, self.hparams.lr_scheduler)
            return {'optimizer': optimizer, 'lr_scheduler': {
                'scheduler': scheduler, 'interval': self.hparams.lr_scheduler.get('interval', 'epoch')}}
        return optimizer

    def prepare_inputs(self, batch):
        """Same interface as the professor's model; batch masks are not read."""
        return batch['image'], batch.get('ndsm')

    def set_patch_2_img_size(self, value):
        # SAM always evaluates in training-size crops to preserve plant scale.
        self.hparams.patch_2_img_size = True

    def _decode_inputs(self, images, height):
        rgb = (images * self.rgb_std + self.rgb_mean).clamp(0, 1)
        if height is None:
            heights = torch.zeros_like(rgb)
        else:
            metres = height * self.hparams.height_std + self.hparams.height_mean
            metres = torch.nan_to_num(metres, nan=0., posinf=0., neginf=0.)
            heights = metres.clamp(0, self.hparams.height_max_m) / self.hparams.height_max_m
            heights = heights.repeat(1, 3, 1, 1)
        return rgb, heights

    def _targets(self, masks, labels):
        target = (masks == labels[:, None, None]).long()
        invalid = (masks == self.hparams.ignore_index) | (masks == 254) | (masks == 255)
        return target.masked_fill(invalid, self.hparams.ignore_index)

    def _loss(self, logits, targets):
        valid = targets != self.hparams.ignore_index
        if not valid.any():
            return logits.sum() * 0
        ce = F.cross_entropy(logits, targets, ignore_index=self.hparams.ignore_index)
        probs = logits.float().softmax(1)[:, 1]
        return ce + self._dice(probs, targets, valid)

    @staticmethod
    def _dice(probs, targets, valid):
        foreground = (targets == 1).float()
        # Per-sample smoothing gives empty masks a finite, useful loss.
        dims = (-2, -1)
        numerator = 2 * (probs * foreground * valid).sum(dims) + 1.
        denominator = (probs * valid).sum(dims) + (foreground * valid).sum(dims) + 1.
        keep = valid.any(dim=-1).any(dim=-1)
        return (1 - numerator / denominator)[keep].mean()

    def training_step(self, batch, batch_idx):
        images, heights = self._decode_inputs(*self.prepare_inputs(batch))
        masks = batch['mask'].long()
        # One prompt per tile limits memory. Alternate a present-species prompt
        # with a uniformly sampled species, including absent-species negatives.
        labels = torch.randint(1, 7, (len(images),), device=images.device)
        for i, mask in enumerate(masks):
            present = torch.unique(mask[(mask > 0) & (mask <= 6)])
            if len(present) and (self.global_step + i) % 2 == 0:
                labels[i] = present[torch.randint(len(present), (), device=images.device)]
        texts = [speciesDictById[int(label)] for label in labels]
        logits = self.ensemble(images, texts, heights)
        loss = self._loss(logits, self._targets(masks, labels))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=len(images), sync_dist=True)
        return loss

    def _crop_scores(self, images, heights):
        return torch.stack([
            self.ensemble(images, [speciesDictById[label]] * len(images), heights).float().softmax(1)[:, 1]
            for label in range(1, 7)
        ], dim=1)

    def species_scores(self, images, height=None):
        """Six binary foreground probabilities, no ground-truth prompt selection."""
        images, heights = self._decode_inputs(images, height)
        h, w = images.shape[-2:]
        size = self.hparams.img_size
        pad = (0, (-w) % size, 0, (-h) % size)
        images, heights = F.pad(images, pad, mode='replicate'), F.pad(heights, pad, mode='replicate')
        scores = images.new_zeros((len(images), 6, *images.shape[-2:]), dtype=torch.float32)
        for y in range(0, images.shape[-2], size):
            for x in range(0, images.shape[-1], size):
                scores[:, :, y:y+size, x:x+size] = self._crop_scores(
                    images[:, :, y:y+size, x:x+size], heights[:, :, y:y+size, x:x+size])
        return scores[:, :, :h, :w]

    def forward(self, images, aux=None):
        scores = self.species_scores(images, aux)
        # A constant rejection channel preserves species argmax and the threshold.
        nothing = torch.full_like(scores[:, :1], self.hparams.threshold)
        scores = torch.cat([nothing, scores], dim=1)
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _evaluation_step(self, batch, stage):
        scores = self.species_scores(*self.prepare_inputs(batch))
        masks = batch['mask'].long()
        confidence, winner = scores.max(1)
        predictions = (winner + 1).masked_fill(confidence <= self.hparams.threshold, 0)
        valid = (masks != self.hparams.ignore_index) & (masks != 254) & (masks != 255)
        confusion = torch.bincount((masks[valid] * 7 + predictions[valid]).long(), minlength=49).reshape(7, 7)
        getattr(self, f'{stage}_confusion').add_(confusion)
        losses = []
        for label in range(1, 7):
            targets = self._targets(masks, torch.full((len(masks),), label, device=masks.device))
            if valid.any():
                prob = scores[:, label-1].clamp(1e-6, 1-1e-6)
                ce = F.binary_cross_entropy(prob[valid], targets[valid].float())
                losses.append(ce + self._dice(prob, targets, valid))
        loss = torch.stack(losses).mean() if losses else scores.sum() * 0
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(masks), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluation_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._evaluation_step(batch, 'test')

    def _epoch_end(self, stage):
        matrix = getattr(self, f'{stage}_confusion').clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(matrix)
        matrix = matrix.float()
        tp = matrix.diag()[1:]
        fp, fn = matrix.sum(0)[1:] - tp, matrix.sum(1)[1:] - tp
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        iou = tp / (tp + fp + fn).clamp_min(1)
        present = (tp + fp + fn) > 0
        self.log(f'{stage}_f1', f1[present].mean() if present.any() else f1.sum(), prog_bar=True)
        self.log(f'{stage}_iou', iou[present].mean() if present.any() else iou.sum(), prog_bar=True)
        for label, value in enumerate(f1, start=1):
            self.log(f'{stage}_f1_class_{label}', value)
        getattr(self, f'{stage}_confusion').zero_()

    def on_validation_epoch_end(self):
        self._epoch_end('val')

    def on_test_epoch_end(self):
        self._epoch_end('test')
