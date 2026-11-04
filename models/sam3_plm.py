

import pytorch_lightning as pl
from transformers import Sam3Model, Sam3Processor
from torch import nn
import torchvision.models as models

import open_clip
import torch
from PIL import Image
from PIL import ImageOps
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from models.sam3_losses import foreground_seg_loss
from utils.scheduler_utils import get_scheduler

from SAM.samPromptsDescriptions import speciesDict


class PromptGeneratorSam(nn.Module):
    def __init__(self,dimensions_in = 1536 ,dimensions_out =0,emb_dims= 512,attention_layer_num=8):

        super().__init__()
        self.dimsIn = dimensions_in
        self.dimsOut = dimensions_out
        self.embDims = emb_dims
        self.layerNum = attention_layer_num


        self.clipModel, _, self.clipPreprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')

        self.tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

        self.projector = nn.Linear(self.dimsIn,self.embDims)

        self.att =  nn.ModuleList([nn.MultiheadAttention(self.embDims,num_heads=8,batch_first=True) for _ in range(self.layerNum)])
        self.norm =nn.ModuleList([nn.LayerNorm(self.embDims) for _ in range(self.layerNum)])
        self.gelu = nn.ModuleList([nn.GELU() for _ in range(self.layerNum)])


        self.output = nn.Linear(self.embDims,self.dimsOut)

        return

    def forward(self,X):

        enhancedIm,heightMap,textPrompt = X

        imgEmb = self.clipModel.encode_image(enhancedIm)
        imgEmb = imgEmb / imgEmb.norm(dim=-1, keepdim=True)


        hEmb = self.clipModel.encode_image(heightMap)
        hEmb = hEmb / hEmb.norm(dim=-1, keepdim=True)


        tokens = self.tokenizer(textPrompt).to(enhancedIm.device)
        #if tokens.device != enhancedIm.device:
        #    tokens = tokens.to(enhancedIm.device)
        tEmb = self.clipModel.encode_text(tokens)
        tEmb = tEmb / tEmb.norm(dim=-1, keepdim=True)


        embedding = torch.cat([hEmb, imgEmb, tEmb], dim=-1)

        projected = self.projector(embedding)

        outTensor = projected.unsqueeze(1)

        for i in range(self.layerNum):

            out,_ = self.att[i](outTensor,outTensor,outTensor)

            norm = self.norm[i](out) + outTensor

            outTensor = self.gelu[i](norm)

        output =self.output(outTensor.squeeze(1))

        return output


class Sam3EnsembleModel(pl.LightningModule):
    """Continuous prompt generator -> SAM 3 masks -> ResNet species classifier.

    Training uses labelled species to choose prompts. Validation/test use the
    complete prompt dictionary so query class presence is not disclosed.
    """


    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate,
                 in_channels=None, ignore_index=-1, optimizer='AdamW', lr_scheduler=None,
                 loss='CE', weight=None, patch_2_img_size=False, fusion_mode='concat',
                 sam_checkpoint='facebook/sam3', prompt_checkpoint=None,
                 classifier_name='resnet18', freeze_sam=True,
                 freeze_prompt_generator=False, height_max_m=10.0,
                 proposal_threshold=0.5, max_proposals=32):
        """Build the three components and store reproducible preprocessing settings.

        prompt_checkpoint is retained for old configs; the generator owns its
        MobileCLIP checkpoint. No external language generator is loaded here.
        """
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sam_model = Sam3Model.from_pretrained(sam_checkpoint)
        self.sam_processor = Sam3Processor.from_pretrained(sam_checkpoint)
        self.prompt_generator = PromptGeneratorSam(
            dimensions_out=self.sam_model.config.detr_encoder_config.hidden_size)
        if classifier_name != 'resnet18':
            raise ValueError('Only resnet18 is currently supported.')
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.classifier = models.resnet18(weights=weights)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        self.classifier_preprocess = weights.transforms()
        if freeze_sam:
            self.sam_model.requires_grad_(False)
        if freeze_prompt_generator:
            self.prompt_generator.requires_grad_(False)
        if height_max_m <= 0 or max_proposals <= 0:
            raise ValueError('height_max_m and max_proposals must be positive.')
        for stage in ('train', 'val', 'test'):
            self.register_buffer(f'{stage}_confusion',
                                 torch.zeros(num_classes, num_classes, dtype=torch.long),
                                 persistent=False)

    def train(self, mode=True):
        """Keep frozen components in eval mode without disabling gradient flow."""
        super().train(mode)
        if self.hparams.freeze_sam:
            self.sam_model.eval()
        if self.hparams.freeze_prompt_generator:
            self.prompt_generator.eval()
        return self

    def prepare_inputs(self, batch, stage=None):
        """Return per-tile (enhanced RGB, height, texts), and (RGB, labels).

        Assumes the current ImageNet RGB and nDSM mean=1.43/std=3.10
        normalization. Training selects labelled species; evaluation uses all.
        """
        image, masks = batch['image'], batch['mask']
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        rgb = (image * std + mean).clamp(0, 1)
        height = ((batch['ndsm'] * 3.10 + 1.43).clamp(0, 10) / 10).repeat(1, 3, 1, 1)
        preprocess = self.prompt_generator.clipPreprocess
        keys = list(speciesDict)
        generator_inputs = []
        training = stage == 'train' if stage is not None else self.training

        for colour, elevation, mask in zip(rgb, height, masks):
            ids = [int(c) for c in mask.unique() if c > 0] if training else []
            ids = ids or list(range(1, len(keys) + 1))
            values = [speciesDict[keys[c - 1]] for c in ids]
            texts = [v if isinstance(v, str) else v['description'] for v in values]
            colour = Image.fromarray((colour.detach().cpu().permute(1, 2, 0) * 255).byte().numpy())
            elevation = Image.fromarray((elevation.detach().cpu().permute(1, 2, 0) * 255).byte().numpy())
            enhanced = preprocess(ImageOps.equalize(colour)).to(image.device)
            heightmap = preprocess(elevation).to(image.device)
            generator_inputs.append((
                enhanced.unsqueeze(0).repeat(len(texts), 1, 1, 1),
                heightmap.unsqueeze(0).repeat(len(texts), 1, 1, 1),
                texts,
            ))

        return generator_inputs, (rgb, masks)

    def segment_instances(self, sam_outputs, image_info):
        """Select candidate masks on the original grid for ResNet classification.

        Hard selection is outside the differentiable foreground-loss path.
        SAM's processor handles model-specific resizing and confidence filtering.
        Query IDs are unnecessary here because classification targets are
        assigned from the restored masks, independently of the foreground loss.
        """
        results = self.sam_processor.post_process_instance_segmentation(
            sam_outputs, threshold=self.hparams.proposal_threshold,
            mask_threshold=0.5, target_sizes=image_info)
        instances = []
        for result, size in zip(results, image_info):
            masks, scores = result['masks'].bool(), result['scores']
            if masks.numel() == 0:
                masks = torch.zeros((0, *size), device=scores.device, dtype=torch.bool)
            elif masks.shape[-2:] != size:
                masks = F.interpolate(masks[:,None].float(), size=size,
                                      mode='nearest')[:,0].bool()
            keep = scores.argsort(descending=True)[:self.hparams.max_proposals]
            instances.append({'masks': masks[keep], 'scores': scores[keep]})
        return instances

    def classify_instances(self, images, instances):
        """Build masked RGB crops, apply ResNet preprocessing and classify them.

        Return logits per image in proposal order, including [0,K] for no masks.
        A fixed-size masked canvas retains that order even for an empty mask.
        """
        crops = []
        for image, instance in zip(images, instances):
            for mask in instance['masks']:
                ys, xs = mask.nonzero(as_tuple=True)
                if len(xs):
                    y0, y1 = int(ys.min()), int(ys.max()) + 1
                    x0, x1 = int(xs.min()), int(xs.max()) + 1
                    crop = image[:,y0:y1,x0:x1] * mask[y0:y1,x0:x1]
                else:
                    crop = image.new_zeros((3,224,224))
                crops.append(self.classifier_preprocess(crop))
        counts = [len(item['masks']) for item in instances]
        if not crops:
            return [images.new_empty((0,self.num_classes)) for _ in instances]
        return list(self.classifier(torch.stack(crops)).split(counts))

    def merge_instances(self, instances, class_logits, output_size):
        """Produce semantic scores [B,K,H,W]; uncovered pixels are background.

        Use the strongest confidence-weighted mask per species, resolve overlaps
        by highest score and reject proposals classified as background.
        """
        output = []
        for instance, logits in zip(instances, class_logits):
            scores = logits.new_zeros((self.num_classes,*output_size))
            scores[0] = 0.5
            probabilities = logits.softmax(-1)
            for mask, confidence, probability in zip(instance['masks'], instance['scores'], probabilities):
                if probability.argmax() == 0:
                    continue
                values = probability[:,None,None] * confidence * mask
                scores[1:] = torch.maximum(scores[1:], values[1:])
            output.append(scores)
        return torch.stack(output)

    def forward(self, images, generator_inputs):
        """Run variable-length per-tile prompt tuples through SAM and ResNet."""
        tokens = [self.prompt_generator(inputs) for inputs in generator_inputs]
        lengths = torch.tensor([len(t) for t in tokens], device=images.device)
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        valid = torch.arange(tokens.shape[1], device=images.device)[None] < lengths[:, None]
        reference = next(self.sam_model.parameters())
        tokens = tokens.to(device=reference.device, dtype=reference.dtype)
        rgb_pil = [Image.fromarray((im.detach().cpu().permute(1, 2, 0) * 255).byte().numpy())
                   for im in images]
        pixels = self.sam_processor(images=rgb_pil, return_tensors='pt')['pixel_values'].to(images.device)
        sam_outputs = self.sam_model(
            pixel_values=pixels,
            text_embeds=BaseModelOutputWithPooling(pooler_output=tokens),
            attention_mask=valid, return_dict=True)
        with torch.no_grad():
            instances = self.segment_instances(sam_outputs, [tuple(images.shape[-2:])] * len(images))
        class_logits = self.classify_instances(images, instances)
        with torch.no_grad():
            scores = self.merge_instances(instances, class_logits, images.shape[-2:])
        return {'sam_outputs': sam_outputs, 'instances': instances,
                'class_logits': class_logits, 'scores': scores,
                'predictions': scores.argmax(1)}

    def _step(self, batch, stage):
        """Train generator with foreground loss and ResNet with region labels.

        Proposal labels use majority valid pixels (including background) within
        each mask. Fully ignored/empty regions do not train the classifier.
        This is semantic-region supervision, not individual-plant matching.
        """
        generator_inputs, (images, targets) = self.prepare_inputs(batch, stage)
        output = self(images, generator_inputs)
        raw = output['sam_outputs']
        foreground = foreground_seg_loss(raw.pred_masks, raw.pred_logits, targets,
                                          ignore_index=self.hparams.ignore_index)
        logits, labels = [], []
        for target, instance, region_logits in zip(targets, output['instances'], output['class_logits']):
            for mask, logit in zip(instance['masks'], region_logits):
                values = target[mask & (target != self.hparams.ignore_index)].long()
                if values.numel():
                    labels.append(torch.bincount(values, minlength=self.num_classes).argmax())
                    logits.append(logit)
        cls_loss = (F.cross_entropy(torch.stack(logits), torch.stack(labels)) if logits
                    else next(self.classifier.parameters()).sum() * 0)
        loss = foreground['loss'] + cls_loss
        for name, value in {'loss': loss, 'foreground_loss': foreground['loss'],
                            'bce': foreground['bce'], 'dice': foreground['dice'],
                            'classification_loss': cls_loss}.items():
            self.log(f'{stage}_{name}', value, on_step=stage == 'train',
                     on_epoch=True, batch_size=targets.shape[0], prog_bar=name == 'loss')
        valid = targets != self.hparams.ignore_index
        encoded = self.num_classes * targets[valid].long() + output['predictions'][valid]
        getattr(self, f'{stage}_confusion').add_(
            torch.bincount(encoded, minlength=self.num_classes**2).reshape(self.num_classes,self.num_classes))
        return loss

    def training_step(self, batch, batch_idx):
        """Return the loss Lightning differentiates during training."""
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """Run the same image-only prediction path on held-out images."""
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        """Compute test losses and semantic metrics without weight updates."""
        return self._step(batch, 'test')

    def _epoch_end(self, stage):
        """Log macro F1/IoU/recall and reset the confusion matrix each epoch."""
        counts = getattr(self, f'{stage}_confusion')
        confusion = counts.float().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(confusion)
        tp = confusion.diag()
        support, predicted = confusion.sum(1), confusion.sum(0)
        f1 = 2 * tp / (support + predicted).clamp_min(1)
        iou = tp / (support + predicted - tp).clamp_min(1)
        self.log_dict({f'{stage}_f1': f1.mean(), f'{stage}_iou': iou.mean(),
                       f'{stage}_acc': (tp / support.clamp_min(1)).mean()}, prog_bar=True)
        for index, value in enumerate(f1):
            self.log(f'{stage}_f1_class_{index}', value)
        counts.zero_()

    def on_train_epoch_end(self):
        """Finalize training metrics."""
        self._epoch_end('train')

    def on_validation_epoch_end(self):
        """Finalize validation metrics, including checkpoint monitor val_f1."""
        self._epoch_end('val')

    def on_test_epoch_end(self):
        """Finalize test metrics."""
        self._epoch_end('test')

    def configure_optimizers(self):
        """Optimize only unfrozen parameters using the shared scheduler config."""
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizers = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}
        optimizer = optimizers[self.hparams.optimizer](parameters, lr=self.hparams.learning_rate)
        if self.hparams.lr_scheduler:
            scheduler = get_scheduler(optimizer, self.hparams.lr_scheduler)
            return {'optimizer': optimizer, 'lr_scheduler': {
                'scheduler': scheduler,
                'interval': self.hparams.lr_scheduler.get('interval', 'epoch')}}
        return optimizer

    def set_patch_2_img_size(self, new_patch_2_img_size):
        """Record legacy evaluation settings; SAM owns its input resizing."""
        self.hparams.patch_2_img_size = new_patch_2_img_size
