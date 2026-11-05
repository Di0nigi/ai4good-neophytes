

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
    def __init__(self,dimensions_in = 1536 ,dimensions_out = 0,emb_dims= 512,attention_layer_num=8):

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
        
        return self

    def prepare_inputs(self, batch, stage=None):
        images = batch["image"]                 
        masks = batch["mask"].long()   

        texts, labels = [], []
        for m in masks:
            present = torch.unique(m).tolist()
            
            cand = [int(c) for c in present if int(c) > 0 and int(c) in speciesDict]
            if cand:
                lab = cand[0]
                texts.append(speciesDict[lab])
                labels.append(lab)
            else:
                texts.append("background")
                labels.append(0)
        
        dup_imgs, dup_masks, dup_texts, dup_labels = [], [], [], []
        for i, m in enumerate(masks):
            present = torch.unique(m).tolist()
            cand = [int(c) for c in present if int(c) > 0 and int(c) in speciesDict]
            if len(cand) <= 1:
                continue
            # first label already used in the original tile; add the rest
            for lab in cand[1:]:
                dup_imgs.append(images[i])
                # keep only that label in the duplicate's mask (rest -> background)
                new_mask = m.clone()
                new_mask[new_mask != lab] = 0
                dup_masks.append(new_mask)
                dup_texts.append(speciesDict[lab])
                dup_labels.append(lab)

        if dup_imgs:
            images = torch.cat([images, torch.stack(dup_imgs)], dim=0)
            masks  = torch.cat([masks,  torch.stack(dup_masks)], dim=0)
            texts  = texts + dup_texts
            labels = labels + dup_labels

        batch["image"] = images
        batch["mask"]  = masks
        batch["prompt_text"] = texts
        batch["prompt_label"] = labels

        enhancedIm = images
        #if "image_clahe" in batch:
        #    enhancedIm = torch.cat([images, batch["image_clahe"]], dim=1)
        if "image_edges" in batch:
            enhancedIm = torch.cat([images, batch["image_edges"]], dim=1)

        if "ndsm" in batch:
            heightMap = batch["ndsm"].float()
            if heightMap.shape[1] == 1:
                heightMap = heightMap.repeat(1, 3, 1, 1)
        else:
            heightMap = torch.zeros_like(images)

        textPrompt = texts

        promptInputs = (enhancedIm, heightMap, textPrompt)
        boxes = []
        for m in masks:
            ys, xs = torch.where(m > 0)
            if ys.numel() == 0:
                # no positive pixels -> full-image box
                h, w = m.shape
                boxes.append([0.0, 0.0, float(w), float(h)])
            else:
                boxes.append([xs.min().item(), ys.min().item(),
                            xs.max().item(), ys.max().item()])
        batch["boxes"] = torch.tensor(boxes, dtype=torch.float32, device=images.device)

        samInputs = {
            "pixel_values": images,
            "input_boxes": batch["boxes"],
        }
        
        return samInputs, promptInputs

    def segment_instances(self, sam_outputs, image_info):
        """Select candidate masks on the original grid for ResNet classification.

        Hard selection is outside the differentiable foreground-loss path.
        SAM's processor handles model-specific resizing and confidence filtering.
        Query IDs are unnecessary here because classification targets are
        assigned from the restored masks, independently of the foreground loss.
        """
        
        return

    def classify_instances(self, images, instances):
        """Build masked RGB crops, apply ResNet preprocessing and classify them.

        Return logits per image in proposal order, including [0,K] for no masks.
        A fixed-size masked canvas retains that order even for an empty mask.
        """
        
        return
    def merge_instances(self, instances, class_logits, output_size):
        """Produce semantic scores [B,K,H,W]; uncovered pixels are background.

        Use the strongest confidence-weighted mask per species, resolve overlaps
        by highest score and reject proposals classified as background.
        """
        return 

    def forward(self, images, generator_inputs):
        """Run variable-length per-tile prompt tuples through SAM and ResNet."""
        return 

    def _step(self, batch, stage):
        """Train generator with foreground loss and ResNet with region labels.

        Proposal labels use majority valid pixels (including background) within
        each mask. Fully ignored/empty regions do not train the classifier.
        This is semantic-region supervision, not individual-plant matching.
        """
        
        return 

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
