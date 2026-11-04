

import pytorch_lightning as pl
from transformers import Sam3Model, Sam3Processor
from torch import nn
import torchvision.models as models



class PromptGeneratorSam(nn):
    def __init__(self,dimensions_in,dimensions_out,attention_layer_num):

        self.dimsIn = dimensions_in
        self.dimsOut = dimensions_out

        self.projector = nn.Linear()

        
        
        self.att =  nn.MultiheadAttention(),
        self.norm =nn.LayerNorm(),
        self.gelu = nn.GELU(),
        

        self.output = nn.Linear()


        return

    def forward(self,X):
        




        return
    

class Sam3EnsembleModel(pl.LightningModule):
    

    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate,
                 in_channels=None, ignore_index=-1, optimizer='AdamW', lr_scheduler=None,
                 loss='CE', weight=None, patch_2_img_size=False):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.prompt_generator = PromptGeneratorSam()
        self.sam_model = Sam3Model()
        self.sam_processor = Sam3Processor()
        self.classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        return 

    def setup(self, stage=None):
        """TODO: build components and apply the configured freezing policy."""
        raise NotImplementedError('SAM 3 ensemble component setup is not implemented.')

    def prepare_inputs(self, batch):
        """TODO: return (RGB, optional auxiliary tensor), matching the baseline."""
        raise NotImplementedError

    def generate_prompts(self, images):
        """TODO: return a list of short text phrases per image, without targets."""
        raise NotImplementedError

    def segment_instances(self, images, prompts):
        """TODO: return masks [N,H,W] and proposal scores [N] per image.

        Support zero proposals and restore masks to the input image grid.
        """
        raise NotImplementedError

    def classify_instances(self, images, instances):
        """TODO: construct masked RGB crops and return [N,K] logits per image."""
        raise NotImplementedError

    def merge_instances(self, instances, class_logits, output_size):
        """TODO: resolve overlaps/background into semantic logits [B,K,H,W]."""
        raise NotImplementedError

    def _forward(self, x, aux=None):
        """TODO: orchestrate prompting, SAM, classification and semantic merging.

        Return raw semantic logits [B,K,H,W]. Discrete text generation and
        proposal postprocessing require an explicit training strategy: ordinary
        backpropagation cannot cross token sampling or hard mask selection.
        """
        raise NotImplementedError

    def forward(self, x, aux=None):
        """TODO: return per-pixel probabilities [B,K,H,W], like the baseline."""
        raise NotImplementedError

    def _step(self, batch, stage):
        """TODO: use batch['mask'] for supervision only; return scalar loss.

        Log train/val/test loss, acc, iou, f1, auc and per-class f1 following
        semseg_plm.py. Existing callbacks require the epoch-level val_f1 metric.
        Semantic labels do not supply true instance IDs for touching plants.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def _epoch_end(self, stage):
        """TODO: compute, log and reset metrics following semseg_plm.py."""
        raise NotImplementedError

    def on_train_epoch_end(self):
        self._epoch_end('train')

    def on_validation_epoch_end(self):
        self._epoch_end('val')

    def on_test_epoch_end(self):
        self._epoch_end('test')

    def configure_optimizers(self):
        """TODO: optimize trainable components using the shared scheduler config."""
        raise NotImplementedError

    def set_patch_2_img_size(self, new_patch_2_img_size):
        self.hparams.patch_2_img_size = new_patch_2_img_size
