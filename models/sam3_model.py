import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import open_clip

from transformers import Sam3Model, Sam3Processor
from SAM.samPromptsDescriptions import speciesDict




class MaskClassifierOld(nn.Module):
    def __init__(self, num_classes, mask_channels=1):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            3 + mask_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.conv1 = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class MaskClassifier(nn.Module):
    def __init__(self, num_classes, max_masks=32):
        super().__init__()

        self.max_masks = max_masks

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)

        # ResNet encoder
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1  # H/4, 64
        self.layer2 = backbone.layer2  # H/8, 128
        self.layer3 = backbone.layer3  # H/16, 256
        self.layer4 = backbone.layer4  # H/32, 512

        # Encode ALL SAM masks together
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(max_masks, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decode3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.decode1 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, images, sam_masks):
        """
        images:   [B, 3, H, W]
        sam_masks:[B, N, H, W]

        returns:  [B, num_classes, H, W]
        """

        B, _, H, W = images.shape

        # Pad or truncate proposal channels
        N = sam_masks.shape[1]

        if N < self.max_masks:
            padding = images.new_zeros(
                B, self.max_masks - N, H, W
            )
            sam_masks = torch.cat([sam_masks, padding], dim=1)

        elif N > self.max_masks:
            sam_masks = sam_masks[:, :self.max_masks]

        # ResNet image encoder
        x = self.stem(images)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Encode all proposals simultaneously
        mask_features = self.mask_encoder(sam_masks)

        # Decoder
        d3 = F.interpolate(
            x4, size=x3.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d3 = self.decode3(torch.cat([d3, x3], dim=1))

        d2 = F.interpolate(
            d3, size=x2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d2 = self.decode2(torch.cat([d2, x2], dim=1))

        d1 = F.interpolate(
            d2, size=x1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        d1 = self.decode1(torch.cat([d1, x1], dim=1))

        # Fuse spatial features with all SAM proposals
        mask_features = F.interpolate(
            mask_features,
            size=d1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        fused = self.fusion(
            torch.cat([d1, mask_features], dim=1)
        )

        logits = self.segmentation_head(fused)

        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return logits


class PromptGeneratorSam(nn.Module):
    def __init__(self, dimensions_in=1536, dimensions_out=0, emb_dims=512, attention_layer_num=8):
        super().__init__()
        self.dimsIn = dimensions_in
        self.dimsOut = dimensions_out
        self.embDims = emb_dims
        self.layerNum = attention_layer_num

        self.clipModel, _, self.clipPreprocess = open_clip.create_model_and_transforms(
            'MobileCLIP2-S0', pretrained='dfndr2b')
        self.tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

        self.projector = nn.Linear(self.dimsIn, self.embDims)
        self.att = nn.ModuleList([nn.MultiheadAttention(self.embDims, num_heads=8, batch_first=True)
                                  for _ in range(self.layerNum)])
        self.norm = nn.ModuleList([nn.LayerNorm(self.embDims) for _ in range(self.layerNum)])
        self.gelu = nn.ModuleList([nn.GELU() for _ in range(self.layerNum)])
        self.output = nn.Linear(self.embDims, self.dimsOut)

    def forward(self, X):
        enhancedIm, heightMap, textPrompt = X

        imgEmb = self.clipModel.encode_image(enhancedIm)
        imgEmb = imgEmb / imgEmb.norm(dim=-1, keepdim=True)

        hEmb = self.clipModel.encode_image(heightMap)
        hEmb = hEmb / hEmb.norm(dim=-1, keepdim=True)

        tokens = self.tokenizer(textPrompt).to(enhancedIm.device)
        tEmb = self.clipModel.encode_text(tokens)
        tEmb = tEmb / tEmb.norm(dim=-1, keepdim=True)

        embedding = torch.cat([hEmb, imgEmb, tEmb], dim=-1)
        projected = self.projector(embedding)
        outTensor = projected.unsqueeze(1)

        for i in range(self.layerNum):
            out, _ = self.att[i](outTensor, outTensor, outTensor)
            norm = self.norm[i](out) + outTensor
            outTensor = self.gelu[i](norm)

        return self.output(outTensor.squeeze(1))


class Sam3EnsembleModel(nn.Module):
    

    def __init__(self, num_classes, sam_checkpoint='facebook/sam3',height_max_m=10.0, proposal_threshold=0.5, max_proposals=32,
                 ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.height_max_m = height_max_m
        self.proposal_threshold = proposal_threshold
        self.max_proposals = max_proposals

        self.sam_model = Sam3Model.from_pretrained(sam_checkpoint)
        self.sam_processor = Sam3Processor.from_pretrained(sam_checkpoint)
        self.prompt_generator = PromptGeneratorSam(
            dimensions_out=self.sam_model.config.detr_encoder_config.hidden_size)
        self.classifier = MaskClassifier( num_classes=num_classes,max_masks=max_proposals)

        self.sam_model.requires_grad_(False)
        self.sam_model.eval()

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    
    def prepare_inputsOld(self, images, masks):
        """Build prompt texts/labels from masks (with multi-species duplication).

        Returns: enhancedIm, heightMap, texts, labels, masks (possibly expanded)
        """
        masks = masks.long()
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
            for lab in cand[1:]:
                dup_imgs.append(images[i])
                new_mask = m.clone()
                new_mask[new_mask != lab] = 0
                dup_masks.append(new_mask)
                dup_texts.append(speciesDict[lab])
                dup_labels.append(lab)

        if dup_imgs:
            images = torch.cat([images, torch.stack(dup_imgs)], dim=0)
            masks = torch.cat([masks, torch.stack(dup_masks)], dim=0)
            texts = texts + dup_texts
            labels = labels + dup_labels

        enhancedIm = images
        heightMap = torch.zeros_like(images)  # swap in batch["ndsm"] if available
        return enhancedIm, heightMap, texts, labels, masks

    

    def prepare_inputs(self, images, masks, height_maps=None):

        masks = masks.long()

        if height_maps is None:
            height_maps = torch.zeros_like(images)

        dup_imgs = []
        dup_heightmaps = []
        dup_masks = []
        dup_texts = []
        dup_labels = []

        for i, m in enumerate(masks):

            present = torch.unique(m).tolist()

            candidates = [
                int(c) for c in present
                if int(c) > 0 and int(c) in speciesDict
            ]

            if not candidates:
                continue

            for lab in candidates:

                # Binary target: prompted species = 1,
                # everything else = 0
                target_mask = (m == lab).long()

                dup_imgs.append(images[i])
                dup_heightmaps.append(height_maps[i])
                dup_masks.append(target_mask)

                dup_texts.append(speciesDict[lab])
                dup_labels.append(lab)

        if not candidates:
            
            dup_imgs.append(images[i])
            dup_heightmaps.append(height_maps[i])

            dup_masks.append(torch.zeros_like(m))

            dup_texts.append("background")
            dup_labels.append(0)

        enhancedIm = torch.stack(dup_imgs)
        heightMap = torch.stack(dup_heightmaps)
        masks = torch.stack(dup_masks)

        texts = dup_texts
        labels = dup_labels

        return enhancedIm, heightMap, texts, labels, masks



    def forwardOld(self, images, texts, height_maps=None):
        
        if height_maps is None:
            height_maps = torch.zeros_like(images)

   
        promptEmbedding = self.prompt_generator((images, height_maps, texts)).unsqueeze(1)

        
        samOutputs = self.sam_model(
            pixel_values=images,
            text_embeds=promptEmbedding,
        )

       
        return 


    def forward(self, images, texts, height_maps=None):

        """
        Full pipeline:

            Images + Height maps + Text prompts
                        |
                    PromptGenerator
                        |
                    Text embeddings
                        |
                        SAM3
                        |
                Instance mask proposals
                        |
                Proposal selection
                        |
                    MaskClassifier
                        |
                Per-pixel semantic logits

        Args:
            images:     [B, 3, H, W]
            texts:      List of B species prompts
            height_maps:[B, 3, H, W] (optional)

        Returns:
            semantic_logits: [B, num_classes, H, W]
        """

        B, _, H, W = images.shape

        

        if height_maps is None:
            height_maps = torch.zeros_like(images)

        

        prompt_embedding = self.prompt_generator(
            (images, height_maps, texts)
        )


        prompt_embedding = prompt_embedding.unsqueeze(1)


        sam_outputs = self.sam_model(
            pixel_values=images,
            text_embeds=prompt_embedding,
            return_dict=True
        )

        

        sam_masks = sam_outputs.pred_masks

        
        sam_masks = torch.sigmoid(sam_masks)

        num_queries = sam_masks.shape[1]

        if sam_outputs.pred_logits is not None:

            # Text matching confidence
            scores = torch.sigmoid(sam_outputs.pred_logits)

            # Account for scene-level presence confidence
            if sam_outputs.presence_logits is not None:
                presence = torch.sigmoid(
                    sam_outputs.presence_logits
                )
                scores = scores * presence

            num_selected = min(
                self.max_proposals,
                num_queries
            )

            top_scores, top_indices = torch.topk(
                scores,
                k=num_selected,
                dim=1
            )
            sam_masks = torch.gather(
                sam_masks,
                dim=1,
                index=top_indices[:, :, None, None].expand(
                    -1, -1,
                    sam_masks.shape[-2],
                    sam_masks.shape[-1]
                )
            )

        else:
            
            sam_masks = sam_masks[:, :self.max_proposals]

        sam_masks = F.interpolate(
            sam_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        semantic_logits = self.classifier(
            images,
            sam_masks
        )

        return semantic_logits






