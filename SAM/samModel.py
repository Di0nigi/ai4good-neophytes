import torch
from torch import nn
import numpy as np
import matplotlib

from pathlib import Path


import os
os.environ["HF_TOKEN"] = ""


from PIL import Image
from transformers import Sam3Model, Sam3Processor




class CustomSam(nn.Module):
    def __init__(self):

        super().__init__()

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
      
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.mask_decoder.parameters():
            p.requires_grad = True

        for p in self.model.detr_decoder.parameters():
            p.requires_grad=True

        for p in self.model.dot_product_scoring.parameters():
            p.requires_grad = True

        return

    @torch.inference_mode()
    def runInference(self, image, prompt=None):

        self.model.eval()

        with Image.open(image) as source:
            image = source.convert("RGB")

        inputs = self.processor(
            images=image,
            text=self.prompt if prompt is None else prompt,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        return result["masks"]
    

    def overlay_masks(self,image, masks):
        image = image.convert("RGBA")
        masks = 255 * masks.cpu().numpy().astype(np.uint8)
        
        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        return image

    def forward(self,data):

        output = self.model(data)

        return output

    def trainModel(self,data,optimizer,loss,epochs=10, lr=0.001):

        train = data[0] 
        val = data[1]

        self.optim = optimizer

        self.loss = loss

        trainLossList, trainAccList = [], []
        valLossList, valAccList = [], []

        for ep in range(epochs):

            print(f"Epoch [{ep+1}/{epochs}]")

            self.train()
            totalLoss= 0.0
            correctT = 0
            totalT = 0
            for batch in train:
                #im = batch1
                #v,_=batch2
                #e,_ = batch3

                x , t , p = batch

                #t = t.to(self.dev)
                #v = v.to(self.dev)
                #e = e.to(self.dev)

                #graph=(v,e)
                y = self.forward(x,p)
                
                #t = t.squeeze(1)
                #t = t.argmax(dim=1).long() 
                #print(t.shape)
                #print(f"shape t {y.shape}") 
                #print(f"shape t {t.shape}") 
                loss = self.loss(y,t)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                #print(loss.item())
                totalLoss += loss.item()
                #print(f"Train loss: {loss.item():.4f}")

                preds = y.argmax(dim=1)
                correctT += (preds == t).sum().item()
                totalT += t.numel()

            trainAccuracy = correctT / max(totalT, 1)
            totalLoss= totalLoss/totalT
            trainLossList.append(totalLoss)
            trainAccList.append(trainAccuracy)


            print(f"Train Loss: {totalLoss:.4f}, Train Accuracy: {trainAccuracy*100:.4f}%")

            '''# test loop
            self.eval()

            totalEvalLoss = 0.0
            correct = 0
            total = 0
            targetList=[]
            predList=[]

            with torch.no_grad():
                for batch1,batch2,batch3 in zip(testIm,testVe,testEd):

                    im, t = batch1
                    v, _ = batch2
                    e, _ = batch3

                    im = im.to(self.dev)
                    t = t.to(self.dev)
                    v = v.to(self.dev)
                    e = e.to(self.dev)

                    graph = (v, e)

                    y = self.forward([im, graph])
                    #t = t.view_as(y) 
                    #t = t.view(-1).long()
                    t = t.squeeze(1)
                    t = t.argmax(dim=1).long() 
                    loss = self.loss(y, t)
                    totalEvalLoss += loss.item()

                    # Classification: accuracy
                    predList.append(y)
                    targetList.append(t)

                    preds = y.argmax(dim=1)
                    
                    correct += (preds == t).sum().item()
                    total += t.numel()

            avgEvalLoss = totalEvalLoss / max(len(testIm), 1)
            
            accuracy = correct / max(total, 1)
            valLossList.append(avgEvalLoss)
            valAccList.append(accuracy)

            print(f"Val Loss: {avgEvalLoss:.4f}, Val Accuracy: {accuracy*100:.4f}%\n")

            if avgEvalLoss < best_val_loss:
                best_val_loss = avgEvalLoss
                epochs_no_improve = 0
                best_model_state = self.state_dict()
                self.bestVal = accuracy*100  
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                    self.load_state_dict(best_model_state)  
                    break
        #self.bestVal = best_val_loss'''
        return  [trainLossList, trainAccList , valLossList, valAccList]#, [predList, targetList]




        return


def main():
    return

main()

