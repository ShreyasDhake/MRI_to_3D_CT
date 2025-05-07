import torch
import torch.nn as nn
from segmentation_models_pytorch import Linknet

class Discriminative_model(nn.Module):
    def __init__(self):  
        super(Discriminative_model, self).__init__()
        self.model = Linknet(
            encoder_name="resnet50",      
            encoder_weights="imagenet",   
            in_channels=1,                
            classes=1                    
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    print("Discriminative_model loaded successfully!")

