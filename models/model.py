import torch
import torch.nn as nn
import timm

class DRNet(nn.Module):
    def __init__(self, model_name='mobilenetv3_small_100', num_classes=5, pretrained=True):
        super(DRNet, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_model(model_name='mobilenetv3_small_100', num_classes=5, pretrained=True):
    return DRNet(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
