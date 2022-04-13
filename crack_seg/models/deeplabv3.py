import torch as th
from torch import nn

from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabv3(nn.Module):

    def __init__(self, num_classes: int = 1):
        super(DeepLabv3, self).__init__()
        self._model = deeplabv3_resnet50(pretrained_backbone=True, num_classes=num_classes)

    def forward(self, x):
        return self._model(x)['out']
