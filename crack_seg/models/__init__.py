from crack_seg.models.unet_mrf import UNetMRF
from crack_seg.models.unet import UNet

from enum import Enum


class SegModel(str, Enum):
  unet = 'unet'
  deeplabv3 = 'deeplabv3'
  unet_mrf = 'unet-mrf'


class Loss(str, Enum):
  crossentropy = 'crossentropy'
  dice = 'dice'
