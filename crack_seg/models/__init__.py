from crack_seg.models.unet_mrf import UNetMRF
from crack_seg.models.unet import UNet
from crack_seg.models.deeplabv3 import DeepLabv3

from enum import Enum


class SegModel(str, Enum):
    unet = 'unet'
    deeplabv3 = 'deeplabv3'
    unet_mrf = 'unet-mrf'

    @staticmethod
    def factory(model: 'SegModel'):
        return {
            SegModel.unet: UNet,
            SegModel.deeplabv3: DeepLabv3,
            SegModel.unet_mrf: UNetMRF
        }[model]


class Loss(str, Enum):
    crossentropy = 'crossentropy'
    dice = 'dice'
