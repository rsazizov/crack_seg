from pathlib import Path
from typing import Tuple
from itertools import chain

import torch as th
import torchvision.transforms as T
import cv2
import imageio

from torchvision.utils import draw_segmentation_masks
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# From: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
  Note that PyTorch optimizers minimize a loss. In this
  case, we would like to maximize the dice loss so we
  return the negated dice loss.
  Args:
      true: a tensor of shape [B, 1, H, W].
      logits: a tensor of shape [B, C, H, W]. Corresponds to
          the raw output or logits of the model.
      eps: added to the denominator for numerical stability.
  Returns:
      dice_loss: the Sørensen–Dice loss.
  """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = th.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = th.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = th.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = th.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = th.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = th.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = th.sum(probas * true_1_hot, dims)
    cardinality = th.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class NormalizeInverse(T.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = th.as_tensor(mean)
        std = th.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def create_augmented_transform(size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(size),
        A.OneOf([
            A.RandomSizedCrop(min_max_height=(128, 256), height=size, width=size, p=0.5),
            A.PadIfNeeded(min_height=size, min_width=size, p=0.5)
        ], p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8),
        A.RandomShadow(),
        ToTensorV2()])


def create_transform(size: int) -> A.Compose:
    return A.Compose([
        A.SmallestMaxSize(size),
        A.CenterCrop(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


SMOOTH = 1e-6


def load_image(path: Path, size: int = 448, transform: bool = True) -> th.Tensor:
    img = cv2.imread(str(path))

    if transform:
        img = create_transform(size)(image=img)

    return img


def threshold_mask(mask: th.Tensor, threshold: float = 0.5) -> th.Tensor:
    return (mask >= threshold).to(int)


def to_numpy_channels(img: th.Tensor) -> th.Tensor:
    return img.permute(1, 2, 0)


def burn_mask(img: th.Tensor, mask: th.Tensor, path: Path = None):
    img = (img.squeeze() * 255).to(th.uint8)
    mask = (mask.squeeze()).to(th.bool)

    out = draw_segmentation_masks(img, mask, colors=['red'])

    if path:
        imageio.imsave(path, to_numpy_channels(out), 'png')
    else:
        return out


def iou(outputs: th.Tensor, labels: th.Tensor) -> th.Tensor:
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = th.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


class SegmentationDataset(Dataset):
    """
    Dataset structure:

    dataset/
        images/
            000.jpg
            001.jpg
            ...
        labels/
            000.jpg
            001.jpg
            ...
    """

    def __init__(self, root: Path, images_dir: str = 'images', labels_dir: str = 'labels',
                 transform=None):
        self.root = root
        self.images_dir = self.root / images_dir
        self.labels_dir = self.root / labels_dir
        self.transform = transform

        # jpg or png
        self.images = list(chain(
            self.images_dir.rglob('*.png'),
            self.images_dir.rglob('*.jpg'),
        ))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[th.Tensor, th.Tensor]:
        img_path = self.images[idx]

        image = cv2.imread(str(img_path))
        label = cv2.imread(str(next(self.labels_dir.rglob(img_path.stem + '.*'))), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image, label = self.transform(image=image, mask=label).values()

        return image / 255.0, label / 255.0
