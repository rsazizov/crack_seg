import typer
import cv2

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from crack_seg.utils import create_augmented_transform, to_numpy_channels, NormalizeInverse


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


def main(
        img: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to image'),
        mask: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to mask'),
        size: Optional[int] = typer.Option(448, help='Input size')
) -> None:
    aug = create_augmented_transform(size)

    img = cv2.imread(str(img))
    mask = cv2.imread(str(mask))

    print(img.shape, mask.shape)

    img_aug, mask_aug = aug(image=img, mask=mask).values()

    img_aug = NormalizeInverse()(img_aug)

    print(img_aug.shape, mask_aug.shape)

    visualize(to_numpy_channels(img_aug), mask_aug, img, mask)
    plt.show()


if __name__ == '__main__':
    typer.run(main)
