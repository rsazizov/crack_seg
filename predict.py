import typer

from pathlib import Path
from typing import Optional
from PIL import Image

import numpy as np

import torch as th
import matplotlib.pyplot as plt

from crack_seg.models import SegModel
from crack_seg.utils import load_image, threshold_mask, to_numpy_channels, NormalizeInverse


def main(
        img: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to image'),
        model: SegModel = typer.Option(..., help='Model to eval'),
        weights: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to weights'),
        size: Optional[int] = typer.Option(448, exists=True, dir_okay=False, help='Image size'),
        show: Optional[bool] = typer.Option(True, exists=True, dir_okay=False, help='Show results'),
        save: Optional[bool] = typer.Option(False, exists=True, dir_okay=False, help='Save result')
) -> None:
    device = 'cpu'

    state_dict = th.load(weights, map_location=th.device(device))['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model = SegModel.factory(model)(num_classes=1)

    model.load_state_dict(state_dict)
    # model.eval()

    print(f'Loaded state dict: {weights}')

    img_input = load_image(img, size=size)['image']

    norm_inv = NormalizeInverse()

    with th.no_grad():
        y = model(img_input.unsqueeze(0))
        # y = th.sigmoid(y)
        print(y.mean())

        mask = threshold_mask(y).squeeze().cpu()
        mask = (to_numpy_channels(mask.repeat(3, 1, 1)).numpy() * 255).astype(np.uint8)

        if show:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

            ax[0].imshow(to_numpy_channels(norm_inv(img_input)))
            ax[0].set_title('Image')

            ax[1].imshow(mask)
            ax[1].set_title('Mask')

            plt.show()

        if save:
            im = Image.fromarray(mask)
            im.save(str(img.with_suffix('.pred.png')))


if __name__ == '__main__':
    typer.run(main)
