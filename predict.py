import typer
from pathlib import Path

import torch as th
import matplotlib.pyplot as plt

from crack_seg import UNet
from data import load_image, threshold_mask, to_numpy_channels, NormalizeInverse


def main(
    path: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to image'),
    weights: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to weights')
) -> None:
  state_dict = th.load(weights, map_location=th.device('cpu'))

  model = UNet(num_classes=1)

  model.load_state_dict(state_dict)
  model.eval()

  print(f'Loaded state dict: {weights}')

  img = load_image(path)

  with th.no_grad():
    y = model(img.unsqueeze(0))
    mask = threshold_mask(y).squeeze().cpu()
    mask = to_numpy_channels(mask.repeat(3, 1, 1)).numpy()

    img = NormalizeInverse((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    plt.imshow(to_numpy_channels(img))
    plt.show()

    plt.imshow((mask * 255).astype(int))
    plt.show()


if __name__ == '__main__':
  typer.run(main)
