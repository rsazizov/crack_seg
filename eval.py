import typer
from pathlib import Path
from pprint import pprint

import torch as th

from torch.utils.data import DataLoader

from crack_seg import UNet
from data import threshold_mask, SegmentationDataset, create_transform
from torchmetrics import Precision, Recall, F1Score, JaccardIndex, MetricCollection

from tqdm.auto import tqdm


def main(
    data: Path = typer.Option(..., exists=True, file_okay=False, help='Path to test dataset'),
    weights: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to weights'),
    cuda: bool = typer.Option(False, help='Use cuda')
) -> None:
  device = 'cuda' if cuda else 'cpu'

  state_dict = th.load(weights, map_location=th.device(device))

  model = UNet(num_classes=1).to(device).eval()

  model.load_state_dict(state_dict)

  print(f'Loaded state dict: {weights}')

  test_df = SegmentationDataset(data, transform=create_transform(512))
  test_dl = DataLoader(test_df, batch_size=2)

  batches = tqdm(enumerate(test_dl), desc=f'Validation', unit='batch', total=len(test_dl))

  metrics = MetricCollection([
    Precision(num_classes=2, average=None),
    Recall(num_classes=2, average=None),
    F1Score(num_classes=2, average=None),
  ]).to(device)

  iou = JaccardIndex(num_classes=2).to(device)

  for i, batch in batches:
    x, y = batch

    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)

    preds = th.flatten(threshold_mask(y_hat))
    targets = th.flatten(y.to(th.int)).clone()

    metrics(preds, targets)
    iou(preds, targets)

  metrics_log = {k: v[1] for k, v in metrics.compute().items()}

  metrics_log.update({
    'mIoU': iou.compute(),
    **metrics_log
  })

  pprint(metrics_log)


if __name__ == '__main__':
  typer.run(main)
