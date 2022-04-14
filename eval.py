import typer
from pathlib import Path
from pprint import pprint
from typing import Optional

import torch as th

from torch.utils.data import DataLoader

from crack_seg.models import SegModel
from crack_seg.utils import threshold_mask, SegmentationDataset, create_transform
from torchmetrics import Precision, Recall, F1Score, JaccardIndex, MetricCollection

from tqdm.auto import tqdm


def main(
        data: Path = typer.Option(..., exists=True, file_okay=False, help='Path to test dataset'),
        model: SegModel = typer.Option(..., help='Model to eval'),
        weights: Path = typer.Option(..., exists=True, dir_okay=False, help='Path to weights'),
        size: int = typer.Option(448, help='Use cuda'),
        bs: int = typer.Option(4, help='Batch size'),
        cuda: Optional[bool] = typer.Option(False, help='Use cuda')
) -> None:
    device = 'cuda' if cuda else 'cpu'

    state_dict = th.load(weights, map_location=th.device(device))['state_dict']
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model = SegModel.factory(model)(num_classes=1).to(device).eval()

    model.load_state_dict(state_dict)

    print(f'Loaded state dict: {weights}')

    test_df = SegmentationDataset(data, transform=create_transform(size))
    test_dl = DataLoader(test_df, batch_size=bs)

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
