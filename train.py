import typer

from pathlib import Path
from typing import Optional
from pprint import pprint

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from torchmetrics import Precision, Recall, F1Score, JaccardIndex, MetricCollection

import torch as th
import datetime

from crack_seg.utils import SegmentationDataset, create_transform, create_augmented_transform, threshold_mask, \
    NormalizeInverse, dice_loss
from crack_seg.models import SegModel, UNetMRF, Loss, UNet

from torchvision.models.segmentation import deeplabv3_resnet50


class LitSegmentationModel(LightningModule):

    def __init__(self, model: SegModel, lr: float = 3e-4, loss=th.nn.CrossEntropyLoss()):
        super(LitSegmentationModel, self).__init__()

        self.lr = lr

        self.model = {
            SegModel.unet: UNet(num_classes=1),
            SegModel.deeplabv3: deeplabv3_resnet50(num_classes=1),
            SegModel.unet_mrf: UNetMRF(num_classes=1)
        }[model]

        self.loss = {
            Loss.crossentropy: th.nn.CrossEntropyLoss(),
            Loss.dice: dice_loss
        }[loss]

        self.train_f1 = F1Score(num_classes=1, multiclass=False)
        self.train_miou = JaccardIndex(num_classes=2)

        self.valid_f1 = F1Score(num_classes=1, multiclass=False)
        self.valid_miou = JaccardIndex(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.lr)

    def get_loss_and_metrics(self, x, y_hat, y, prefix=''):
        loss = self.loss(th.flatten(y_hat, start_dim=1), th.flatten(y, start_dim=1))

        preds = th.flatten(threshold_mask(y_hat))
        targets = th.flatten(y.to(th.int))

        if prefix == 'Train/':
            self.train_f1(preds, targets)
            self.train_miou(preds, targets)
        else:
            self.valid_f1(preds, targets)
            self.valid_miou(preds, targets)

        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)

        loss = self.get_loss_and_metrics(x, y_hat, y, 'Train/')

        self.log('Train/Loss', loss, on_epoch=True, on_step=False)
        self.log('Train/F1', self.train_f1, on_epoch=True, on_step=False)
        self.log('Train/mIoU', self.train_miou, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.get_loss_and_metrics(x, y_hat, y, 'Valid/')

        self.log('Valid/Loss', loss, on_epoch=True, on_step=False)
        self.log('Valid/F1', self.valid_f1, on_epoch=True, on_step=False)
        self.log('Valid/mIoU', self.valid_miou, on_epoch=True, on_step=False)


def main(
        data: Path = typer.Option(..., exists=True, file_okay=False, help='Path to train dataset'),
        model: SegModel = typer.Option(..., help='Model to train'),
        loss: Optional[Loss] = typer.Option('crossentropy', help='Loss function to optimize'),
        valid_frac: Optional[float] = typer.Option(0.3, help='Fraction of validation set'),
        cuda: Optional[bool] = typer.Option(False, help='Use CUDA'),
        size: Optional[int] = typer.Option(512, help='Input size'),
        lr_scheduler: Optional[bool] = typer.Option(False, help='Use cosine annealing LR scheduler'),
        half: Optional[bool] = typer.Option(False, help='Use half precision'),
        lr: Optional[float] = typer.Option(3e-4, help='Learning rate'),
        augment: Optional[bool] = typer.Option(False, help='Apply augmentations'),
        epochs: Optional[int] = typer.Option(40, help='Number of epochs'),
        bs: Optional[int] = typer.Option(4, help='Batch size'),
        workers: Optional[int] = typer.Option(2, help='Number of workers for DataLoader'),
        log_root: Optional[Path] = typer.Option('runs', exists=False, help='Path to checkpoints and logs'),
        run: Optional[str] = typer.Option(None, help='Name of run (used for logging)'),
        log: Optional[bool] = typer.Option(False, help='Log to W&B'),
        project: Optional[str] = typer.Option('crack_seg', help='W&B project name'),
        seed: Optional[int] = typer.Option(0, help='Random seed'),
):
    th.manual_seed(seed)

    if not run:
        now = datetime.datetime.now()
        run = now.strftime("%m_%d_%Y_%H_%M_%S")

    if log:
        wandb_logger = WandbLogger(project=project, name=run)
        wandb_logger.experiment.config.update({
            'data': data,
            'model': model,
            'valid_frac': valid_frac,
            'cuda': cuda,
            'size': size,
            'lr_scheduler': lr_scheduler,
            'half': half,
            'lr': lr,
            'augment': augment,
            'epochs': epochs,
            'log_root': log_root,
            'run': run,
            'log': log,
            'seed': seed
        })

    run_dir = log_root / Path(run)

    train_valid_ds = SegmentationDataset(data, transform=None)
    train_valid_ds_len = len(train_valid_ds)
    valid_ds_len = int(train_valid_ds_len * valid_frac)

    valid_ds, train_ds = th.utils.data.random_split(train_valid_ds, [
        valid_ds_len, train_valid_ds_len - valid_ds_len
    ])

    train_ds.dataset = SegmentationDataset(data,
                                           transform=create_augmented_transform(size) if augment else create_transform(
                                               size))
    valid_ds.dataset.transform = create_transform(size)

    print(f'Train/Valid split: {len(train_ds)}/{len(valid_ds)}')

    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir,
            monitor='Valid/Loss',
            save_top_k=5
        ),
    ]

    wandb_logger = WandbLogger()

    trainer = Trainer(
        max_epochs=epochs,
        gpus=0,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=16 if half else 32
    )

    seg_module = LitSegmentationModel(model, lr, loss)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers)
    valid_dl = DataLoader(valid_ds, batch_size=bs, num_workers=workers)

    trainer.fit(seg_module, train_dl, valid_dl)


if __name__ == '__main__':
    typer.run(main)
