import torch
import torch.nn as nn


class UNetMRF(nn.Module):
  def __init__(self, num_classes: int = 1, num_layers: int = 5, features_start: int = 64):
    super().__init__()
    self.num_layers = num_layers

    layers = [DoubleConvMRF(3, features_start)]

    feats = features_start
    for _ in range(num_layers - 1):
      layers.append(Down(feats, feats * 2))
      feats *= 2

    for _ in range(num_layers - 1):
      layers.append(Up(feats, feats // 2))
      feats //= 2

    layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

    self.layers = nn.ModuleList(layers)

  def forward(self, x):
    xi = [self.layers[0](x)]
    # Down path
    for layer in self.layers[1: self.num_layers]:
      xi.append(layer(xi[-1]))
    # Up path
    for i, layer in enumerate(self.layers[self.num_layers: -1]):
      xi[-1] = layer(xi[-1], xi[-2 - i])
    return torch.sigmoid(self.layers[-1](xi[-1]))


class DoubleConvMRF(nn.Module):

  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.shallow_path = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.SELU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.SELU(inplace=True),
    )

    self.wide_path = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
      nn.BatchNorm2d(out_ch),
      nn.SELU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),
      nn.BatchNorm2d(out_ch),
      nn.SELU(inplace=True),
    )

  def forward(self, x):
    return self.shallow_path(x) + self.wide_path(x)


class Down(nn.Module):

  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConvMRF(in_ch, out_ch))

  def forward(self, x):
    return self.net(x)


class Up(nn.Module):

  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()

    self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
    self.conv = DoubleConvMRF(in_ch, out_ch)

  def forward(self, x1, x2):
    x1 = self.upsample(x1)

    # Concatenate along the channels axis
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)
