import typer

import torch as th
from torch import nn

from crack_seg.models import SegModel
from time import perf_counter_ns
from prettytable import PrettyTable


def get_num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_latencies(module: nn.Module, size: int = 448) -> dict[str, float]:
    x = th.rand(1, 3, size, size)

    # speed up
    _ = th.rand(5000, 5000) @ th.rand(5000, 5000)

    with th.no_grad():
        start = perf_counter_ns()
        _ = module(x)
        elapsed = perf_counter_ns() - start

    return elapsed * 1e-6


def main(
) -> None:
    results = []

    for model in SegModel:
        net = SegModel.factory(model)(num_classes=1)
        net.eval()

        n_params = f'{round(get_num_params(net) / 1e6)}M'
        latency = f'{get_latencies(net):.3f}ms'

        results.append([
            model,
            n_params,
            latency
        ])

    pt_results = PrettyTable()
    pt_results.field_names = ['model', '# params', 'latency (batch=1)']
    pt_results.add_rows(results)

    print(pt_results)


if __name__ == '__main__':
    typer.run(main)
