# Copyright (c) 2021-2023, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""
Evaluate a model.

Evaluates a model on the configured ``dataset.infer`` (i.e. test set).
Saves bitstreams and reconstructed images to ``paths.output_dir``.
Computes metrics and saves per-file results and averaged results to
JSON and TSV files.

To evaluate models trained using CompressAI Trainer:

.. code-block:: bash

    compressai-eval \
        --config-path="$HOME/data/runs/e4e6d4d5e5c59c69f3bd7be2/configs" \
        --config-path="$HOME/data/runs/d4d5e5c5e4e6bd7be29c69f3/configs" \
        ...

To evaluate models from the CompressAI zoo:

.. code-block:: bash

    compressai-eval \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=1 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=2 \
        ...

By default, the following options are used, if not specified:

.. code-block:: bash

    --config-path="conf"
    --config-name="config"

    ++model.source="config"
    ++paths.output_dir="outputs"

    # if model.source == "config":
    ++paths.model_checkpoint='${paths.checkpoints}/runner.last.pth'

The model is evaluated on ``dataset.infer``, which may be configured as follows:

.. code-block:: yaml

    dataset:
      infer:
        type: "ImageFolder"
        config:
          root: "/path/to/directory/containing/images"
          split: ""
        loader:
          shuffle: False
          batch_size: 1
          num_workers: 2
        settings:
        transforms:
          - "ToTensor": {}
        meta:
          name: "Custom dataset"
          identifier: "image/custom"
          num_samples: 0  # ignored during eval

To evaluate a model on a custom directory of samples, use the above
config and override ``dataset.infer.config.root``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import catalyst
import catalyst.utils
import numpy as np
import plotly.graph_objects as go
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from compressai_trainer.config import (
    configure_conf,
    create_criterion,
    create_dataloaders,
    create_runner,
    load_model,
)
from compressai_trainer.runners.image_compression import (
    RD_PLOT_DESCRIPTIONS,
    RD_PLOT_METRICS,
    RD_PLOT_SETTINGS_COMMON,
    RD_PLOT_TITLE,
    RdFigureLogger,
)
from compressai_trainer.typing import TRunner
from compressai_trainer.utils.args import iter_configs
from compressai_trainer.utils.metrics import compute_metrics
from compressai_trainer.utils.utils import ld_to_dl, tensor_to_np_img

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../../conf").resolve())

DEFAULT_MODEL_SOURCE = "config"
DEFAULT_PATHS_MODEL_CHECKPOINT = "${paths.checkpoints}/runner.last.pth"
DEFAULT_PATHS_OUTPUT_DIR = "outputs"


def setup(conf: DictConfig) -> TRunner:
    catalyst.utils.set_global_seed(conf.misc.seed)
    catalyst.utils.prepare_cudnn(benchmark=False, deterministic=True)

    configure_conf(conf)

    model = load_model(conf).eval()
    criterion = create_criterion(conf.criterion)
    loaders = create_dataloaders(conf)

    runner = create_runner(conf.runner)
    runner.model = model
    runner.criterion = criterion
    runner.loaders = loaders
    runner.engine = catalyst.utils.get_available_engine()
    runner._hparams = OmegaConf.to_container(conf, resolve=True)

    return runner


def get_filenames(conf, num_files):
    if conf.dataset.infer.type == "ImageFolder":
        root = Path(conf.dataset.infer.config.root) / conf.dataset.infer.config.split
        return sorted(str(f.relative_to(root)) for f in root.iterdir())

    return [f"unknown_{i:06d}" for i in range(1, num_files + 1)]


def run_eval_model(runner, batches, filenames, output_dir, metrics):
    runner.model_module.update(force=True)

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    outputs = []

    for batch, filename in zip(batches, filenames):
        assert len(batch) == 1
        x = batch.to(runner.engine.device)
        out_infer = runner.predict_batch(x)
        out_criterion = runner.criterion(out_infer["out_net"], x)
        out_metrics = compute_metrics(x, out_infer["out_dec"]["x_hat"], metrics)

        output = {
            "filename": filename,
            "bpp": out_infer["bpp"],
            "loss": out_criterion["loss"].item(),
            **out_metrics,
            "encoding_time": out_infer["encoding_time"],
            "decoding_time": out_infer["decoding_time"],
        }
        outputs.append(output)
        print(json.dumps(output, indent=2))

        output_filename = (output_dir / filename).with_suffix("")
        os.makedirs(output_filename.parent, exist_ok=True)
        _write_bitstreams(out_infer["out_enc"]["strings"], output_filename)
        _write_image(out_infer["out_dec"]["x_hat"], output_filename)

    return outputs


def _write_bitstreams(strings, filename):
    for i, s in enumerate(strings):
        assert len(s) == 1
        with open(filename.with_suffix(f".{i:02d}.bin"), "wb") as f:
            f.write(s[0])


def _write_image(x, filename):
    assert len(x.shape) == 4 and x.shape[0] == 1
    Image.fromarray(tensor_to_np_img(x[0])).save(filename.with_suffix(".png"))


def _current_rd_traces(conf, results, metric):
    lmbda = conf.criterion.lmbda
    num_points = len(results["results_by_file"]["bpp"])
    samples_scatter = go.Scatter(
        x=results["results_by_file"]["bpp"],
        y=results["results_by_file"][metric],
        mode="markers",
        name=f"{conf.model.name} {lmbda:.4f}",
        text=[f"lmbda={lmbda:.4f}\nsample_idx={i}" for i in range(num_points)],
        visible="legendonly",
    )
    return [samples_scatter]


def _plot_rd(runner, conf, results, metrics):
    runner_ = SimpleNamespace(
        hparams=conf,
        epoch_step=None,
        loader_metrics=results["results_averaged"],
    )

    for metric, description in zip(RD_PLOT_METRICS, RD_PLOT_DESCRIPTIONS):
        if metric not in metrics:
            continue

        fig = RdFigureLogger(runner=None).log(
            df=runner.__class__._current_dataframe.fget(runner_),
            traces=_current_rd_traces(conf, results, metric),
            metric=metric,
            dataset=conf.dataset.infer.meta.identifier,
            **RD_PLOT_SETTINGS_COMMON,
            layout_kwargs=dict(
                title=RD_PLOT_TITLE.format(
                    dataset=conf.dataset.infer.meta.name,
                    metric=description,
                )
            ),
            log_figure=False,
        )

        fig.write_html(f"rd-curves-{metric}.html")


def _results_dict(conf, outputs, metrics):
    result_avg_keys = ["bpp", "loss", *metrics, "encoding_time", "decoding_time"]
    result_keys = ["filename", *result_avg_keys]
    return {
        "name": conf.model.name,
        "description": "",
        "dataset": conf.dataset.infer.meta.name,
        "results_averaged": {
            **{k: np.mean([out[k] for out in outputs]) for k in result_avg_keys},
        },
        "results_by_file": {
            **{k: [out[k] for out in outputs] for k in result_keys},
        },
    }


def write_results(conf, outputs, metrics):
    results = _results_dict(conf, outputs, metrics)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    table = [
        list(results["results_by_file"].keys()),
        *zip(*results["results_by_file"].values()),
    ]

    with open("results.tsv", "w") as f:
        for row in table:
            print("\t".join(f"{x}" for x in row), file=f)

    return results


def prepare_conf(conf):
    if "source" not in conf.get("model", {}):
        with open_dict(conf):
            conf.model.source = DEFAULT_MODEL_SOURCE

    if (conf.model.source == "config") and (
        conf.get("paths", {}).get("model_checkpoint", None) is None
    ):
        with open_dict(conf):
            conf.paths.model_checkpoint = DEFAULT_PATHS_MODEL_CHECKPOINT

    if "output_dir" not in conf.get("paths", {}):
        with open_dict(conf):
            conf.paths.output_dir = DEFAULT_PATHS_OUTPUT_DIR


def main():
    results_avg = []

    for conf in iter_configs(start=thisdir):
        prepare_conf(conf)
        runner = setup(conf)

        batches = runner.loaders["infer"]
        filenames = get_filenames(conf, len(batches))
        output_dir = Path(conf.paths.output_dir)
        metrics = ["psnr", "ms-ssim"]

        outputs = run_eval_model(runner, batches, filenames, output_dir, metrics)
        results = write_results(conf, outputs, metrics)
        _plot_rd(runner, conf, results, metrics)
        results_avg.append(results["results_averaged"])

    results_avg = ld_to_dl(results_avg)
    print(json.dumps(results_avg, indent=2))


if __name__ == "__main__":
    main()
