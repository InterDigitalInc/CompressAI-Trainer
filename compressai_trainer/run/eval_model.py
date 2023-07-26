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

To evaluate a model trained using CompressAI Trainer:

.. code-block:: bash

    compressai-eval \
        --config-path="$HOME/data/runs/e4e6d4d5e5c59c69f3bd7be2/configs" \
        --config-name="config" \
        ++model.source="config" \
        ++paths.output_dir="outputs" \
        ++paths.model_checkpoint='${paths.checkpoints}/runner.last.pth'

To evaluate a model from the CompressAI zoo:

.. code-block:: bash

    compressai-eval --config-name="example_eval_zoo"

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

import catalyst
import catalyst.utils
import compressai.zoo.image
import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from compressai.registry import MODELS
from compressai.zoo import load_state_dict
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from compressai_trainer.config import (
    configure_conf,
    create_criterion,
    create_dataloaders,
    create_model,
    create_runner,
    state_dict_from_checkpoint,
)
from compressai_trainer.runners.image_compression import (
    RD_PLOT_DESCRIPTIONS,
    RD_PLOT_METRICS,
    RD_PLOT_SETTINGS_COMMON,
    RD_PLOT_TITLE,
    RdFigureLogger,
)
from compressai_trainer.typing import TModel, TRunner
from compressai_trainer.utils.metrics import compute_metrics, db
from compressai_trainer.utils.utils import tensor_to_np_img

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../../conf").resolve())


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


def load_model(conf: DictConfig) -> TModel:
    """Load a model from one of various sources.

    The source is determined by setting the config setting
    ``model.source`` to one of the following:

    - "config":
        Uses CompressAI Trainer standard config.
        (e.g. ``hp``, ``paths.model_checkpoint``, etc.)

    - "from_state_dict":
        Uses model's ``from_state_dict()`` factory method.
        Requires ``model.name`` and ``paths.model_checkpoint`` to be set.
        For example:

        .. code-block:: yaml

            model:
              name: "bmshj2018-factorized"
            paths:
              model_checkpoint: "/home/user/.cache/torch/hub/checkpoints/bmshj2018-factorized-prior-3-5c6f152b.pth.tar"

    - "zoo":
        Uses CompressAI's zoo of models.
        Requires ``model.name``, ``model.metric``, ``model.quality``,
        and ``model.pretrained`` to be set.
        For example:

        .. code-block:: yaml

            model:
              name: "bmshj2018-factorized"
              metric: "mse"
              quality: 3
              pretrained: True
    """
    source = conf.model.get("source", None)

    if source is None:
        raise ValueError(
            "Please override model.source with one of "
            '"config", "from_state_dict", or "zoo".\n'
            "\nExample: "
            '++model.source="config"'
        )

    if source is None or source == "config":
        if not conf.paths.model_checkpoint:
            raise ValueError(
                "Please override paths.model_checkpoint.\nExample: "
                "++paths.model_checkpoint='${paths.checkpoints}/runner.last.pth'"
            )
        return create_model(conf)

    if source == "from_state_dict":
        return load_checkpoint_from_state_dict(
            conf.model.name,
            conf.paths.model_checkpoint,
        ).to(conf.misc.device)

    if source == "zoo":
        return compressai.zoo.image._load_model(
            conf.model.name,
            conf.model.metric,
            conf.model.quality,
            conf.model.pretrained,
        ).to(conf.misc.device)


def load_checkpoint_from_state_dict(arch: str, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path)
    state_dict = state_dict_from_checkpoint(ckpt)
    state_dict = load_state_dict(state_dict)  # for pre-trained models
    model = MODELS[arch].from_state_dict(state_dict)
    return model


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


def _current_dataframe(conf, results):
    r = lambda x: float(f"{x:.4g}")
    d = {
        "name": conf.model.name + "*",
        "epoch": None,
        "criterion.lmbda": conf.criterion.lmbda,
        "loss": r(results["results_averaged"]["loss"]),
        "bpp": r(results["results_averaged"]["bpp"]),
        "psnr": r(results["results_averaged"]["psnr"]),
        "ms-ssim": r(results["results_averaged"]["ms-ssim"]),
        # NOTE: The dB of the mean of MS-SSIM samples
        # is not the same as the mean of MS-SSIM dB samples.
        "ms-ssim-db": r(db(1 - results["results_averaged"]["ms-ssim"])),
    }
    return pd.DataFrame.from_dict([d])


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


def _plot_rd(conf, results, metrics):
    for metric, description in zip(RD_PLOT_METRICS, RD_PLOT_DESCRIPTIONS):
        if metric not in metrics:
            continue

        fig = RdFigureLogger(runner=None).log(
            df=_current_dataframe(conf, results),
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


def write_results(conf, outputs, metrics):
    result_avg_keys = ["bpp", "loss", *metrics, "encoding_time", "decoding_time"]
    result_keys = ["filename", *result_avg_keys]
    results = {
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

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    table = [
        list(results["results_by_file"].keys()),
        *zip(*results["results_by_file"].values()),
    ]

    with open("results.tsv", "w") as f:
        for row in table:
            print("\t".join(f"{x}" for x in row), file=f)

    _plot_rd(conf, results, metrics)


@hydra.main(version_base=None, config_path=config_path)
def main(conf: DictConfig):
    runner = setup(conf)

    batches = runner.loaders["infer"]
    filenames = get_filenames(conf, len(batches))
    output_dir = conf.paths.output_dir
    metrics = ["psnr", "ms-ssim"]

    outputs = run_eval_model(runner, batches, filenames, output_dir, metrics)
    write_results(conf, outputs, metrics)


if __name__ == "__main__":
    main()
