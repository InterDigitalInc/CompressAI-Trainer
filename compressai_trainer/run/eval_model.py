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
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=1 ++criterion.lmbda=0.0018 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=2 ++criterion.lmbda=0.0035 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=3 ++criterion.lmbda=0.0067 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=4 ++criterion.lmbda=0.0130 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=5 ++criterion.lmbda=0.0250 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=6 ++criterion.lmbda=0.0483 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=7 ++criterion.lmbda=0.0932 \
        --config-name="eval_zoo" ++model.name="bmshj2018-factorized" ++model.quality=8 ++criterion.lmbda=0.1800

By default, the following options are used, if not specified:

.. code-block:: bash

    --config-path="conf"
    --config-name="config"

    ++model.source="config"

    # if model.source == "config":
    ++paths.output_dir="outputs/${model.source}-${env.aim.run_hash}-${model.name}"
    ++paths.model_checkpoint='${paths.checkpoints}/runner.last.pth'

    # if model.source == "from_state_dict":
    ++paths.output_dir="outputs/${model.name}-${Path(paths.model_checkpoint).stem}"

    # if model.source == "zoo":
    ++paths.output_dir="outputs/${model.source}-${model.name}-${model.metric}-${model.quality}"

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
from compressai_trainer.utils.metrics import _METRICS, compute_metrics
from compressai_trainer.utils.utils import ld_to_dl, tensor_to_np_img

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../../conf").resolve())

DEFAULT_MODEL_SOURCE = "config"
DEFAULT_PATHS_MODEL_CHECKPOINT = "${paths.checkpoints}/runner.last.pth"
DEFAULT_PATHS_OUTPUT_DIR_ROOT = "outputs"


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


def _get_filenames(runner, num_files):
    dataset = runner.loaders["infer"].dataset

    if type(dataset).__name__ == "ImageFolder":
        return [x.stem for x in dataset.samples]

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
            **{
                k: v.item()
                for k, v in out_criterion.items()
                if k in runner.hparams["runner"]["meters"]["infer"]
            },
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

        fig.write_html(f"{conf.paths.output_dir}/rd-curves-{metric}.html")


def _results_dict(conf, outputs):
    result_keys = list(outputs[0].keys())
    result_non_avg_keys = ["filename"]
    result_avg_keys = [k for k in result_keys if k not in result_non_avg_keys]

    return {
        "name": conf.model.name,
        "description": "",
        "meta": {
            "dataset": conf.dataset.infer.meta.name,
            "env.aim.run_hash": conf.env.aim.run_hash,
            "misc.device": conf.misc.device,
            "model.source": conf.model.get("source"),
            "model.name": conf.model.get("name"),
            "model.metric": conf.model.get("metric"),
            "model.quality": conf.model.get("quality"),
            "criterion.lmbda": conf.criterion.get("lmbda"),
            "paths.model_checkpoint": conf.paths.get("model_checkpoint"),
        },
        "results_averaged": {
            **{k: np.mean([out[k] for out in outputs]) for k in result_avg_keys},
        },
        "results_by_file": {
            **{k: [out[k] for out in outputs] for k in result_keys},
        },
    }


def _write_results(conf, results):
    with open(f"{conf.paths.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    table = [
        list(results["results_by_file"].keys()),
        *zip(*results["results_by_file"].values()),
    ]

    with open(f"{conf.paths.output_dir}/results.tsv", "w") as f:
        _write_tsv(table, file=f)


def _write_results_final(results_list):
    meta_common = {
        key: _get_common_value(results_list, ("meta", key))
        for key in results_list[0]["meta"]
        if _is_common_value(results_list, ("meta", key))
    }

    meta_noncommon = {
        key: [_get_value(results, ("meta", key)) for results in results_list]
        for key in results_list[0]["meta"]
        if not _is_common_value(results_list, ("meta", key))
    }

    results_averaged = ld_to_dl(
        [results["results_averaged"] for results in results_list]
    )

    results = {
        "name": _get_common_value(results_list, ("name",)),
        "description": _get_common_value(results_list, ("description",)),
        "meta": meta_common,
        "meta_": meta_noncommon,
        "results": results_averaged,
    }

    with open(f"{DEFAULT_PATHS_OUTPUT_DIR_ROOT}/results.json", "w") as f:
        json.dump(results, f, indent=2)


def _write_tsv(rows, file):
    for row in rows:
        print("\t".join(f"{x}" for x in row), file=file)


def _is_common_value(ds, path):
    value = _get_value(ds[0], path)
    return all(_get_value(d, path) == value for d in ds)


def _get_common_value(ds, path):
    value = _get_value(ds[0], path)
    assert all(_get_value(d, path) == value for d in ds)
    return value


def _get_value(d, path):
    for key in path:
        d = d[key]
    return d


def _prepare_conf(conf):
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
            conf.paths.output_dir = _get_output_dir(conf)


def _get_output_dir(conf):
    source = conf.model.source
    if conf.paths.get("output_dir") is not None:
        return conf.paths.output_dir
    if source == "config":
        return (
            f"{DEFAULT_PATHS_OUTPUT_DIR_ROOT}/"
            "${model.source}-${env.aim.run_hash}-${model.name}"
        )
    if source == "from_state_dict":
        return (
            f"{DEFAULT_PATHS_OUTPUT_DIR_ROOT}/"
            "${model.name}-"
            f"{Path(conf.paths.model_checkpoint).stem}"
        )
    if source == "zoo":
        return (
            f"{DEFAULT_PATHS_OUTPUT_DIR_ROOT}/"
            "${model.source}-${model.name}-${model.metric}-${model.quality}"
        )
    raise ValueError(f"Unknown model.source: {source}")


def main():
    results_list = []

    for conf in iter_configs(start=thisdir):
        _prepare_conf(conf)
        runner = setup(conf)

        batches = runner.loaders["infer"]
        filenames = _get_filenames(runner, len(batches))
        output_dir = Path(conf.paths.output_dir)
        metrics = [x for x in conf.runner.meters.infer if x in _METRICS]

        outputs = run_eval_model(runner, batches, filenames, output_dir, metrics)
        results = _results_dict(conf, outputs)
        results_list.append(results)
        _write_results(conf, results)
        _plot_rd(runner, conf, results, metrics)

    _write_results_final(results_list)


if __name__ == "__main__":
    main()
