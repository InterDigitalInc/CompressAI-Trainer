# Copyright (c) 2021-2022, InterDigital Communications, Inc
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import aim
import numpy as np
from aim.storage.object import CustomObject
from aim.storage.types import BLOB
from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class AimLogger(ILogger):
    """Aim logger for parameters, metrics, images and other artifacts.

    Aim documentation: https://aimstack.readthedocs.io/en/latest/.

    Args:
        experiment: Name of the experiment in Aim to log to.
        run_hash: Run hash.
        exclude: Name of key to exclude from logging.
        log_batch_metrics: boolean flag to log batch metrics
            (default: SETTINGS.log_batch_metrics or False).
        log_epoch_metrics: boolean flag to log epoch metrics
            (default: SETTINGS.log_epoch_metrics or True).
        repo: Aim repo object.
        run: Aim run object.
            If specified, `experiment`, `run_hash`, and `repo` are ignored.

    Python API examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            ...,
            loggers={"aim": dl.AimLogger(experiment="test_exp")}
        )

    .. code-block:: python

        from catalyst import dl

        class CustomRunner(dl.IRunner):
            # ...

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "aim": dl.AimLogger(experiment="test_exp")
                }

            # ...

        runner = CustomRunner().run()
    """

    exclude: List[str]
    run: aim.Run

    def __init__(
        self,
        *,
        experiment: Optional[str] = None,
        run_hash: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        log_batch_metrics: bool = SETTINGS.log_batch_metrics,
        log_epoch_metrics: bool = SETTINGS.log_epoch_metrics,
        repo: Optional[Union[str, aim.Repo]] = None,
        run: Optional[aim.Run] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            log_batch_metrics=log_batch_metrics,
            log_epoch_metrics=log_epoch_metrics,
        )
        self.exclude = [] if exclude is None else exclude
        self.run = (
            run
            if run is not None
            else aim.Run(
                run_hash=run_hash,
                repo=repo,
                experiment=experiment,
                **kwargs,
            )
        )

    @property
    def logger(self):
        """Internal logger/experiment/etc. from the monitoring system."""
        return self.run

    def log_artifact(
        self,
        tag: str,
        runner: "IRunner",
        artifact: object = None,
        path_to_artifact: Optional[str] = None,
        scope: Optional[str] = None,
        kind: str = "text",
        **kwargs,
    ) -> None:
        """Logs a local file or directory as an artifact to the logger."""
        if path_to_artifact:
            mode = "r" if kind == "text" else "rb"
            with open(path_to_artifact, mode) as f:
                artifact = f.read()
        kind_dict = {
            "audio": aim.Audio,
            "figure": aim.Figure,
            "image": aim.Image,
            "text": aim.Text,
        }
        value = kind_dict[kind](artifact, **kwargs)
        context, kwargs = _aim_context(runner, scope)
        self.run.track(value, tag, context=context, **kwargs)

    def log_image(
        self,
        tag: str,
        image,
        runner: "IRunner",
        scope: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Logs image to Aim for current scope on current step."""
        value = aim.Image(image, **kwargs)
        context, kwargs = _aim_context(runner, scope)
        self.run.track(value, tag, context=context, **kwargs)

    def log_hparams(self, hparams: Dict, runner: "Optional[IRunner]" = None) -> None:
        """Logs parameters for current scope.

        Args:
            hparams: Parameters to log.
            runner: experiment runner
        """
        d: dict[str, Any] = {}
        _build_params_dict(hparams, d, self.exclude)
        for k, v in d.items():
            self.run[k] = v

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str,
        runner: "IRunner",
    ) -> None:
        """Logs batch and epoch metrics to Aim."""
        if scope == "batch" and self.log_batch_metrics:
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics,
                runner=runner,
                loader_key=runner.loader_key,
                scope=scope,
            )
        elif scope == "epoch" and self.log_epoch_metrics:
            for loader_key, per_loader_metrics in metrics.items():
                self._log_metrics(
                    metrics=per_loader_metrics,  # type: ignore
                    runner=runner,
                    loader_key=loader_key,
                    scope=scope,
                )

    def log_distribution(
        self,
        tag: str,
        unused: Any,
        runner: "IRunner",
        scope: Optional[str] = None,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Logs distribution to Aim for current scope on current step."""
        assert unused is None
        value = aim.Distribution(**kwargs)
        context_default, kwargs = _aim_context(runner, scope)
        context = context_default if context is None else context
        self.run.track(value, tag, context=context, **kwargs)

    def log_figure(
        self,
        tag: str,
        fig: Any,
        runner: "IRunner",
        scope: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Logs figure to Aim for current scope on current step."""
        value = aim.Figure(fig, **kwargs)
        context, kwargs = _aim_context(runner, scope)
        self.run.track(value, tag, context=context, **kwargs)

    def close_log(self) -> None:
        """End an active Aim run."""
        self.run.close()

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        runner: "IRunner",
        loader_key: str,
        scope: str = "",
    ):
        context, kwargs = _aim_context(runner, scope, loader_key)
        for key, value in metrics.items():
            self.run.track(value, key, context=context, **kwargs)


def _aim_context(
    runner: "IRunner",
    scope: Optional[str],
    loader_key: Optional[str] = None,
    all_scope_steps: bool = False,
):
    if loader_key is None:
        loader_key = runner.loader_key
    context = {}
    if loader_key is not None:
        context["loader"] = loader_key
    if scope is not None:
        context["scope"] = scope
    kwargs = {}
    if all_scope_steps or scope == "batch":
        kwargs["step"] = runner.batch_step
    if all_scope_steps or scope == "epoch" or scope == "loader":
        kwargs["epoch"] = runner.epoch_step
    return context, kwargs


def _build_params_dict(
    dictionary: Dict[str, Any],
    prefix: Dict[str, Any],
    exclude: List[str],
):
    for name, value in dictionary.items():
        if name in exclude:
            continue

        if isinstance(value, dict):
            if name not in prefix:
                prefix[name] = {}
            _build_params_dict(value, prefix[name], exclude)
        else:
            prefix[name] = value


@CustomObject.alias("aim.distribution", exist_ok=True)
class Distribution(CustomObject):
    """Distribution object used to store distribution objects in Aim repository.

    Args:
        hist (:obj:): Array-like object representing bin frequency counts.
            Must be specified alongside `bin_edges`. `data` must not be specified.
        bin_edges (:obj:): Array-like object representing bin edges.
            Must be specified alongside `hist`. `data` must not be specified.
            Max 512 bins allowed.
    """

    AIM_NAME = "aim.distribution"

    def __init__(self, hist, bin_edges):
        super().__init__()
        hist = np.asanyarray(hist)
        bin_edges = np.asanyarray(bin_edges)
        self._from_np_histogram(hist, bin_edges)

    @classmethod
    def from_histogram(cls, hist, bin_edges):
        """Create Distribution object from histogram.

        Args:
            hist (:obj:): Array-like object representing bin frequency counts.
                Must be specified alongside `bin_edges`. `data` must not be specified.
            bin_edges (:obj:): Array-like object representing bin edges.
                Must be specified alongside `hist`. `data` must not be specified.
                Max 512 bins allowed.
        """
        return cls(hist, bin_edges)

    @classmethod
    def from_samples(cls, samples, bin_count=64):
        """Create Distribution object from data samples.

        Args:
            samples (:obj:): Array-like object of data sampled from a distribution.
            bin_count (:obj:`int`, optional): Optional distribution bin count for
                binning `samples`. 64 by default, max 512.
        """
        hist, bin_edges = np.histogram(samples, bins=bin_count)
        return cls(hist, bin_edges)

    def _from_np_histogram(self, hist, bin_edges):
        bin_count = len(bin_edges) - 1
        if 1 > bin_count > 512:
            raise ValueError("Supported range for `bin_count` is [1, 512].")

        self.storage["data"] = BLOB(data=hist.tobytes())
        self.storage["dtype"] = str(hist.dtype)
        self.storage["bin_count"] = bin_count
        self.storage["range"] = [bin_edges[0].item(), bin_edges[-1].item()]


aim.Distribution = Distribution
