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

from __future__ import annotations

from catalyst.core.logger import ILogger

IMAGE_LOGGERS = ["aim"]


class DistributionSuperlogger:
    loggers: dict[str, ILogger]

    def log_distribution(self, *args, **kwargs) -> None:
        """Logs distribution to available loggers."""
        for logger in self.loggers.values():
            if not hasattr(logger, "log_distribution"):
                continue
            logger.log_distribution(*args, **kwargs, runner=self)  # type: ignore


class FigureSuperlogger:
    loggers: dict[str, ILogger]

    def log_figure(self, *args, **kwargs) -> None:
        """Logs figure to available loggers."""
        for logger in self.loggers.values():
            if not hasattr(logger, "log_figure"):
                continue
            logger.log_figure(*args, **kwargs, runner=self)  # type: ignore


class ImageSuperlogger:
    loggers: dict[str, ILogger]

    def __init__(self, enabled_image_loggers: list[str] = IMAGE_LOGGERS):
        self._enabled_image_loggers = enabled_image_loggers

    def log_image(self, *args, **kwargs) -> None:
        """Logs image to available loggers."""
        for name, logger in self.loggers.items():
            if name not in self._enabled_image_loggers:
                continue
            if not hasattr(logger, "log_image"):
                continue
            if name != "aim":
                valid_keys = ["tag", "image", "runner", "scope"]
                kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            logger.log_image(*args, **kwargs, runner=self)  # type: ignore


class AllSuperlogger(
    DistributionSuperlogger,
    FigureSuperlogger,
    ImageSuperlogger,
):
    def __init__(self, enabled_image_loggers: list[str] = IMAGE_LOGGERS):
        DistributionSuperlogger.__init__(self)
        FigureSuperlogger.__init__(self)
        ImageSuperlogger.__init__(self, enabled_image_loggers)
