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

import os
from datetime import datetime, timezone
from typing import Any

from omegaconf import DictConfig

from compressai_trainer.registry import GIT_PACKAGES
from compressai_trainer.utils import git, system


def get_env(conf: DictConfig) -> dict[str, Any]:
    return {
        "aim": conf.env.aim,
        "git": {
            package.__name__: _get_git_repo_info(
                package.__path__[0],
                conf.env.git.get(package.__name__, {}).get("main_branch", "HEAD"),
            )
            for package in GIT_PACKAGES
        },
        "slurm": {
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "system": {
            "hostname": system.hostname(),
            "username": system.username(),
            "utc_start_time": _utc_timestamp(),
        },
    }


def _get_git_repo_info(root: str, main_branch: str) -> dict[str, str]:
    return {
        "version": git.commit_version(root=root),
        "main_version": git.common_ancestor_commit_version(rev2=main_branch, root=root),
        "branch": git.branch_name(root=root),
        "main_branch": main_branch,
    }


def _utc_timestamp() -> int:
    """Returns milliseconds since UNIX epoch."""
    now = datetime.now(timezone.utc)
    ms = int(now.timestamp() * 1000)
    return ms
