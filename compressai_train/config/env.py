from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import compressai
from aim.sdk.utils import generate_run_hash
from omegaconf import DictConfig

import compressai_train
from compressai_train.utils import git, system


def get_env(conf: DictConfig) -> dict[str, Any]:
    compressai_path = next(iter(compressai.__path__))
    compressai_train_path = next(iter(compressai_train.__path__))

    return {
        "aim": {
            "repo": conf.env.aim.repo,
            "run_hash": generate_run_hash(),
        },
        "git": {
            "compressai": _get_git_repo_info(compressai_path),
            "compressai_train": _get_git_repo_info(compressai_train_path),
        },
        "slurm": {
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
        },
        "system": {
            "hostname": system.hostname(),
            "username": system.username(),
            "utc_start_time": _utc_timestamp(),
        },
    }


def _get_git_repo_info(root: str) -> dict[str, str]:
    return {
        "hash": git.commit_hash(root=root)[:7],
        "main_hash": git.common_ancestor_hash(root=root)[:7],
        "branch": git.branch_name(root=root),
    }


def _utc_timestamp() -> int:
    """Returns milliseconds since UNIX epoch."""
    now = datetime.now(timezone.utc)
    ms = int(now.timestamp() * 1000)
    return ms
