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

import argparse
import os
import sys

from hydra import compose, initialize

DEFAULT_CONFIG_PATH = "conf"
DEFAULT_CONFIG_NAME = "config"
SPECIAL_OPTIONS = ["--config-path", "--config-name"]

__all__ = [
    "iter_configs",
]


def split_argv_groups(argv, special_options):
    argv_groups = [[]]
    visited_options = set()
    for arg in argv:
        for option in special_options:
            if arg.startswith(option):
                if option in visited_options:
                    visited_options = set()
                    argv_groups.append([])
                visited_options.add(option)
        argv_groups[-1].append(arg)
    return argv_groups


def parse_argv_group(argv):
    args = argparse.Namespace()
    args.config_path = DEFAULT_CONFIG_PATH
    args.config_name = DEFAULT_CONFIG_NAME
    args.overrides = []

    option_name = None

    for arg in argv:
        # Parse the value portion of --option value.
        if option_name is not None:
            option_value = arg
            setattr(args, option_name, option_value)
            option_name = None
            continue

        option_detected = False
        for option in SPECIAL_OPTIONS:
            arg_option, *arg_value = arg.split("=", 1)
            if arg_option != option:
                continue
            option_detected = True
            # Parse --option=value or --option value.
            option_name = arg_option[len("--") :].replace("-", "_")
            if len(arg_value) != 0:
                [option_value] = arg_value
                setattr(args, option_name, option_value)
                option_name = None
            break
        if option_detected:
            continue

        # All other arguments are overrides.
        args.overrides.append(arg)

    return args


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    argv_groups = split_argv_groups(argv, SPECIAL_OPTIONS)
    args_groups = [parse_argv_group(argv_group) for argv_group in argv_groups]

    # Merge all args into a single namespace containing lists.
    args = argparse.Namespace()
    for args_group in args_groups:
        for key, value in vars(args_group).items():
            if not hasattr(args, key):
                setattr(args, key, [])
            getattr(args, key).append(value)

    return args


def relpath(path, start=None):
    return str(os.path.relpath(path, start))


def iter_configs(argv=None, start=None):
    args = parse_args(argv)
    for config_path, config_name, overrides in zip(
        args.config_path, args.config_name, args.overrides
    ):
        with initialize(version_base=None, config_path=relpath(config_path, start)):
            conf = compose(config_name=config_name, overrides=overrides)
        yield conf
