"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import copy
import importlib
import logging
import math
import os

import torch
import torch.distributed as dist


def unwrap_model(model_wrapper):
    """Remove model's wrapper."""
    if hasattr(model_wrapper, 'module'):
        model = model_wrapper.module
    else:
        model = model_wrapper
    return model


def add_prefix(name, prefix=None, split='.'):
    """Add prefix to name if given."""
    if prefix is not None:
        return '{}{}{}'.format(prefix, split, name)
    else:
        return name


def get_device(x):
    """Find device given tensor or module.

    NOTE: assume all model parameters reside on the same devices.
    """
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, torch.nn.Module):
        return next(x.parameters()).device
    else:
        raise RuntimeError('{} do not have `device`'.format(type(x)))
