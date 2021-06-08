"""Modified from https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/model_profiling.py"""
import functools
import numpy as np
import time
import torch
import torch.nn as nn
from models.modules import inception_modules as incmod
from models.modules.sync_batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum([np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input, num_forwards=10):
    if num_forwards <= 0:
        return 0.0
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, num_forwards, verbose, logger=None):
    def add_sub(m, sub_op):
        if hasattr(sub_op, 'n_macs'):
            m.n_macs += sub_op.n_macs
            m.n_params += sub_op.n_params
            m.n_seconds += sub_op.n_seconds
        elif isinstance(sub_op, nn.Sequential):
            sub_op.n_macs = 0
            sub_op.n_params = 0
            sub_op.n_seconds = 0
            for op_ in sub_op:
                add_sub(sub_op, op_)
                add_sub(m, op_)

    _run_forward = functools.partial(run_forward, num_forwards=num_forwards)
    ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    self._profiling_input_size = ins
    self._profiling_output_size = outs
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] * self.kernel_size[0] *
                       self.kernel_size[1] * outs[2] * outs[3] //
                       self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] * self.kernel_size[0] *
                       self.kernel_size[1] * outs[2] * outs[3] //
                       self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.BatchNorm2d):
        if not self.track_running_stats:
            self.n_macs = outs[1] * outs[2] * outs[3] * outs[0]
            self.n_params = get_params(self)
            self.n_seconds = _run_forward(self, input)
        else:
            self.n_macs = 0
            self.n_params = 0
            self.n_seconds = 0
        self.name = self.__repr__()
    elif isinstance(self, SynchronizedBatchNorm2d):
        if not self.track_running_stats:
            self.n_macs = outs[1] * outs[2] * outs[3] * outs[0]
            self.n_params = get_params(self)
            self.n_seconds = _run_forward(self, input)
        else:
            self.n_macs = 0
            self.n_params = 0
            self.n_seconds = 0
        self.name = self.__repr__()
    elif isinstance(self, nn.InstanceNorm2d):
        if not self.track_running_stats:
            self.n_macs = outs[1] * outs[2] * outs[3] * outs[0]
            self.n_params = get_params(self)
            self.n_seconds = _run_forward(self, input)
        else:
            self.n_macs = 0
            self.n_params = 0
            self.n_seconds = 0
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        if getattr(self, 'zero_macs', False):
            self.n_macs = 0
            self.n_params = 0
            self.n_seconds = 0
        else:
            self.n_macs = outs[1] * outs[2] * outs[
                3] * self.kernel_size * self.kernel_size * outs[0]
            self.n_params = 0
            self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        if getattr(self, 'zero_macs', False):
            self.n_macs = 0
            self.n_params = 0
            self.n_seconds = 0
        else:
            self.n_macs = outs[1] * outs[2] * outs[3] * self.kernel_size[
                0] * self.kernel_size[1] * outs[0]
            self.n_params = 0
            self.n_seconds = _run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, incmod.InvertedResidualChannels):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for res_op in self.res_ops:
            add_sub(self, res_op)
        for dw_op in self.dw_ops:
            add_sub(self, dw_op)
        add_sub(self, self.pw_bn)
        self.name = self.__repr__()
    elif isinstance(self, incmod.SPADEInvertedResidualChannels):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        for res_op in self.res_ops:
            add_sub(self, res_op)
        for dw_op in self.dw_ops:
            add_sub(self, dw_op)
        if self.shortcut is not None:
            add_sub(self, self.shortcut)
        add_sub(self, self.spade)
        self.name = self.__repr__()
    elif isinstance(self, incmod.ConvSyncBNReLU):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.conv)
        add_sub(self, self.norm)
        self.name = self.__repr__()
    elif isinstance(self, incmod.Conv):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.conv)
        self.name = self.__repr__()
    elif isinstance(self, incmod.InceptionSPADE):
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        add_sub(self, self.param_free_norm)
        for res_op in self.res_ops:
            add_sub(self, res_op)
        for dw_op in self.dw_ops:
            add_sub(self, dw_op)
        self.name = self.__repr__()
    else:
        # NOTE: This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d,
            SynchronizedBatchNorm2d,
            nn.InstanceNorm2d,
            nn.Dropout2d,
            nn.Dropout,
            nn.ReLU6,
            nn.ReLU,
            nn.Tanh,
            nn.MaxPool2d,
            nn.ConstantPad2d,
            nn.ReflectionPad2d,
            nn.ReplicationPad2d,
            nn.modules.padding.ZeroPad2d,
            nn.modules.activation.Sigmoid,
        ]
        if (not getattr(self, 'ignore_model_profiling', False)
                and self.n_macs == 0 and t not in ignore_zeros_t):
            if logger is not None:
                logger.print_info(
                    'WARNING: leaf module {} has zero n_macs.\n'.format(
                        type(self)))
            else:
                print('WARNING: leaf module {} has zero n_macs.'.format(
                    type(self)))
        return
    if verbose:
        if logger is not None:
            logger.print_info(
                self.name.ljust(name_space, ' ') +
                '{:,}'.format(self.n_params).rjust(params_space, ' ') +
                '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
                '{:,}'.format(self.n_seconds).rjust(seconds_space, ' ') + '\n')
        else:
            print(
                self.name.ljust(name_space, ' ') +
                '{:,}'.format(self.n_params).rjust(params_space, ' ') +
                '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
                '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, num_forwards, verbose, logger=None):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(lambda m, input, output: module_profiling(
            m, input, output, num_forwards, verbose=verbose, logger=logger)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model,
                    height,
                    width,
                    batch=1,
                    channel=3,
                    use_cuda=True,
                    num_forwards=10,
                    verbose=True,
                    logger=None):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.apply(lambda m: setattr(m, 'profiling', True))
    model.eval()
    data = torch.rand(batch, channel, height, width)
    origin_device = next(model.parameters()).device
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(
        m, num_forwards, verbose=verbose, logger=logger))
    if verbose:
        if logger is not None:
            logger.print_info('Item'.ljust(name_space, ' ') +
                              'params'.rjust(macs_space, ' ') +
                              'macs'.rjust(macs_space, ' ') +
                              'nanosecs'.rjust(seconds_space, ' ') + '\n')
            logger.print_info(''.center(
                name_space + params_space + macs_space + seconds_space, '-') +
                              '\n')
        else:
            print('Item'.ljust(name_space, ' ') +
                  'params'.rjust(macs_space, ' ') +
                  'macs'.rjust(macs_space, ' ') +
                  'nanosecs'.rjust(seconds_space, ' '))
            print(''.center(
                name_space + params_space + macs_space + seconds_space, '-'))
    with torch.no_grad():
        model(data)
    if verbose:
        if logger is not None:
            logger.print_info(''.center(
                name_space + params_space + macs_space + seconds_space, '-') +
                              '\n')
            logger.print_info(
                'Total'.ljust(name_space, ' ') +
                '{:,}'.format(model.n_params).rjust(params_space, ' ') +
                '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
                '{:,}'.format(model.n_seconds).rjust(seconds_space, ' ') +
                '\n')
        else:
            print(''.center(
                name_space + params_space + macs_space + seconds_space, '-'))
            print('Total'.ljust(name_space, ' ') +
                  '{:,}'.format(model.n_params).rjust(params_space, ' ') +
                  '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
                  '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    model = model.to(origin_device)
    model.apply(lambda m: setattr(m, 'profiling', False))
    return model.n_macs, model.n_params
