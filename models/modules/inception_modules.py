import re
import collections
import functools
from torch import nn
import torch.nn.functional as F

from common import add_prefix

from models.modules.sync_batchnorm import SynchronizedBatchNorm2d


def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
        'nn.LeakyReLU': functools.partial(nn.LeakyReLU, inplace=True),
    }[name]
    return active_fn


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 use_bias=True,
                 norm_layer=nn.InstanceNorm2d,
                 norm_kwargs=None,
                 active_fn=None):
        if norm_kwargs is None:
            norm_kwargs = {}
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      0,
                      groups=groups,
                      bias=use_bias), norm_layer(out_planes, **norm_kwargs),
            active_fn())


class InvertedResidualChannels(nn.Module):
    """MobiletNetV2 building block."""
    def __init__(self,
                 inp,
                 res_channels,
                 dw_channels,
                 channels_reduction_factor,
                 res_kernel_sizes,
                 dw_kernel_sizes,
                 padding_type='reflect',
                 use_bias=True,
                 norm_layer=nn.InstanceNorm2d,
                 norm_kwargs=None,
                 dropout_rate=0.0,
                 active_fn=None):
        super(InvertedResidualChannels, self).__init__()
        if type(res_kernel_sizes) == int:
            res_kernel_sizes = [res_kernel_sizes]
        if res_channels is not None:
            assert type(res_channels) == int or len(res_channels) == len(
                res_kernel_sizes)
        if type(dw_kernel_sizes) == int:
            dw_kernel_sizes = [dw_kernel_sizes]
        if dw_channels is not None:
            assert type(dw_channels) == int or len(dw_channels) == len(
                dw_kernel_sizes)

        self.input_dim = inp
        if res_channels is None:
            self.res_channels = [
                inp // channels_reduction_factor for _ in res_kernel_sizes
            ]
        elif type(res_channels) == int:
            self.res_channels = [
                res_channels // channels_reduction_factor
                for _ in res_kernel_sizes
            ]
        else:
            assert len(res_channels) == len(res_kernel_sizes)
            self.res_channels = [
                c // channels_reduction_factor for c in res_channels
            ]
        if dw_channels is None:
            self.dw_channels = [
                inp // channels_reduction_factor for _ in dw_kernel_sizes
            ]
        elif type(dw_channels) == int:
            self.dw_channels = [
                dw_channels // channels_reduction_factor
                for _ in dw_kernel_sizes
            ]
        else:
            assert len(dw_channels) == len(dw_kernel_sizes)
            self.dw_channels = [
                c // channels_reduction_factor for c in dw_channels
            ]
        self.res_kernel_sizes = res_kernel_sizes
        self.dw_kernel_sizes = dw_kernel_sizes
        self.padding_type = padding_type
        self.use_bias = use_bias
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.dropout_rate = dropout_rate
        self.active_fn = active_fn

        if self.padding_type == 'reflect':
            self.pad = nn.ReflectionPad2d
        elif self.padding_type == 'replicate':
            self.pad = nn.ReplicationPad2d
        elif self.padding_type == 'zero':
            self.pad = functools.partial(nn.ConstantPad2d, value=0.0)
        else:
            raise NotImplementedError('padding [%s] is not implemented' %
                                      padding_type)

        self.res_ops, self.dw_ops, self.pw_bn = self._build()

    def _build(self):
        _norm_kwargs = self.norm_kwargs \
            if self.norm_kwargs is not None else {}

        res_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.res_channels,
                                      self.res_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(self.pad((k - 1) // 2))
            layers.append(
                ConvBNReLU(self.input_dim,
                           midp,
                           kernel_size=k,
                           use_bias=self.use_bias,
                           norm_layer=self.norm_layer,
                           norm_kwargs=_norm_kwargs,
                           active_fn=self.active_fn))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(self.pad((k - 1) // 2))
            layers.append(
                nn.Conv2d(midp, self.input_dim, k, 1, 0, bias=self.use_bias))
            res_ops.append(nn.Sequential(*layers))

        dw_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.dw_channels, self.dw_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(
                ConvBNReLU(self.input_dim,
                           midp,
                           kernel_size=1,
                           use_bias=self.use_bias,
                           norm_layer=self.norm_layer,
                           norm_kwargs=_norm_kwargs,
                           active_fn=self.active_fn))
            layers.extend([
                self.pad((k - 1) // 2),
                ConvBNReLU(midp,
                           midp,
                           kernel_size=k,
                           groups=midp,
                           use_bias=self.use_bias,
                           norm_layer=self.norm_layer,
                           norm_kwargs=_norm_kwargs,
                           active_fn=self.active_fn),
                nn.Dropout(self.dropout_rate),
                nn.Conv2d(midp, self.input_dim, 1, 1, 0, bias=self.use_bias),
            ])
            dw_ops.append(nn.Sequential(*layers))
        pw_bn = self.norm_layer(self.input_dim, **_norm_kwargs)

        return res_ops, dw_ops, pw_bn

    def get_first_res_bn(self):
        """Get `[module]` list of res BN after the first convolution."""
        return list(self.get_named_first_res_bn().values())

    def get_first_dw_bn(self):
        """Get `[module]` list of dw BN after the first convolution."""
        return list(self.get_named_first_dw_bn().values())

    def get_first_bn(self):
        """Get `[module]` list of BN after the first convolution."""
        return self.get_first_res_bn() + self.get_first_dw_bn()

    def get_named_first_res_bn(self, prefix=None):
        """Get `{name: module}` pairs of res BN after the first convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.res_ops):
            assert isinstance(op[1], ConvBNReLU)
            norm_layer_ = op[1][1]
            if type(self.norm_layer) == functools.partial:
                assert isinstance(norm_layer_, self.norm_layer.func)
            else:
                assert isinstance(norm_layer_, self.norm_layer)
            name = f'res_ops.{i}.1.1'
            name = add_prefix(name, prefix)
            res[name] = norm_layer_
        return res

    def get_named_first_dw_bn(self, prefix=None):
        """Get `{name: module}` pairs of dw BN after the first convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.dw_ops):
            assert isinstance(op[0], ConvBNReLU)
            norm_layer_ = op[0][1]
            if type(self.norm_layer) == functools.partial:
                assert isinstance(norm_layer_, self.norm_layer.func)
            else:
                assert isinstance(norm_layer_, self.norm_layer)
            name = f'dw_ops.{i}.0.1'
            name = add_prefix(name, prefix)
            res[name] = norm_layer_
        return res

    def get_named_first_bn(self, prefix=None):
        """Get `{name: module}` pairs of dw BN after the first convolution."""
        return collections.OrderedDict(
            list(self.get_named_first_res_bn().items()) +
            list(self.get_named_first_dw_bn().items()))

    def forward(self, x):
        if len(self.res_ops) == 0 and len(self.dw_ops) == 0:
            return x
        tmp = sum([op(x) for op in self.res_ops]) + sum(
            [op(x) for op in self.dw_ops])
        tmp = self.pw_bn(tmp)
        return x + tmp

    def __repr__(self):
        return (
            '{}({}, {}, res_channels={}, dw_channels={}, res_kernel_sizes={}, dw_kernel_sizes={})'
        ).format(self._get_name(), self.input_dim, self.input_dim,
                 self.res_channels, self.dw_channels, self.res_kernel_sizes,
                 self.dw_kernel_sizes)


def output_network(model):
    """Output network kwargs in `searched_network` style."""
    model_kwargs = {}
    blocks = list(model.get_named_block_list().values())

    res = []
    for block in blocks:
        res.append([
            block.input_dim, block.res_channels, block.dw_channels,
            block.res_kernel_sizes, block.dw_kernel_sizes,
            getattr(block, 'stride', 1)
        ])
    model_kwargs['inverted_residual_setting'] = res
    return model_kwargs


def _get_named_block_list(m, spade=False, num_upsampling_layers=None):
    """Get `{name: module}` dictionary for inverted residual blocks."""
    if spade:
        features_blocks = [('head_0', m.head_0)]
        features_blocks += [(f'G_middle_{i}', getattr(m, f'G_middle_{i}'))
                            for i in range(2)]
        features_blocks += [(f'up_{i}', getattr(m, f'up_{i}'))
                            for i in range(4)]
        if num_upsampling_layers == 'most':
            features_blocks += [('up_4', m.up_4)]
        return collections.OrderedDict([('{}'.format(name), block)
                                        for name, block in features_blocks])
    else:
        features_blocks = list(m.features.named_children())
        return collections.OrderedDict([('features.{}'.format(name), block)
                                        for name, block in features_blocks])


class ConvSyncBNReLU(nn.Module):
    """Convolution-SyncBatchNormalization-ActivateFn."""
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 use_bias=True,
                 norm_layer=None,
                 active_fn=None,
                 spectral_norm=False,
                 spade=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size,
                              stride, (kernel_size - 1) // 2,
                              groups=groups,
                              bias=use_bias)
        if spade:
            self.norm = norm_layer(norm_nc=out_planes)
        else:
            self.norm = norm_layer(out_planes)
        self.active = active_fn()
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x, seg=None):
        if seg is not None:
            return self.active(self.norm(self.conv(x), seg))
        else:
            return self.active(self.norm(self.conv(x)))

    def remove_spectral_norm(self):
        self.conv = nn.utils.remove_spectral_norm(self.conv)


class Conv(nn.Module):
    """Convolution."""
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 use_bias=True,
                 spectral_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size,
                              stride, (kernel_size - 1) // 2,
                              groups=groups,
                              bias=use_bias)
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    def remove_spectral_norm(self):
        self.conv = nn.utils.remove_spectral_norm(self.conv)


class SPADEInvertedResidualChannels(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        res_kernel_sizes = opt.kernel_sizes
        dw_kernel_sizes = opt.kernel_sizes
        res_channels = opt.channels
        dw_channels = opt.channels
        channels_reduction_factor = opt.channels_reduction_factor
        active_fn = get_active_fn(opt.active_fn)
        if type(res_kernel_sizes) == int:
            res_kernel_sizes = [res_kernel_sizes]
        if res_channels is not None:
            assert type(res_channels) == int or len(res_channels) == len(
                res_kernel_sizes)
        if type(dw_kernel_sizes) == int:
            dw_kernel_sizes = [dw_kernel_sizes]
        if dw_channels is not None:
            assert type(dw_channels) == int or len(dw_channels) == len(
                dw_kernel_sizes)

        self.input_dim = fin
        self.output_dim = fout
        if res_channels is None:
            self.res_channels = [
                fmiddle // channels_reduction_factor for _ in res_kernel_sizes
            ]
        elif type(res_channels) == int:
            self.res_channels = [
                res_channels // channels_reduction_factor
                for _ in res_kernel_sizes
            ]
        else:
            assert len(res_channels) == len(res_kernel_sizes)
            self.res_channels = [
                c // channels_reduction_factor for c in res_channels
            ]
        if dw_channels is None:
            self.dw_channels = [
                fmiddle // channels_reduction_factor for _ in dw_kernel_sizes
            ]
        elif type(dw_channels) == int:
            self.dw_channels = [
                dw_channels // channels_reduction_factor
                for _ in dw_kernel_sizes
            ]
        else:
            assert len(dw_channels) == len(dw_kernel_sizes)
            self.dw_channels = [
                c // channels_reduction_factor for c in dw_channels
            ]
        self.res_kernel_sizes = res_kernel_sizes
        self.dw_kernel_sizes = dw_kernel_sizes
        self.active_fn = active_fn
        self.active = active_fn()

        self.spectral_norm = 'spectral' in opt.norm_G

        spade_config_str = opt.norm_G.replace('spectral', '')
        if spade_config_str.startswith('spade'):
            parsed = re.search(r'spade(\D+)(\d)x\d', spade_config_str)
            param_free_norm_type = str(parsed.group(1))
            spade_kernel_size = int(parsed.group(2))
            self.spade_norm = InceptionSPADE
        else:
            raise NotImplementedError
        if param_free_norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d
        elif param_free_norm_type == 'syncbatch':
            self.norm_layer = SynchronizedBatchNorm2d
        elif param_free_norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(
                f'{param_free_norm_type} is not a recognized param-free norm type in SPADE'
            )
        self.semantic_nc = opt.semantic_nc

        self.res_ops, self.dw_ops, self.shortcut, self.spade = self._build()

    def _build(self, build_only=False):
        res_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.res_channels,
                                      self.res_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(
                ConvSyncBNReLU(self.input_dim,
                               midp,
                               kernel_size=k,
                               norm_layer=functools.partial(self.norm_layer,
                                                            affine=True),
                               active_fn=self.active_fn,
                               spectral_norm=self.spectral_norm,
                               spade=False))
            layers.append(
                Conv(midp,
                     self.output_dim,
                     kernel_size=k,
                     spectral_norm=self.spectral_norm))
            res_ops.append(nn.Sequential(*layers))

        dw_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.dw_channels, self.dw_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(
                ConvSyncBNReLU(self.input_dim,
                               midp,
                               kernel_size=1,
                               norm_layer=functools.partial(self.norm_layer,
                                                            affine=True),
                               active_fn=self.active_fn,
                               spectral_norm=self.spectral_norm,
                               spade=False))
            layers.append(
                ConvSyncBNReLU(midp,
                               midp,
                               kernel_size=k,
                               groups=midp,
                               norm_layer=functools.partial(self.norm_layer,
                                                            affine=False),
                               active_fn=self.active_fn,
                               spectral_norm=self.spectral_norm,
                               spade=False))
            layers.append(
                Conv(midp,
                     self.output_dim,
                     kernel_size=1,
                     spectral_norm=self.spectral_norm))
            dw_ops.append(nn.Sequential(*layers))

        if self.learned_shortcut:
            shortcut = nn.Sequential(
                self.norm_layer(self.input_dim, affine=True),
                Conv(self.input_dim,
                     self.output_dim,
                     kernel_size=1,
                     use_bias=False,
                     spectral_norm=self.spectral_norm))
        else:
            shortcut = None

        if build_only:
            self.spade.param_free_norm, self.spade.res_ops, self.spade.dw_ops = self.spade._build(
            )
            spade = self.spade
        else:
            spade = self.spade_norm(norm=self.norm_layer,
                                    norm_nc=self.input_dim,
                                    label_nc=self.semantic_nc,
                                    opt=self.opt)

        return res_ops, dw_ops, shortcut, spade

    def get_first_res_bn(self):
        """Get `[module]` list of res BN after the first convolution."""
        return list(self.get_named_first_res_bn().values())

    def get_first_dw_bn(self):
        """Get `[module]` list of dw BN after the first convolution."""
        return list(self.get_named_first_dw_bn().values())

    def get_first_bn(self):
        """Get `[module]` list of BN after the first convolution."""
        return self.get_first_res_bn() + self.get_first_dw_bn()

    def get_named_first_res_bn(self, prefix=None):
        """Get `{name: module}` pairs of res BN after the first convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.res_ops):
            assert isinstance(op[0], ConvSyncBNReLU)
            norm_layer_ = op[0].norm
            assert isinstance(norm_layer_, self.norm_layer)
            name = f'res_ops.{i}.0.norm'
            name = add_prefix(name, prefix)
            res[name] = norm_layer_
        return res

    def get_named_first_dw_bn(self, prefix=None):
        """Get `{name: module}` pairs of dw BN after the first convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.dw_ops):
            assert isinstance(op[0], ConvSyncBNReLU)
            norm_layer_ = op[0].norm
            assert isinstance(norm_layer_, self.norm_layer)
            name = f'dw_ops.{i}.0.norm'
            name = add_prefix(name, prefix)
            res[name] = norm_layer_
        return res

    def get_named_first_bn(self, prefix=None):
        """Get `{name: module}` pairs of dw BN after the first convolution."""
        return collections.OrderedDict(
            list(self.get_named_first_res_bn().items()) +
            list(self.get_named_first_dw_bn().items()))

    def forward(self, x, seg):
        if len(self.res_ops) == 0 and len(self.dw_ops) == 0:
            if self.shortcut is not None:
                x = self.shortcut(x)
            return x

        tmp = self.spade(x, seg)
        tmp = self.active(tmp)
        tmp = sum([op(tmp) for op in self.res_ops]) + sum(
            [op(tmp) for op in self.dw_ops])
        if self.shortcut is not None:
            return tmp + self.shortcut(x)
        else:
            return tmp + x

    def __repr__(self):
        return (
            '{}({}, {}, res_channels={}, dw_channels={}, res_kernel_sizes={}, dw_kernel_sizes={})\n\tSPADE: {}'
        ).format(self._get_name(), self.input_dim, self.input_dim,
                 self.res_channels, self.dw_channels, self.res_kernel_sizes,
                 self.dw_kernel_sizes, self.spade)

    def remove_spectral_norm(self):
        for op in self.res_ops:
            assert isinstance(op[0], ConvSyncBNReLU)
            op[0].remove_spectral_norm()
            assert isinstance(op[1], Conv)
            op[1].remove_spectral_norm()
        for op in self.dw_ops:
            assert isinstance(op[0], ConvSyncBNReLU)
            op[0].remove_spectral_norm()
            assert isinstance(op[1], ConvSyncBNReLU)
            op[1].remove_spectral_norm()
            assert isinstance(op[2], Conv)
            op[2].remove_spectral_norm()
        if self.shortcut is not None:
            assert isinstance(self.shortcut[1], Conv)
            self.shortcut[1].remove_spectral_norm()


class InceptionSPADE(nn.Module):
    def __init__(self, norm, norm_nc, label_nc, nhidden=128, opt=None):
        super(InceptionSPADE, self).__init__()

        res_kernel_sizes = opt.kernel_sizes
        dw_kernel_sizes = opt.kernel_sizes
        res_channels = opt.channels
        dw_channels = opt.channels
        channels_reduction_factor = opt.channels_reduction_factor
        self.norm_layer = functools.partial(SynchronizedBatchNorm2d,
                                            affine=True)
        self.active_fn = functools.partial(nn.ReLU, inplace=True)
        if type(res_kernel_sizes) == int:
            res_kernel_sizes = [res_kernel_sizes]
        if res_channels is not None:
            assert type(res_channels) == int or len(res_channels) == len(
                res_kernel_sizes)
        if type(dw_kernel_sizes) == int:
            dw_kernel_sizes = [dw_kernel_sizes]
        if dw_channels is not None:
            assert type(dw_channels) == int or len(dw_channels) == len(
                dw_kernel_sizes)

        self.param_free_norm_layer = norm

        self.input_dim = label_nc
        self.output_dim = norm_nc
        if res_channels is None:
            self.res_channels = [
                nhidden // channels_reduction_factor for _ in res_kernel_sizes
            ]
        elif type(res_channels) == int:
            self.res_channels = [
                res_channels // channels_reduction_factor
                for _ in res_kernel_sizes
            ]
        else:
            assert len(res_channels) == len(res_kernel_sizes)
            self.res_channels = [
                c // channels_reduction_factor for c in res_channels
            ]
        if dw_channels is None:
            self.dw_channels = [
                nhidden // channels_reduction_factor for _ in dw_kernel_sizes
            ]
        elif type(dw_channels) == int:
            self.dw_channels = [
                dw_channels // channels_reduction_factor
                for _ in dw_kernel_sizes
            ]
        else:
            assert len(dw_channels) == len(dw_kernel_sizes)
            self.dw_channels = [
                c // channels_reduction_factor for c in dw_channels
            ]
        self.res_kernel_sizes = res_kernel_sizes
        self.dw_kernel_sizes = dw_kernel_sizes
        self.param_free_norm, self.res_ops, self.dw_ops = self._build()

    def _build(self):
        param_free_norm = self.param_free_norm_layer(self.output_dim,
                                                     affine=False)

        res_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.res_channels,
                                      self.res_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(
                ConvSyncBNReLU(self.input_dim,
                               midp,
                               kernel_size=k,
                               norm_layer=self.norm_layer,
                               active_fn=self.active_fn,
                               spectral_norm=False,
                               spade=False))
            layers.append(
                nn.Conv2d(midp,
                          2 * self.output_dim,
                          kernel_size=k,
                          padding=(k - 1) // 2))
            res_ops.append(nn.Sequential(*layers))

        dw_ops = nn.ModuleList()
        for idx, (midp,
                  k) in enumerate(zip(self.dw_channels, self.dw_kernel_sizes)):
            if midp == 0:
                continue
            layers = []
            layers.append(
                ConvSyncBNReLU(self.input_dim,
                               midp,
                               kernel_size=1,
                               norm_layer=self.norm_layer,
                               active_fn=self.active_fn,
                               spectral_norm=False,
                               spade=False))
            layers.append(
                ConvSyncBNReLU(midp,
                               midp,
                               kernel_size=k,
                               groups=midp,
                               norm_layer=self.norm_layer,
                               active_fn=self.active_fn,
                               spectral_norm=False,
                               spade=False))
            layers.append(nn.Conv2d(midp, 2 * self.output_dim, kernel_size=1))
            dw_ops.append(nn.Sequential(*layers))

        return param_free_norm, res_ops, dw_ops

    def get_first_res_bn(self):
        """Get `[module]` list of res BN after the first convolution."""
        return list(self.get_named_first_res_bn().values())

    def get_first_dw_bn(self):
        """Get `[module]` list of dw BN after the first convolution."""
        return list(self.get_named_first_dw_bn().values())

    def get_first_bn(self):
        """Get `[module]` list of BN after the first convolution."""
        return self.get_first_res_bn() + self.get_first_dw_bn()

    def get_named_first_res_bn(self, prefix=None):
        """Get `{name: module}` pairs of res BN after the first convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.res_ops):
            assert isinstance(op[0], ConvSyncBNReLU)
            norm_layer_ = op[0].norm
            assert isinstance(
                norm_layer_, getattr(self.norm_layer, 'func', self.norm_layer))
            name = f'res_ops.{i}.0.norm'
            name = add_prefix(name, prefix)
            res[name] = norm_layer_
        return res

    def get_named_first_dw_bn(self, prefix=None):
        """Get `{name: module}` pairs of dw BN after the first convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.dw_ops):
            assert isinstance(op[0], ConvSyncBNReLU)
            norm_layer_ = op[0].norm
            assert isinstance(
                norm_layer_, getattr(self.norm_layer, 'func', self.norm_layer))
            name = f'dw_ops.{i}.0.norm'
            name = add_prefix(name, prefix)
            res[name] = norm_layer_
        return res

    def get_named_first_bn(self, prefix=None):
        """Get `{name: module}` pairs of dw BN after the first convolution."""
        return collections.OrderedDict(
            list(self.get_named_first_res_bn().items()) +
            list(self.get_named_first_dw_bn().items()))

    def forward(self, x, segmap):

        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        if len(self.res_ops) == 0 and len(self.dw_ops) == 0:
            gamma = 0
            beta = 0
        else:
            tmp = sum([op(segmap) for op in self.res_ops]) + sum(
                [op(segmap) for op in self.dw_ops])
            gamma = tmp[:, :self.output_dim]
            beta = tmp[:, self.output_dim:]

        out = normalized * (1 + gamma) + beta

        return out

    def __repr__(self):
        return (
            '{}({}, {}, res_channels={}, dw_channels={}, res_kernel_sizes={}, dw_kernel_sizes={})'
        ).format(self._get_name(), self.input_dim, self.input_dim,
                 self.res_channels, self.dw_channels, self.res_kernel_sizes,
                 self.dw_kernel_sizes)
