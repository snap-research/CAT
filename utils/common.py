"""Common utilities."""
import re
import functools
import copy
import numbers
import itertools
import os
import sys
import shutil
import time
import logging
import yaml
import torch
from torch.optim.adam import Adam
import torch.nn as nn
from packaging import version

import common as mc
from utils import prune
from utils.model_profiling import model_profiling
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d
from models import networks


def get_params_by_name(model, names):
    """Get params/buffers by name."""
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    named_vars = {**named_parameters, **named_buffers}
    res = [named_vars[name].abs() for name in names]
    return res


def get_prune_weights(model, names):
    return get_params_by_name(mc.unwrap_model(model), names)


def KA(X, Y):
    X_ = X.view(X.size(0), -1)
    Y_ = Y.view(Y.size(0), -1)
    assert X_.shape[0] == Y_.shape[
        0], f'X_ and Y_ must have the same shape on dim 0, but got {X_.shape[0]} for X_ and {Y_.shape[0]} for Y_.'
    X_vec = X_ @ X_.T
    Y_vec = Y_ @ Y_.T
    ret = (X_vec * Y_vec).sum() / ((X_vec**2).sum() * (Y_vec**2).sum())**0.5
    return ret


def load_pretrained_student(model, opt):
    pretrained_studentG_state = torch.load(opt.pretrained_student_G_path)
    model.remove_mapping_hook()
    norm_layer = {
        'instance': nn.InstanceNorm2d,
        'batch': nn.BatchNorm2d,
        'syncbatch': SynchronizedBatchNorm2d
    }[opt.norm]
    netG_tmp = copy.deepcopy(mc.unwrap_model(model.netG_teacher))
    ds_idx_list = []
    us_idx_list = []
    for idx, layer in enumerate(netG_tmp.down_sampling):
        if isinstance(layer, norm_layer):
            ds_idx_list.append(idx)
    for idx, layer in enumerate(netG_tmp.up_sampling):
        if isinstance(layer, norm_layer):
            us_idx_list.append(idx)
    in_channels = None
    for idx in ds_idx_list:
        out_channels = pretrained_studentG_state[
            f'down_sampling.{idx}.weight'].shape[0]
        netG_tmp.down_sampling[idx] = norm_layer(
            out_channels,
            affine=opt.norm_affine,
            track_running_stats=opt.norm_track_running_stats)
        if in_channels is None:
            in_channels = netG_tmp.down_sampling[idx - 1].in_channels
        kernel_size = netG_tmp.down_sampling[idx - 1].kernel_size
        stride = netG_tmp.down_sampling[idx - 1].stride
        padding = netG_tmp.down_sampling[idx - 1].padding
        bias = netG_tmp.down_sampling[idx - 1].bias is not None
        netG_tmp.down_sampling[idx - 1] = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=bias)
        in_channels = out_channels
    ngf_netA = in_channels
    for idx, layer in enumerate(netG_tmp.features):
        layer.input_dim = in_channels
        layer.res_channels, layer.res_kernel_sizes = [], []
        for k, v in pretrained_studentG_state.items():
            if f'features.{idx}.res_ops.' in k and '.1.0.weight' in k:
                ch_, _, k_, _ = v.shape
                layer.res_channels.append(ch_)
                layer.res_kernel_sizes.append(k_)
        layer.dw_channels, layer.dw_kernel_sizes = [], []
        for k, v in pretrained_studentG_state.items():
            if f'features.{idx}.dw_ops.' in k and '.2.0.weight' in k:
                ch_, _, k_, _ = v.shape
                layer.dw_channels.append(ch_)
                layer.dw_kernel_sizes.append(k_)
        print(f'features.{idx}.res_ops', layer.res_channels,
              layer.res_kernel_sizes)
        print(f'features.{idx}.dw_ops', layer.dw_channels,
              layer.dw_kernel_sizes)
        layer.res_ops, layer.dw_ops, layer.pw_bn = layer._build()
    for idx in us_idx_list:
        out_channels = pretrained_studentG_state[
            f'up_sampling.{idx}.weight'].shape[0]
        netG_tmp.up_sampling[idx] = norm_layer(
            out_channels,
            affine=opt.norm_affine,
            track_running_stats=opt.norm_track_running_stats)
        kernel_size = netG_tmp.up_sampling[idx - 1].kernel_size
        stride = netG_tmp.up_sampling[idx - 1].stride
        padding = netG_tmp.up_sampling[idx - 1].padding
        output_padding = netG_tmp.up_sampling[idx - 1].output_padding
        bias = netG_tmp.up_sampling[idx - 1].bias is not None
        netG_tmp.up_sampling[idx - 1] = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias)
        in_channels = out_channels
    out_channels = netG_tmp.up_sampling[-2].out_channels
    kernel_size = netG_tmp.up_sampling[-2].kernel_size
    stride = netG_tmp.up_sampling[-2].stride
    padding = netG_tmp.up_sampling[-2].padding
    bias = netG_tmp.up_sampling[-2].bias is not None
    netG_tmp.up_sampling[-2] = nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=bias)

    netG_tmp.load_state_dict(pretrained_studentG_state)

    model.netG_student = netG_tmp
    if len(opt.gpu_ids) > 1:
        model.netG_student = torch.nn.DataParallel(
            model.netG_student, opt.gpu_ids).to(model.device)
    else:
        model.netG_student = model.netG_student.to(model.device)
    model_profiling(mc.unwrap_model(model.netG_student),
                    opt.data_height,
                    opt.data_width,
                    num_forwards=0,
                    verbose=opt.prune_logging_verbose)
    G_params = []
    netAs = []
    for netA in model.netAs:
        netA_new = nn.Conv2d(in_channels=ngf_netA,
                             out_channels=netA.out_channels,
                             kernel_size=netA.kernel_size).to(model.device)
        G_params.append(netA_new.parameters())
        netAs.append(netA_new)
    model.netAs = netAs
    model.add_mapping_hook()

    model.optimizer_G = Adam([{
        'params': model.netG_student.parameters()
    }, {
        'params': itertools.chain(*G_params)
    }],
                             lr=opt.lr,
                             betas=(opt.beta1, 0.999))
    model.optimizers = [model.optimizer_G, model.optimizer_D]
    if model.isTrain:
        model.schedulers = [
            networks.get_scheduler(optimizer, opt)
            for optimizer in model.optimizers
        ]

    del netG_tmp

    print('Pretrained studentG state is loaded.')


def load_pretrained_spade_student(model, opt):
    pretrained_studentG_state = torch.load(opt.pretrained_student_G_path)
    modules_on_one_gpu = model.modules_on_one_gpu
    netG_tmp = copy.deepcopy(modules_on_one_gpu.netG_teacher)
    spade_config_str = opt.teacher_norm_G.replace('spectral', '')
    if spade_config_str.startswith('spade'):
        parsed = re.search(r'spade(\D+)(\d)x\d', spade_config_str)
        param_free_norm_type = str(parsed.group(1))
    else:
        raise NotImplementedError
    norm_layer = {
        'instance': nn.InstanceNorm2d,
        'batch': nn.BatchNorm2d,
        'syncbatch': SynchronizedBatchNorm2d
    }[param_free_norm_type]

    in_channels = netG_tmp.fc.in_channels
    out_channels = pretrained_studentG_state['fc.weight'].shape[0]
    kernel_size = netG_tmp.fc.kernel_size
    stride = netG_tmp.fc.stride
    padding = netG_tmp.fc.padding
    bias = netG_tmp.fc.bias is not None
    netG_tmp.fc = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias)

    out_channels = pretrained_studentG_state['fc_norm.weight'].shape[0]
    netG_tmp.fc_norm = norm_layer(out_channels, affine=True)
    ngf_stu = out_channels // 16

    in_channels = out_channels

    if opt.num_upsampling_layers == 'most':
        features = ['head_0'] + [f'G_middle_{i}' for i in range(2)
                                 ] + [f'up_{i}' for i in range(5)]
    else:
        features = ['head_0'] + [f'G_middle_{i}' for i in range(2)
                                 ] + [f'up_{i}' for i in range(4)]
    for layer_name in features:
        layer = getattr(netG_tmp, layer_name)
        layer.input_dim = in_channels
        if 'up' in layer_name:
            out_channels = in_channels // 2
        else:
            out_channels = in_channels
        layer.output_dim = out_channels
        layer.res_channels, layer.res_kernel_sizes = [], []
        layer.dw_channels, layer.dw_kernel_sizes = [], []
        for k, v in pretrained_studentG_state.items():
            if f'{layer_name}.res_ops' in k and '.0.conv.weight' in k:
                ch_, _, k_, _ = v.shape
                layer.res_channels.append(ch_)
                layer.res_kernel_sizes.append(k_)
            if f'{layer_name}.dw_ops' in k and '.1.conv.weight' in k:
                ch_, _, k_, _ = v.shape
                layer.dw_channels.append(ch_)
                layer.dw_kernel_sizes.append(k_)
        layer.spade.output_dim = layer.input_dim
        layer.spade.res_channels, layer.spade.res_kernel_sizes = [], []
        layer.spade.dw_channels, layer.spade.dw_kernel_sizes = [], []
        for k, v in pretrained_studentG_state.items():
            if f'{layer_name}.spade.res_ops' in k and '.0.conv.weight' in k:
                ch_, _, k_, _ = v.shape
                layer.spade.res_channels.append(ch_)
                layer.spade.res_kernel_sizes.append(k_)
            if f'{layer_name}.spade.dw_ops' in k and '.1.conv.weight' in k:
                ch_, _, k_, _ = v.shape
                layer.spade.dw_channels.append(ch_)
                layer.spade.dw_kernel_sizes.append(k_)
        layer.res_ops, layer.dw_ops, layer.shortcut, layer.spade = layer._build(
            build_only=True)
        in_channels = out_channels
    out_channels = netG_tmp.conv_img.out_channels
    kernel_size = netG_tmp.conv_img.kernel_size
    stride = netG_tmp.conv_img.stride
    padding = netG_tmp.conv_img.padding
    bias = netG_tmp.conv_img.bias is not None
    netG_tmp.conv_img = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=bias)

    netG_tmp.load_state_dict(pretrained_studentG_state)

    modules_on_one_gpu.netG_student = netG_tmp
    modules_on_one_gpu.netG_student = modules_on_one_gpu.netG_student.to(
        model.device)
    model_profiling(modules_on_one_gpu.netG_student,
                    opt.data_height,
                    opt.data_width,
                    channel=opt.data_channel,
                    num_forwards=0,
                    verbose=True)

    netAs = nn.ModuleList()
    for i, mapping_layer in enumerate(modules_on_one_gpu.mapping_layers):
        if mapping_layer != 'up_1':
            fs, ft = ngf_stu * 16, opt.teacher_ngf * 16
        else:
            fs, ft = ngf_stu * 4, opt.teacher_ngf * 4
        netA_new = nn.Conv2d(in_channels=fs, out_channels=ft, kernel_size=1)
        netAs.append(netA_new)
    modules_on_one_gpu.netAs = netAs.to(model.device)

    if opt.no_TTUR:
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = opt.lr, opt.lr
    else:
        beta1, beta2 = 0, 0.9
        G_lr, D_lr = opt.lr / 2, opt.lr * 2
    G_params = list(modules_on_one_gpu.netG_student.parameters())
    for netA in modules_on_one_gpu.netAs:
        G_params += list(netA.parameters())
    modules_on_one_gpu.optimizer_G = Adam(G_params,
                                          lr=G_lr,
                                          betas=(beta1, beta2))
    model.optimizer_G = modules_on_one_gpu.optimizer_G
    model.optimizers = [model.optimizer_G, model.optimizer_D]
    if model.isTrain:
        model.schedulers = [
            networks.get_scheduler(optimizer, opt)
            for optimizer in model.optimizers
        ]

    del netG_tmp


def shrink_model(model, target_flops, opt):
    torch.cuda.synchronize()
    time_before_prune = time.time()
    model.remove_mapping_hook()
    netG_tmp = copy.deepcopy(mc.unwrap_model(model.netG_teacher))
    norm_layer = {
        'instance': nn.InstanceNorm2d,
        'batch': nn.BatchNorm2d,
        'syncbatch': SynchronizedBatchNorm2d
    }[opt.norm]
    ds_idx_list = []
    ds_weight_list = []
    ft_weight_list = []
    us_idx_list = []
    us_weight_list = []
    for idx, layer in enumerate(netG_tmp.down_sampling):
        if isinstance(layer, norm_layer):
            ds_idx_list.append(idx)
            ds_weight_list += [layer.weight]
    bn_weights_to_prune = prune.get_bn_to_prune(netG_tmp)
    ft_weight_list = get_prune_weights(netG_tmp, bn_weights_to_prune)
    for idx, layer in enumerate(netG_tmp.up_sampling):
        if isinstance(layer, norm_layer):
            us_idx_list.append(idx)
            us_weight_list += [layer.weight]
    all_weights = torch.cat(ds_weight_list + ft_weight_list + us_weight_list)
    scale_lb, scale_ub = all_weights.detach().abs().min(), all_weights.detach(
    ).abs().max()
    print(f'scale range: [{scale_lb}, {scale_ub}]')
    searched_flops = float('inf')
    while (abs(scale_ub - scale_lb) > 1e-3 * scale_lb) or (searched_flops >
                                                           target_flops):
        netG_to_prune = copy.deepcopy(netG_tmp)
        scale_threshold = (scale_lb + scale_ub) / 2
        in_channels = None
        for idx in ds_idx_list:
            mask = netG_to_prune.down_sampling[idx].weight.detach().abs(
            ) > scale_threshold
            out_channels = mask.detach().sum().item()
            out_channels = max(out_channels, getattr(opt, 'prune_cin_lb', 1))
            if idx == ds_idx_list[0]:
                out_channels = min(out_channels,
                                   getattr(opt, 'prune_cin_ub', float('inf')))
            if idx == ds_idx_list[-1]:
                out_channels = max(out_channels,
                                   getattr(opt, 'prune_ft_cin_lb', 1))
            netG_to_prune.down_sampling[idx] = norm_layer(
                out_channels,
                affine=opt.norm_affine,
                track_running_stats=opt.norm_track_running_stats)
            if in_channels is None:
                in_channels = netG_to_prune.down_sampling[idx - 1].in_channels
            kernel_size = netG_to_prune.down_sampling[idx - 1].kernel_size
            stride = netG_to_prune.down_sampling[idx - 1].stride
            padding = netG_to_prune.down_sampling[idx - 1].padding
            bias = netG_to_prune.down_sampling[idx - 1].bias is not None
            netG_to_prune.down_sampling[idx - 1] = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias)
            in_channels = out_channels
        for idx, layer in enumerate(netG_to_prune.features):
            layer.input_dim = in_channels
            layer.res_channels = [
                sum(bn.weight.detach().abs() > scale_threshold).item()
                for bn in layer.get_first_res_bn()
            ]
            layer.dw_channels = [
                sum(bn.weight.detach().abs() > scale_threshold).item()
                for bn in layer.get_first_dw_bn()
            ]
            layer.res_ops, layer.dw_ops, layer.pw_bn = layer._build()
        for idx in us_idx_list:
            mask = netG_to_prune.up_sampling[idx].weight.detach().abs(
            ) > scale_threshold
            out_channels = mask.detach().sum().item()
            out_channels = max(out_channels, getattr(opt, 'prune_cin_lb', 1))
            netG_to_prune.up_sampling[idx] = norm_layer(
                out_channels,
                affine=opt.norm_affine,
                track_running_stats=opt.norm_track_running_stats)
            kernel_size = netG_to_prune.up_sampling[idx - 1].kernel_size
            stride = netG_to_prune.up_sampling[idx - 1].stride
            padding = netG_to_prune.up_sampling[idx - 1].padding
            output_padding = netG_to_prune.up_sampling[idx - 1].output_padding
            bias = netG_to_prune.up_sampling[idx - 1].bias is not None
            netG_to_prune.up_sampling[idx - 1] = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias)
            in_channels = out_channels
        out_channels = netG_to_prune.up_sampling[-2].out_channels
        kernel_size = netG_to_prune.up_sampling[-2].kernel_size
        stride = netG_to_prune.up_sampling[-2].stride
        padding = netG_to_prune.up_sampling[-2].padding
        bias = netG_to_prune.up_sampling[-2].bias is not None
        netG_to_prune.up_sampling[-2] = nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  bias=bias)
        if len(opt.gpu_ids) > 1:
            netG_to_prune = torch.nn.DataParallel(netG_to_prune,
                                                  opt.gpu_ids).to(model.device)
        else:
            netG_to_prune = netG_to_prune.to(model.device)
        model_profiling(mc.unwrap_model(netG_to_prune),
                        opt.data_height,
                        opt.data_width,
                        num_forwards=0,
                        verbose=opt.prune_logging_verbose)
        searched_flops = mc.unwrap_model(netG_to_prune).n_macs
        if searched_flops > target_flops:
            scale_lb = scale_threshold
        else:
            scale_ub = scale_threshold

        del netG_to_prune

    print(
        f'scale threshold: {scale_threshold}, searched flops: {searched_flops}, target flops: {target_flops}, flops diff: {searched_flops - target_flops}.'
    )

    netG_to_prune = copy.deepcopy(netG_tmp)
    in_channels = None
    in_mask = None

    for idx in ds_idx_list:
        out_mask = netG_to_prune.down_sampling[idx].weight.detach().abs(
        ) > scale_threshold
        out_channels = out_mask.detach().sum().item()
        if out_channels < getattr(opt, 'prune_cin_lb', 1):
            private_scale_threshold = torch.sort(
                netG_to_prune.down_sampling[idx].weight.detach().abs().view(
                    -1),
                descending=True)[0][getattr(opt, 'prune_cin_lb', 1) - 1]
            out_mask = netG_to_prune.down_sampling[idx].weight.detach().abs(
            ) >= private_scale_threshold
            out_channels = out_mask.detach().sum().item()
        if idx == ds_idx_list[0]:
            if out_channels > getattr(opt, 'prune_cin_ub', float('inf')):
                private_scale_threshold = torch.sort(
                    netG_to_prune.down_sampling[idx].weight.detach().abs(
                    ).view(-1),
                    descending=False)[0][getattr(opt, 'prune_cin_ub', 1) - 1]
                out_mask = netG_to_prune.down_sampling[idx].weight.detach(
                ).abs() <= private_scale_threshold
                out_channels = out_mask.detach().sum().item()
        if idx == ds_idx_list[-1]:
            if out_channels < getattr(opt, 'prune_ft_cin_lb', 1):
                private_scale_threshold = torch.sort(
                    netG_to_prune.down_sampling[idx].weight.detach().abs(
                    ).view(-1),
                    descending=True)[0][getattr(opt, 'prune_ft_cin_lb', 1) - 1]
                out_mask = netG_to_prune.down_sampling[idx].weight.detach(
                ).abs() >= private_scale_threshold
                out_channels = out_mask.detach().sum().item()
        netG_to_prune.down_sampling[idx] = norm_layer(
            out_channels,
            affine=opt.norm_affine,
            track_running_stats=opt.norm_track_running_stats)
        netG_to_prune.down_sampling[idx].weight.data.copy_(
            netG_tmp.down_sampling[idx].weight.data[out_mask])
        netG_to_prune.down_sampling[idx].bias.data.copy_(
            netG_tmp.down_sampling[idx].bias.data[out_mask])
        if netG_tmp.down_sampling[idx].track_running_stats:
            assert netG_to_prune.down_sampling[idx].track_running_stats
            netG_to_prune.down_sampling[idx].running_mean.data.copy_(
                netG_tmp.down_sampling[idx].running_mean.data[out_mask])
            netG_to_prune.down_sampling[idx].running_var.data.copy_(
                netG_tmp.down_sampling[idx].running_var.data[out_mask])
            netG_to_prune.down_sampling[idx].num_batches_tracked.data.copy_(
                netG_tmp.down_sampling[idx].num_batches_tracked.data)
        if in_channels is None:
            in_channels = netG_to_prune.down_sampling[idx - 1].in_channels
        kernel_size = netG_to_prune.down_sampling[idx - 1].kernel_size
        stride = netG_to_prune.down_sampling[idx - 1].stride
        padding = netG_to_prune.down_sampling[idx - 1].padding
        bias = netG_to_prune.down_sampling[idx - 1].bias is not None
        netG_to_prune.down_sampling[idx - 1] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        if in_mask is None:
            netG_to_prune.down_sampling[idx - 1].weight.data.copy_(
                netG_tmp.down_sampling[idx - 1].weight.data[out_mask])
        else:
            netG_to_prune.down_sampling[idx - 1].weight.data.copy_(
                netG_tmp.down_sampling[idx - 1].weight.data[out_mask][:,
                                                                      in_mask])
        if netG_to_prune.down_sampling[idx - 1].bias is not None:
            netG_to_prune.down_sampling[idx - 1].bias.data.copy_(
                netG_tmp.down_sampling[idx - 1].bias.data[out_mask])
        in_channels = out_channels
        in_mask = out_mask

    ngf_netA = in_channels
    for idx, layer in enumerate(netG_to_prune.features):
        layer.input_dim = in_channels
        layer.res_channels = [
            sum(bn.weight.detach().abs() > scale_threshold).item()
            for bn in layer.get_first_res_bn()
        ]
        layer.dw_channels = [
            sum(bn.weight.detach().abs() > scale_threshold).item()
            for bn in layer.get_first_dw_bn()
        ]
        layer.res_ops, layer.dw_ops, layer.pw_bn = layer._build()
        op_idx = 0
        for old_op in netG_tmp.features[idx].res_ops:
            mid_mask = old_op[1][1].weight.detach().abs() > scale_threshold
            if sum(mid_mask) == 0:
                continue
            new_op = layer.res_ops[op_idx]
            new_op[1][0].weight.data.copy_(
                old_op[1][0].weight.data[mid_mask][:, in_mask])
            if new_op[1][0].bias is not None:
                new_op[1][0].bias.data.copy_(old_op[1][0].bias.data[mid_mask])
            new_op[1][1].weight.data.copy_(old_op[1][1].weight.data[mid_mask])
            new_op[1][1].bias.data.copy_(old_op[1][1].bias.data[mid_mask])
            if old_op[1][1].track_running_stats:
                assert new_op[1][1].track_running_stats
                new_op[1][1].running_mean.data.copy_(
                    old_op[1][1].running_mean.data[mid_mask])
                new_op[1][1].running_var.data.copy_(
                    old_op[1][1].running_var.data[mid_mask])
                new_op[1][1].num_batches_tracked.data.copy_(
                    old_op[1][1].num_batches_tracked.data)
            new_op[4].weight.data.copy_(
                old_op[4].weight.data[in_mask][:, mid_mask])
            if new_op[4].bias is not None:
                new_op[4].bias.data.copy_(old_op[4].bias.data[in_mask])
            op_idx += 1
        assert len(layer.res_ops) == op_idx
        op_idx = 0
        for old_op in netG_tmp.features[idx].dw_ops:
            mid_mask = old_op[0][1].weight.detach().abs() > scale_threshold
            if sum(mid_mask) == 0:
                continue
            new_op = layer.dw_ops[op_idx]
            new_op[0][0].weight.data.copy_(
                old_op[0][0].weight.data[mid_mask][:, in_mask])
            if new_op[0][0].bias is not None:
                new_op[0][0].bias.data.copy_(old_op[0][0].bias.data[mid_mask])
            new_op[0][1].weight.data.copy_(old_op[0][1].weight.data[mid_mask])
            new_op[0][1].bias.data.copy_(old_op[0][1].bias.data[mid_mask])
            if old_op[0][1].track_running_stats:
                assert new_op[0][1].track_running_stats
                new_op[0][1].running_mean.data.copy_(
                    old_op[0][1].running_mean.data[mid_mask])
                new_op[0][1].running_var.data.copy_(
                    old_op[0][1].running_var.data[mid_mask])
                new_op[0][1].num_batches_tracked.data.copy_(
                    old_op[0][1].num_batches_tracked.data)
            new_op[2][0].weight.data.copy_(old_op[2][0].weight.data[mid_mask])
            if new_op[2][0].bias is not None:
                new_op[2][0].bias.data.copy_(old_op[2][0].bias.data[mid_mask])
            new_op[2][1].weight.data.copy_(old_op[2][1].weight.data[mid_mask])
            new_op[2][1].bias.data.copy_(old_op[2][1].bias.data[mid_mask])
            if old_op[2][1].track_running_stats:
                assert new_op[2][1].track_running_stats
                new_op[2][1].running_mean.data.copy_(
                    old_op[2][1].running_mean.data[mid_mask])
                new_op[2][1].running_var.data.copy_(
                    old_op[2][1].running_var.data[mid_mask])
                new_op[2][1].num_batches_tracked.data.copy_(
                    old_op[2][1].num_batches_tracked.data)
            new_op[4].weight.data.copy_(
                old_op[4].weight.data[in_mask][:, mid_mask])
            if new_op[4].bias is not None:
                new_op[4].bias.data.copy_(old_op[4].bias.data[in_mask])
            op_idx += 1
        assert len(layer.dw_ops) == op_idx

    for idx in us_idx_list:
        out_mask = netG_to_prune.up_sampling[idx].weight.detach().abs(
        ) > scale_threshold
        out_channels = out_mask.detach().sum().item()
        if out_channels < getattr(opt, 'prune_cin_lb', 1):
            private_scale_threshold = torch.sort(
                netG_to_prune.up_sampling[idx].weight.detach().abs().view(-1),
                descending=True)[0][getattr(opt, 'prune_cin_lb', 1) - 1]
            out_mask = netG_to_prune.up_sampling[idx].weight.detach().abs(
            ) >= private_scale_threshold
            out_channels = out_mask.detach().sum().item()
        netG_to_prune.up_sampling[idx] = norm_layer(
            out_channels,
            affine=opt.norm_affine,
            track_running_stats=opt.norm_track_running_stats)
        netG_to_prune.up_sampling[idx].weight.data.copy_(
            netG_tmp.up_sampling[idx].weight.data[out_mask])
        netG_to_prune.up_sampling[idx].bias.data.copy_(
            netG_tmp.up_sampling[idx].bias.data[out_mask])
        if netG_tmp.up_sampling[idx].track_running_stats:
            assert netG_to_prune.up_sampling[idx].track_running_stats
            netG_to_prune.up_sampling[idx].running_mean.data.copy_(
                netG_tmp.up_sampling[idx].running_mean.data[out_mask])
            netG_to_prune.up_sampling[idx].running_var.data.copy_(
                netG_tmp.up_sampling[idx].running_var.data[out_mask])
            netG_to_prune.up_sampling[idx].num_batches_tracked.data.copy_(
                netG_tmp.up_sampling[idx].num_batches_tracked.data)
        kernel_size = netG_to_prune.up_sampling[idx - 1].kernel_size
        stride = netG_to_prune.up_sampling[idx - 1].stride
        padding = netG_to_prune.up_sampling[idx - 1].padding
        output_padding = netG_to_prune.up_sampling[idx - 1].output_padding
        bias = netG_to_prune.up_sampling[idx - 1].bias is not None
        netG_to_prune.up_sampling[idx - 1] = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias)
        netG_to_prune.up_sampling[idx - 1].weight.data.copy_(
            netG_tmp.up_sampling[idx - 1].weight.data[in_mask][:, out_mask])
        if netG_to_prune.up_sampling[idx - 1].bias is not None:
            netG_to_prune.up_sampling[idx - 1].bias.data.copy_(
                netG_tmp.up_sampling[idx - 1].bias.data[out_mask])
        in_channels = out_channels
        in_mask = out_mask
    out_channels = netG_to_prune.up_sampling[-2].out_channels
    kernel_size = netG_to_prune.up_sampling[-2].kernel_size
    stride = netG_to_prune.up_sampling[-2].stride
    padding = netG_to_prune.up_sampling[-2].padding
    bias = netG_to_prune.up_sampling[-2].bias is not None
    netG_to_prune.up_sampling[-2] = nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              bias=bias)
    netG_to_prune.up_sampling[-2].weight.data.copy_(
        netG_tmp.up_sampling[-2].weight.data[:, in_mask])
    if netG_to_prune.up_sampling[-2].bias is not None:
        netG_to_prune.up_sampling[-2].bias.data.copy_(
            netG_tmp.up_sampling[-2].bias.data)

    model.netG_student = netG_to_prune
    torch.cuda.synchronize()
    time_after_prune = time.time()
    pruning_time = time_after_prune - time_before_prune
    if len(opt.gpu_ids) > 1:
        model.netG_student = torch.nn.DataParallel(
            model.netG_student, opt.gpu_ids).to(model.device)
    else:
        model.netG_student = model.netG_student.to(model.device)
    model_profiling(mc.unwrap_model(model.netG_student),
                    opt.data_height,
                    opt.data_width,
                    num_forwards=0,
                    verbose=opt.prune_logging_verbose)
    G_params = []
    netAs = []
    for netA in model.netAs:
        netA_new = nn.Conv2d(in_channels=ngf_netA,
                             out_channels=netA.out_channels,
                             kernel_size=netA.kernel_size).to(model.device)
        G_params.append(netA_new.parameters())
        netAs.append(netA_new)
    model.netAs = netAs
    model.add_mapping_hook()

    model.optimizer_G = Adam([{
        'params': model.netG_student.parameters()
    }, {
        'params': itertools.chain(*G_params)
    }],
                             lr=opt.lr,
                             betas=(opt.beta1, 0.999))
    model.optimizers = [model.optimizer_G, model.optimizer_D]
    if model.isTrain:
        model.schedulers = [
            networks.get_scheduler(optimizer, opt)
            for optimizer in model.optimizers
        ]

    del netG_tmp, netG_to_prune

    print('All layers are pruned.')

    return pruning_time


def shrink_spade_model(model, target_flops, opt):
    torch.cuda.synchronize()
    time_before_prune = time.time()
    modules_on_one_gpu = model.modules_on_one_gpu
    netG_tmp = copy.deepcopy(modules_on_one_gpu.netG_teacher)
    spade_config_str = opt.teacher_norm_G.replace('spectral', '')
    if spade_config_str.startswith('spade'):
        parsed = re.search(r'spade(\D+)(\d)x\d', spade_config_str)
        param_free_norm_type = str(parsed.group(1))
    else:
        raise NotImplementedError
    norm_layer = {
        'instance': nn.InstanceNorm2d,
        'batch': nn.BatchNorm2d,
        'syncbatch': SynchronizedBatchNorm2d
    }[param_free_norm_type]
    fc_norm_weight_list = [netG_tmp.fc_norm.weight]
    bn_weights_to_prune = prune.get_bn_to_prune(netG_tmp, spade=True)
    ft_weight_list = get_prune_weights(netG_tmp, bn_weights_to_prune)
    all_weights = torch.cat(fc_norm_weight_list + ft_weight_list)
    scale_lb, scale_ub = all_weights.detach().abs().min(), all_weights.detach(
    ).abs().max()
    print(f'scale range: [{scale_lb}, {scale_ub}]')
    searched_flops = float('inf')
    while (abs(scale_ub - scale_lb) > 1e-3 * scale_lb) or (searched_flops >
                                                           target_flops):
        netG_to_prune = copy.deepcopy(netG_tmp)
        scale_threshold = (scale_lb + scale_ub) / 2
        ch_div = 16
        if opt.num_upsampling_layers == 'most':
            ch_div = 32
        mask = netG_to_prune.fc_norm.weight.detach().abs() > scale_threshold
        out_channels = mask.detach().sum().item()
        out_channels = max(out_channels // ch_div,
                           getattr(opt, 'prune_cin_lb', 1)) * ch_div
        out_channels = min(out_channels // ch_div,
                           getattr(opt, 'prune_cin_ub', float('inf'))) * ch_div
        ngf_stu = out_channels // 16
        netG_to_prune.fc_norm = norm_layer(out_channels, affine=True)
        in_channels = netG_to_prune.fc.in_channels
        kernel_size = netG_to_prune.fc.kernel_size
        stride = netG_to_prune.fc.stride
        padding = netG_to_prune.fc.padding
        bias = netG_to_prune.fc.bias is not None
        netG_to_prune.fc = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        in_channels = out_channels

        if opt.num_upsampling_layers == 'most':
            features = ['head_0'] + [f'G_middle_{i}' for i in range(2)
                                     ] + [f'up_{i}' for i in range(5)]
        else:
            features = ['head_0'] + [f'G_middle_{i}' for i in range(2)
                                     ] + [f'up_{i}' for i in range(4)]
        for layer_name in features:
            layer = getattr(netG_to_prune, layer_name)
            layer.input_dim = in_channels
            if 'up' in layer_name:
                out_channels = in_channels // 2
            else:
                out_channels = in_channels
            layer.output_dim = out_channels
            layer.res_channels = [
                sum(bn.weight.detach().abs() > scale_threshold).item()
                for bn in layer.get_first_res_bn()
            ]
            layer.dw_channels = [
                sum(bn.weight.detach().abs() > scale_threshold).item()
                for bn in layer.get_first_dw_bn()
            ]
            layer.spade.output_dim = layer.input_dim
            layer.spade.res_channels = [
                sum(bn.weight.detach().abs() > scale_threshold).item()
                for bn in layer.spade.get_first_res_bn()
            ]
            layer.spade.dw_channels = [
                sum(bn.weight.detach().abs() > scale_threshold).item()
                for bn in layer.spade.get_first_dw_bn()
            ]
            layer.res_ops, layer.dw_ops, layer.shortcut, layer.spade = layer._build(
                build_only=True)
            in_channels = out_channels
        out_channels = netG_to_prune.conv_img.out_channels
        kernel_size = netG_to_prune.conv_img.kernel_size
        stride = netG_to_prune.conv_img.stride
        padding = netG_to_prune.conv_img.padding
        bias = netG_to_prune.conv_img.bias is not None
        netG_to_prune.conv_img = nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=bias)
        model_profiling(netG_to_prune,
                        opt.data_height,
                        opt.data_width,
                        channel=opt.data_channel,
                        num_forwards=0,
                        verbose=False)
        searched_flops = netG_to_prune.n_macs
        if searched_flops > target_flops:
            scale_lb = scale_threshold
        else:
            scale_ub = scale_threshold

    print(
        f'scale threshold: {scale_threshold}, searched flops: {searched_flops}, target flops: {target_flops}, flops diff: {searched_flops - target_flops}.'
    )

    modules_on_one_gpu.netG_student = netG_to_prune
    modules_on_one_gpu.netG_student = modules_on_one_gpu.netG_student.to(
        model.device)
    torch.cuda.synchronize()
    time_after_prune = time.time()
    pruning_time = time_after_prune - time_before_prune
    model_profiling(modules_on_one_gpu.netG_student,
                    opt.data_height,
                    opt.data_width,
                    channel=opt.data_channel,
                    num_forwards=0,
                    verbose=True)
    netAs = nn.ModuleList()
    for i, mapping_layer in enumerate(modules_on_one_gpu.mapping_layers):
        if mapping_layer != 'up_1':
            fs, ft = ngf_stu * 16, opt.teacher_ngf * 16
        else:
            fs, ft = ngf_stu * 4, opt.teacher_ngf * 4
        netA_new = nn.Conv2d(in_channels=fs, out_channels=ft, kernel_size=1)
        netAs.append(netA_new)
    modules_on_one_gpu.netAs = netAs.to(model.device)

    if opt.no_TTUR:
        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = opt.lr, opt.lr
    else:
        beta1, beta2 = 0, 0.9
        G_lr, D_lr = opt.lr / 2, opt.lr * 2
    G_params = list(modules_on_one_gpu.netG_student.parameters())
    for netA in modules_on_one_gpu.netAs:
        G_params += list(netA.parameters())
    modules_on_one_gpu.optimizer_G = Adam(G_params,
                                          lr=G_lr,
                                          betas=(beta1, beta2))
    model.optimizer_G = modules_on_one_gpu.optimizer_G
    model.optimizers = [model.optimizer_G, model.optimizer_D]
    if model.isTrain:
        model.schedulers = [
            networks.get_scheduler(optimizer, opt)
            for optimizer in model.optimizers
        ]

    del netG_tmp, netG_to_prune

    print('All layers are pruned.')

    return pruning_time


def shrink(model, opt):
    target_flops = getattr(opt, 'target_flops', 0.0)
    assert target_flops > 0
    if 'spade' in opt.distiller:
        return shrink_spade_model(model, target_flops, opt)
    else:
        return shrink_model(model, target_flops, opt)
