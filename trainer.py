"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn

from data import create_dataloader
import common as mc
from utils.logger import Logger
from utils.common import shrink

from models.networks import init_net


def set_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, task):
        if task == 'train':
            from options.train_options import TrainOptions as Options
            from models import create_model as create_model
        elif task == 'distill':
            from options.distill_options import DistillOptions as Options
            from distillers import create_distiller as create_model
        else:
            raise NotImplementedError('Unknown task [%s]!!!' % task)
        opt = Options().parse()
        opt.tensorboard_dir = opt.log_dir if opt.tensorboard_dir is None else opt.tensorboard_dir
        print(' '.join(sys.argv))
        if opt.phase != 'train':
            warnings.warn('You are not using training set for %s!!!' % task)
        with open(os.path.join(opt.log_dir, 'opt.txt'), 'a') as f:
            f.write(' '.join(sys.argv) + '\n')
        set_seed(opt.seed)

        dataloader = create_dataloader(opt)
        dataset_size = len(dataloader.dataset)
        print('The number of training images = %d' % dataset_size)
        opt.iters_per_epoch = len(dataloader)
        if opt.dataset_mode in ['aligned', 'unaligned']:
            opt.data_channel, opt.data_height, opt.data_width = next(
                iter(dataloader))['A' if opt.direction ==
                                  'AtoB' else 'B'].shape[1:]
        elif opt.dataset_mode in ['cityscapes']:
            input_ = next(iter(dataloader))
            opt.data_height, opt.data_width = input_['label'].shape[2:]
            opt.data_channel = opt.input_nc
            if opt.contain_dontcare_label:
                opt.data_channel += 1
            if not opt.no_instance:
                opt.data_channel += input_['instance'].shape[1]
        else:
            raise NotImplementedError
        print(
            f'data shape is: channel={opt.data_channel}, height={opt.data_height}, width={opt.data_width}.'
        )

        model = create_model(opt)
        model.setup(opt)
        logger = Logger(opt)

        self.opt = opt
        self.dataloader = dataloader
        self.model = model
        self.logger = logger
        self.task = task

    def evaluate(self, epoch, iter, message):
        start_time = time.time()
        metrics = self.model.evaluate_model(iter)
        self.logger.print_current_metrics(epoch, iter, metrics,
                                          time.time() - start_time)
        self.logger.plot(metrics, iter)
        self.logger.print_info(message)
        self.model.save_networks('latest')

    def start(self):
        opt = self.opt
        dataloader = self.dataloader
        model = self.model
        modules_on_one_gpu = getattr(model, 'modules_on_one_gpu', model)
        logger = self.logger

        if self.task == 'distill':
            shrink(model, opt)
            modules_on_one_gpu.netG_student = init_net(
                modules_on_one_gpu.netG_student, opt.init_type, opt.init_gain,
                []).to(model.device)
            if getattr(opt, 'prune_continue', False):
                model.load_networks(restore_pretrain=False)
                logger.print_info('All networks loaded.')
            model.print_networks()
            if 'spade' in self.opt.distiller:
                logger.print_info(
                    f'netG student FLOPs: {mc.unwrap_model(modules_on_one_gpu.netG_student).n_macs}.'
                )
            else:
                logger.print_info(
                    f'netG student FLOPs: {mc.unwrap_model(modules_on_one_gpu.netG_student).n_macs}; down sampling: {mc.unwrap_model(modules_on_one_gpu.netG_student).down_sampling.n_macs}; features: {mc.unwrap_model(modules_on_one_gpu.netG_student).features.n_macs}; up sampling: {mc.unwrap_model(modules_on_one_gpu.netG_student).up_sampling.n_macs}.'
                )
            if getattr(opt, 'prune_only', False):
                return

        start_epoch = opt.epoch_base
        end_epoch = opt.epoch_base + opt.nepochs + opt.nepochs_decay - 1
        total_iter = opt.iter_base
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_start_time = time.time()
            for i, data_i in enumerate(dataloader):
                iter_start_time = time.time()
                model.set_input(data_i)
                model.optimize_parameters(total_iter)

                if total_iter % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    logger.print_current_errors(epoch, total_iter, losses,
                                                time.time() - iter_start_time)
                    logger.plot(losses, total_iter)

                if total_iter % opt.save_latest_freq == 0 or total_iter == opt.iter_base:
                    self.evaluate(
                        epoch, total_iter,
                        'Saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_iter))
                    if getattr(model, 'is_best', False):
                        model.save_networks('iter%d' % total_iter)
                        model.save_networks('best')
                    if getattr(model, 'is_best_A', False):
                        model.save_networks('iter%d' % total_iter)
                        model.save_networks('best_A')
                    if getattr(model, 'is_best_B', False):
                        model.save_networks('iter%d' % total_iter)
                        model.save_networks('best_B')

                total_iter += 1
            logger.print_info(
                'End of epoch %d / %d \t Time Taken: %.2f sec' %
                (epoch, end_epoch, time.time() - epoch_start_time))
            if epoch % opt.save_epoch_freq == 0 or epoch == end_epoch:
                self.evaluate(
                    epoch, total_iter,
                    'Saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_iter))
                model.save_networks(epoch)
                if getattr(model, 'is_best', False):
                    model.save_networks('iter%d' % total_iter)
                    model.save_networks('best')
                if getattr(model, 'is_best_A', False):
                    model.save_networks('iter%d' % total_iter)
                    model.save_networks('best_A')
                if getattr(model, 'is_best_B', False):
                    model.save_networks('iter%d' % total_iter)
                    model.save_networks('best_B')
            model.update_learning_rate(logger)
