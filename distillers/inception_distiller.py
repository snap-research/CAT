"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import ntpath
import os
from collections import OrderedDict
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel.data_parallel import gather, replicate, parallel_apply
from tqdm import tqdm

from metric import get_fid, get_mIoU
from utils import util
from utils.weight_transfer import load_pretrained_weight
from utils.model_profiling import model_profiling
from utils import prune
from utils.common import KA

from .base_inception_distiller import BaseInceptionDistiller


class InceptionDistiller(BaseInceptionDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(InceptionDistiller,
                       InceptionDistiller).modify_commandline_options(
                           parser, is_train)
        parser.add_argument('--restore_pretrained_G_path',
                            type=str,
                            default=None,
                            help='the path to restore pretrained G')
        parser.add_argument('--pretrained_netG',
                            type=str,
                            default='inception_9blocks',
                            help='specify pretrained generator architecture',
                            choices=['inception_9blocks'])
        parser.add_argument('--pretrained_ngf', type=int, default=64)
        parser.add_argument('--target_flops',
                            type=float,
                            default=0,
                            help='target flops')
        parser.add_argument('--prune_cin_lb',
                            type=int,
                            default=0,
                            help='lower bound for input channel number')
        parser.add_argument('--pretrained_student_G_path',
                            type=str,
                            default=None,
                            help='the path for pretrained student G')
        parser.add_argument('--prune_only',
                            action='store_true',
                            help='prune without training')
        parser.add_argument('--prune_continue',
                            action='store_true',
                            help='continue training after pruning all layers')
        parser.add_argument('--prune_logging_verbose',
                            action='store_true',
                            help='logging verbose for pruning')

        parser.set_defaults(norm='instance',
                            dataset_mode='aligned',
                            log_dir='logs/inception',
                            teacher_netG='inception_9blocks',
                            student_netG='inception_9blocks')
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(InceptionDistiller, self).__init__(opt)
        self.best_fid = 1e9
        self.best_mIoU = -1e9
        self.fids, self.mIoUs = [], []
        self.npz = np.load(opt.real_stat_path)
        model_profiling(getattr(self.netG_teacher, 'module',
                                self.netG_teacher),
                        self.opt.data_height,
                        self.opt.data_width,
                        channel=self.opt.data_channel,
                        num_forwards=0,
                        verbose=self.opt.prune_logging_verbose)
        model_profiling(getattr(self.netG_student, 'module',
                                self.netG_student),
                        self.opt.data_height,
                        self.opt.data_width,
                        channel=self.opt.data_channel,
                        num_forwards=0,
                        verbose=self.opt.prune_logging_verbose)

    def forward(self, teacher_forward=True):
        if teacher_forward:
            with torch.no_grad():
                self.Tfake_B = self.netG_teacher(self.real_A)
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, nn.Conv2d)
            n = self.mapping_layers[i]
            if self.opt.distill_G_loss_type == 'mse':
                if len(self.gpu_ids) > 1:
                    netA_replicas = replicate(netA, self.gpu_ids)
                    kwargs = tuple([{} for idx in self.gpu_ids])
                    Sacts = parallel_apply(
                        netA_replicas,
                        tuple([
                            self.Sacts[key]
                            for key in sorted(self.Sacts.keys()) if n in key
                        ]), kwargs)
                    Tacts = [
                        self.Tacts[key] for key in sorted(self.Tacts.keys())
                        if n in key
                    ]
                    loss = [
                        F.mse_loss(Sact, Tact)
                        for Sact, Tact in zip(Sacts, Tacts)
                    ]
                    loss = gather(loss, self.gpu_ids[0]).sum()
                else:
                    Tact = self.Tacts[n + str(netA.weight.device)]
                    Sact = self.Sacts[n + str(netA.weight.device)]
                    Sact = netA(Sact)
                    loss = F.mse_loss(Sact, Tact)
            elif self.opt.distill_G_loss_type == 'ka':
                if len(self.gpu_ids) > 1:
                    Sacts = [
                        self.Sacts[key] for key in sorted(self.Sacts.keys())
                        if n in key
                    ]
                    Tacts = [
                        self.Tacts[key] for key in sorted(self.Tacts.keys())
                        if n in key
                    ]
                    loss = [
                        -KA(Sact, Tact) for Sact, Tact in zip(Sacts, Tacts)
                    ]
                    loss = gather(loss, self.gpu_ids[0]).sum()
                else:
                    Tact = self.Tacts[n + str(netA.weight.device)]
                    Sact = self.Sacts[n + str(netA.weight.device)]
                    loss = -KA(Sact, Tact)
            else:
                raise NotImplementedError
            setattr(self, 'loss_G_distill%d' % i, loss)
            losses.append(loss)
        return sum(losses)

    def backward_G(self, steps):
        if self.opt.dataset_mode == 'aligned':
            self.loss_G_recon = self.criterionRecon(
                self.Sfake_B, self.real_B) * self.opt.lambda_recon
            fake = torch.cat((self.real_A, self.Sfake_B), 1)
        else:
            self.loss_G_recon = self.criterionRecon(
                self.Sfake_B, self.Tfake_B) * self.opt.lambda_recon
            fake = self.Sfake_B
        pred_fake = self.netD(fake)
        self.loss_G_gan = self.criterionGAN(
            pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        if self.opt.lambda_distill > 0:
            self.loss_G_distill = self.calc_distill_loss(
            ) * self.opt.lambda_distill
        else:
            self.loss_G_distill = 0
        self.loss_G = self.loss_G_gan + self.loss_G_recon + self.loss_G_distill
        self.loss_G.backward()

    def optimize_parameters(self, steps):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(steps)
        self.optimizer_G.step()

    def load_networks(self,
                      verbose=True,
                      teacher_only=False,
                      restore_pretrain=True):
        if self.opt.restore_pretrained_G_path is not None:
            util.load_network(self.netG_pretrained,
                              self.opt.restore_pretrained_G_path, verbose)
            load_pretrained_weight(self.opt.pretrained_netG,
                                   self.opt.student_netG, self.netG_pretrained,
                                   self.netG_student, self.opt.pretrained_ngf,
                                   self.opt.student_ngf)
            del self.netG_pretrained
        super(InceptionDistiller, self).load_networks()

    def evaluate_model(self, step, save_image=False):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG_student.eval()
        fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader)):
            if self.opt.dataset_mode == 'aligned':
                self.set_input(data_i)
            else:
                self.set_single_input(data_i)
            self.test()
            fakes.append(self.Sfake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10 or save_image:
                    input_im = util.tensor2im(self.real_A[j])
                    Sfake_im = util.tensor2im(self.Sfake_B[j])
                    Tfake_im = util.tensor2im(self.Tfake_B[j])
                    util.save_image(input_im,
                                    os.path.join(save_dir, 'input', '%s.png') %
                                    name,
                                    create_dir=True)
                    util.save_image(Sfake_im,
                                    os.path.join(save_dir, 'Sfake',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(Tfake_im,
                                    os.path.join(save_dir, 'Tfake',
                                                 '%s.png' % name),
                                    create_dir=True)
                    if self.opt.dataset_mode == 'aligned':
                        real_im = util.tensor2im(self.real_B[j])
                        util.save_image(real_im,
                                        os.path.join(save_dir, 'real',
                                                     '%s.png' % name),
                                        create_dir=True)
                cnt += 1

        fid = get_fid(fakes,
                      self.inception_model,
                      self.npz,
                      device=self.device,
                      batch_size=self.opt.eval_batch_size)
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        self.fids.append(fid)
        if len(self.fids) > 3:
            self.fids.pop(0)
        ret = {
            'metric/fid': fid,
            'metric/fid-mean': sum(self.fids) / len(self.fids),
            'metric/fid-best': self.best_fid
        }
        if 'cityscapes' in self.opt.dataroot and self.opt.direction == 'BtoA':
            mIoU = get_mIoU(fakes,
                            names,
                            self.drn_model,
                            self.device,
                            table_path=self.opt.table_path,
                            data_dir=self.opt.cityscapes_path,
                            batch_size=self.opt.eval_batch_size,
                            num_workers=self.opt.num_threads)
            if mIoU > self.best_mIoU:
                self.is_best = True
                self.best_mIoU = mIoU
            self.mIoUs.append(mIoU)
            if len(self.mIoUs) > 3:
                self.mIoUs = self.mIoUs[1:]
            ret['metric/mIoU'] = mIoU
            ret['metric/mIoU-mean'] = sum(self.mIoUs) / len(self.mIoUs)
            ret['metric/mIoU-best'] = self.best_mIoU
        self.netG_student.train()
        return ret
