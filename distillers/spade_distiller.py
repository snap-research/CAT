"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import argparse
import ntpath
import os

import torch
from tqdm import tqdm

from metric import get_fid, get_mIoU
from models import networks
from utils import util
from .base_spade_distiller import BaseSPADEDistiller


class SPADEDistiller(BaseSPADEDistiller):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = super(SPADEDistiller,
                       SPADEDistiller).modify_commandline_options(
                           parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument('--restore_pretrained_G_path',
                            type=str,
                            default=None,
                            help='the path to restore pretrained G')
        parser.add_argument('--pretrained_student_G_path',
                            type=str,
                            default=None,
                            help='the path for pretrained student G')
        parser.add_argument('--pretrained_netG',
                            type=str,
                            default='mobile_spade',
                            help='specify pretrained generator architecture',
                            choices=['inception_spade'])
        parser.add_argument(
            '--pretrained_ngf',
            type=int,
            default=64,
            help='the base number of filters of the pretrained generator')
        parser.add_argument(
            '--pretrained_norm_G',
            type=str,
            default='spadesyncbatch3x3',
            help=
            'instance normalization or batch normalization of the student model'
        )
        parser.add_argument('--target_flops',
                            type=float,
                            default=0,
                            help='target flops')
        parser.add_argument('--prune_cin_lb',
                            type=int,
                            default=1,
                            help='lower bound for input channel number')
        parser.add_argument('--prune_only',
                            action='store_true',
                            help='prune without training')
        parser.add_argument('--prune_continue',
                            action='store_true',
                            help='continue training after pruning all layers')
        parser.add_argument('--prune_logging_verbose',
                            action='store_true',
                            help='logging verbose for pruning')
        parser.set_defaults(netD='multi_scale',
                            dataset_mode='cityscapes',
                            batch_size=16,
                            print_freq=50,
                            save_latest_freq=10000000000,
                            save_epoch_freq=10,
                            nepochs=100,
                            nepochs_decay=100,
                            init_type='xavier',
                            teacher_ngf=64,
                            student_ngf=48)
        parser = networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super(SPADEDistiller, self).__init__(opt)

    def forward(self, on_one_gpu=False):
        if on_one_gpu:
            self.Tfake_B, self.Sfake_B = self.modules_on_one_gpu(
                self.input_semantics)
        else:
            self.Tfake_B, self.Sfake_B = self.modules(self.input_semantics)

    def evaluate_model(self, step, save_image=False):
        self.is_best = False
        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.modules_on_one_gpu.netG_student.eval()
        torch.cuda.empty_cache()
        fakes, names = [], []
        ret = {}
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader)):
            self.set_input(data_i)
            self.test()
            fakes.append(self.Sfake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10 or save_image:
                    input_im = util.tensor2label(self.input_semantics[j],
                                                 self.opt.input_nc + 2)
                    real_im = util.tensor2im(self.real_B[j])
                    Tfake_im = util.tensor2im(self.Tfake_B[j])
                    Sfake_im = util.tensor2im(self.Sfake_B[j])
                    util.save_image(input_im,
                                    os.path.join(save_dir, 'input',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(real_im,
                                    os.path.join(save_dir, 'real',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(Tfake_im,
                                    os.path.join(save_dir, 'Tfake',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(Sfake_im,
                                    os.path.join(save_dir, 'Sfake',
                                                 '%s.png' % name),
                                    create_dir=True)
                cnt += 1
        if not self.opt.no_fid:
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
            ret['metric/fid'] = fid
            ret['metric/fid-mean'] = sum(self.fids) / len(self.fids)
            ret['metric/fid-best'] = self.best_fid
        if 'cityscapes' in self.opt.dataroot and not self.opt.no_mIoU:
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

        self.modules_on_one_gpu.netG_student.train()
        torch.cuda.empty_cache()
        return ret
