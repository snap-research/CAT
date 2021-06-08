import ntpath
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import create_eval_dataloader
from metric import get_fid, get_mIoU
from metric.inception import InceptionV3
from metric.mIoU_score import DRNSeg
from utils import util
from . import networks
from .base_model import BaseModel
from .modules.loss import GANLoss


class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert is_train
        parser = super(Pix2PixModel, Pix2PixModel).modify_commandline_options(
            parser, is_train)
        parser.add_argument('--restore_G_path',
                            type=str,
                            default=None,
                            help='the path to restore the generator')
        parser.add_argument('--restore_D_path',
                            type=str,
                            default=None,
                            help='the path to restore the discriminator')
        parser.add_argument('--recon_loss_type',
                            type=str,
                            default='l1',
                            choices=['l1', 'l2', 'smooth_l1'],
                            help='the type of the reconstruction loss')
        parser.add_argument('--lambda_recon',
                            type=float,
                            default=100,
                            help='weight for reconstruction loss')
        parser.add_argument('--lambda_gan',
                            type=float,
                            default=1,
                            help='weight for gan loss')
        parser.add_argument(
            '--real_stat_path',
            type=str,
            required=True,
            help=
            'the path to load the groud-truth images information to compute FID.'
        )
        parser.add_argument('--lambda_comp_cost',
                            type=float,
                            default=0,
                            help='weight for comp cost loss')
        parser.add_argument('--comp_cost',
                            type=str,
                            default='l1',
                            choices=['l1'],
                            help='the type of the comp cost loss')
        parser.add_argument('--l1_renorm',
                            action='store_true',
                            help='renorm for l1')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert opt.isTrain
        BaseModel.__init__(self, opt)
        self.loss_names = [
            'G_gan', 'G_recon', 'D_real', 'D_fake', 'G_comp_cost'
        ]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D']
        self.netG = networks.define_G(opt.input_nc,
                                      opt.output_nc,
                                      opt.ngf,
                                      opt.netG,
                                      opt.norm,
                                      opt.dropout_rate,
                                      opt.init_type,
                                      opt.init_gain,
                                      self.gpu_ids,
                                      opt=opt)

        self.netD = networks.define_D(opt.input_nc + opt.output_nc,
                                      opt.ndf,
                                      opt.netD,
                                      opt.n_layers_D,
                                      opt.norm,
                                      opt.init_type,
                                      opt.init_gain,
                                      self.gpu_ids,
                                      opt=opt)

        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        if opt.recon_loss_type == 'l1':
            self.criterionRecon = torch.nn.L1Loss()
        elif opt.recon_loss_type == 'l2':
            self.criterionRecon = torch.nn.MSELoss()
        elif opt.recon_loss_type == 'smooth_l1':
            self.criterionRecon = torch.nn.SmoothL1Loss()
        else:
            raise NotImplementedError(
                'Unknown reconstruction loss type [%s]!' % opt.loss_type)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(self.device)
        self.inception_model.eval()

        if 'cityscapes' in opt.dataroot:
            self.drn_model = DRNSeg('drn_d_105', 19, pretrained=False)
            util.load_network(self.drn_model, opt.drn_path, verbose=False)
            if len(opt.gpu_ids) > 0:
                self.drn_model.to(self.device)
                self.drn_model = nn.DataParallel(self.drn_model, opt.gpu_ids)
            self.drn_model.eval()

        self.best_fid = 1e9
        self.best_mIoU = -1e9
        self.fids, self.mIoUs = [], []
        self.is_best = False
        self.Tacts, self.Sacts = {}, {}
        self.npz = np.load(opt.real_stat_path)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).detach()
        real_AB = torch.cat((self.real_A, self.real_B), 1).detach()
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake,
                                             False,
                                             for_discriminator=True)

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real,
                                             True,
                                             for_discriminator=True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_gan = self.criterionGAN(
            pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        self.loss_G_recon = self.criterionRecon(
            self.fake_B, self.real_B) * self.opt.lambda_recon

        if getattr(self.opt, 'lambda_comp_cost', 0) > 0:
            p = int(self.opt.comp_cost[-1])
            self.loss_G_comp_cost = getattr(
                getattr(self.netG, 'module', self.netG), 'get_comp_cost',
                lambda p, renorm: 0)(p=p, renorm=self.opt.l1_renorm)
            self.loss_G_comp_cost *= self.opt.lambda_comp_cost

        self.loss_G = self.loss_G_gan + self.loss_G_recon
        if getattr(self.opt, 'lambda_comp_cost', 0) > 0:
            self.loss_G += self.loss_G_comp_cost
        self.loss_G.backward()

    def optimize_parameters(self, steps):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def evaluate_model(self, step, save_image=False):
        self.is_best = False

        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG.eval()

        fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader)):
            self.set_input(data_i)
            self.test()
            fakes.append(self.fake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10 or save_image:
                    input_im = util.tensor2im(self.real_A[j])
                    real_im = util.tensor2im(self.real_B[j])
                    fake_im = util.tensor2im(self.fake_B[j])
                    util.save_image(input_im,
                                    os.path.join(save_dir, 'input',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(real_im,
                                    os.path.join(save_dir, 'real',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(fake_im,
                                    os.path.join(save_dir, 'fake',
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
        if 'cityscapes' in self.opt.dataroot:
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

        self.netG.train()
        return ret
