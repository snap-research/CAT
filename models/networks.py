import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from models.modules.sync_batchnorm import SynchronizedBatchNorm2d

###############################################################################
# Helper Functions
###############################################################################


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance',
                   affine=True,
                   track_running_stats=True):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d,
                                       affine=affine,
                                       track_running_stats=track_running_stats)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=affine,
                                       track_running_stats=track_running_stats)
    elif norm_type == 'syncbatch':
        if not track_running_stats:
            import warnings
            warnings.warn(
                'track_running_stats=False is not supported by the SynchronizedBatchNorm.'
            )
        norm_layer = functools.partial(SynchronizedBatchNorm2d,
                                       affine=affine,
                                       track_running_stats=track_running_stats)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 - opt.nepochs) / float(opt.nepochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.niter,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.weight is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)

    return net


def define_G(input_nc,
             output_nc,
             ngf,
             netG,
             norm='batch',
             dropout_rate=0,
             init_type='normal',
             init_gain=0.02,
             gpu_ids=[],
             opt=None):
    norm_layer = get_norm_layer(norm_type=norm,
                                affine=getattr(opt, 'norm_affine', False),
                                track_running_stats=getattr(
                                    opt, 'norm_track_running_stats', False))
    if netG == 'inception_9blocks':
        from .modules.inception_architecture.inception_generator import InceptionGenerator
        net = InceptionGenerator(
            input_nc,
            output_nc,
            ngf=ngf,
            channels=opt.channels,
            channels_reduction_factor=opt.channels_reduction_factor,
            kernel_sizes=opt.kernel_sizes,
            norm_layer=norm_layer,
            norm_momentum=opt.norm_momentum,
            norm_epsilon=opt.norm_epsilon,
            dropout_rate=dropout_rate,
            active_fn=opt.active_fn,
            n_blocks=9)
    elif netG == 'inception_spade':
        from .modules.inception_architecture.inception_spade_generator import InceptionSPADEGenerator
        net = InceptionSPADEGenerator(opt)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc,
             ndf,
             netD,
             n_layers_D=3,
             norm='batch',
             init_type='normal',
             init_gain=0.02,
             gpu_ids=[],
             opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm,
                                affine=getattr(opt, 'norm_affine_D', False),
                                track_running_stats=getattr(
                                    opt, 'norm_track_running_stats', False))
    if netD == 'multi_scale':
        from models.modules.discriminators import MultiscaleDiscriminator
        net = MultiscaleDiscriminator(opt)
    elif netD == 'n_layers':
        from models.modules.discriminators import NLayerDiscriminator
        net = NLayerDiscriminator(input_nc,
                                  ndf,
                                  n_layers_D,
                                  norm_layer=norm_layer,
                                  active_fn=opt.active_fn_D)
    elif netD == 'pixel':
        from models.modules.discriminators import PixelDiscriminator
        net = PixelDiscriminator(input_nc,
                                 ndf,
                                 norm_layer=norm_layer,
                                 active_fn=opt.active_fn_D)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def get_netG_cls(netG):
    if netG == 'inception_9blocks':
        from .modules.inception_architecture.inception_generator import InceptionGenerator
        return InceptionGenerator
    elif netG == 'inception_spade':
        from .modules.inception_architecture.inception_spade_generator import InceptionSPADEGenerator
        return InceptionSPADEGenerator
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)


def get_netD_cls(netD):
    if netD == 'n_layers':
        from .modules.discriminators import NLayerDiscriminator
        return NLayerDiscriminator
    elif netD == 'pixel':
        from .modules.discriminators import PixelDiscriminator
        return PixelDiscriminator
    elif netD == 'multi_scale':
        from .modules.discriminators import MultiscaleDiscriminator
        return MultiscaleDiscriminator
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % netD)


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()
    if hasattr(opt, 'netG'):
        netG_cls = get_netG_cls(opt.netG)
        parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = get_netD_cls(opt.netD)
        parser = netD_cls.modify_commandline_options(parser, is_train)

    return parser
