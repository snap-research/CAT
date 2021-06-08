from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_spectral_norm

from models.modules.inception_modules import SPADEInvertedResidualChannels
from models.modules.inception_modules import _get_named_block_list
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d
from models.networks import BaseNetwork


class InceptionSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super(InceptionSPADEGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.fc_norm = SynchronizedBatchNorm2d(16 * nf, affine=True)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEInvertedResidualChannels(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEInvertedResidualChannels(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEInvertedResidualChannels(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEInvertedResidualChannels(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEInvertedResidualChannels(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEInvertedResidualChannels(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEInvertedResidualChannels(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEInvertedResidualChannels(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, mapping_layers=[]):
        seg = input

        ret_acts = {}

        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.fc_norm(x)

        if 'fc' in mapping_layers:
            ret_acts['fc'] = x

        x = self.head_0(x, seg)
        if 'head_0' in mapping_layers:
            ret_acts['head_0'] = x

        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if 'G_middle_0' in mapping_layers:
            ret_acts['G_middle_0'] = x

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        if 'G_middle_1' in mapping_layers:
            ret_acts['G_middle_1'] = x

        x = self.up(x)
        x = self.up_0(x, seg)
        if 'up_0' in mapping_layers:
            ret_acts['up_0'] = x

        x = self.up(x)
        x = self.up_1(x, seg)
        if 'up_1' in mapping_layers:
            ret_acts['up_1'] = x

        x = self.up(x)
        x = self.up_2(x, seg)
        if 'up_2' in mapping_layers:
            ret_acts['up_2'] = x

        x = self.up(x)
        x = self.up_3(x, seg)
        if 'up_3' in mapping_layers:
            ret_acts['up_3'] = x

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
            if 'up_4' in mapping_layers:
                ret_acts['up_4'] = x

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        if len(mapping_layers) == 0:
            return x
        else:
            return x, ret_acts

    def remove_spectral_norm(self):
        x = self.head_0.remove_spectral_norm()
        x = self.G_middle_0.remove_spectral_norm()
        x = self.G_middle_1.remove_spectral_norm()

        x = self.up_0.remove_spectral_norm()
        x = self.up_1.remove_spectral_norm()
        x = self.up_2.remove_spectral_norm()
        x = self.up_3.remove_spectral_norm()

        if self.opt.num_upsampling_layers == 'most':
            x = self.up_4.remove_spectral_norm()

    def get_named_block_list(self):
        return _get_named_block_list(
            self,
            spade=True,
            num_upsampling_layers=self.opt.num_upsampling_layers)
