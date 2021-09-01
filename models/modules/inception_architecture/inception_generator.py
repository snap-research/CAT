import functools

from torch import nn

from models.modules.inception_modules import InvertedResidualChannels
from models.modules.inception_modules import get_active_fn
from models.modules.inception_modules import _get_named_block_list
from models.networks import BaseNetwork


class InceptionGenerator(BaseNetwork):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 channels,
                 channels_reduction_factor,
                 kernel_sizes,
                 padding_type='reflect',
                 norm_layer=nn.InstanceNorm2d,
                 norm_momentum=0.1,
                 norm_epsilon=1e-5,
                 dropout_rate=0,
                 active_fn='nn.ReLU',
                 n_blocks=9):
        assert (n_blocks >= 0)
        assert len(kernel_sizes) == len(
            set(kernel_sizes)), 'no duplicate in kernel sizes is allowed.'
        super(InceptionGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        norm_kwargs = {'momentum': norm_momentum, 'eps': norm_epsilon}
        active_fn = get_active_fn(active_fn)

        down_sampling = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            down_sampling += [
                nn.Conv2d(ngf * mult,
                          ngf * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2**n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        features = []
        for i in range(n_blocks1):
            features += [
                InvertedResidualChannels(
                    ngf * mult,
                    res_channels=channels,
                    dw_channels=channels,
                    channels_reduction_factor=channels_reduction_factor,
                    res_kernel_sizes=kernel_sizes,
                    dw_kernel_sizes=kernel_sizes,
                    padding_type=padding_type,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    dropout_rate=dropout_rate,
                    active_fn=active_fn)
            ]

        for i in range(n_blocks2):
            features += [
                InvertedResidualChannels(
                    ngf * mult,
                    res_channels=channels,
                    dw_channels=channels,
                    channels_reduction_factor=channels_reduction_factor,
                    res_kernel_sizes=kernel_sizes,
                    dw_kernel_sizes=kernel_sizes,
                    padding_type=padding_type,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    dropout_rate=dropout_rate,
                    active_fn=active_fn)
            ]

        for i in range(n_blocks3):
            features += [
                InvertedResidualChannels(
                    ngf * mult,
                    res_channels=channels,
                    dw_channels=channels,
                    channels_reduction_factor=channels_reduction_factor,
                    res_kernel_sizes=kernel_sizes,
                    dw_kernel_sizes=kernel_sizes,
                    padding_type=padding_type,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                    norm_kwargs=norm_kwargs,
                    dropout_rate=dropout_rate,
                    active_fn=active_fn)
            ]

        up_sampling = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            up_sampling += [
                nn.ConvTranspose2d(ngf * mult,
                                   int(ngf * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        up_sampling += [nn.ReflectionPad2d(3)]
        up_sampling += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up_sampling += [nn.Tanh()]
        self.down_sampling = nn.Sequential(*down_sampling)
        self.features = nn.Sequential(*features)
        self.up_sampling = nn.Sequential(*up_sampling)

    def forward(self, input):
        """Standard forward"""
        res = self.down_sampling(input)
        res = self.features(res)
        res = self.up_sampling(res)
        return res

    def get_named_block_list(self):
        return _get_named_block_list(self)
