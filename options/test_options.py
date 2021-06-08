from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self, isTrain=False):
        super(TestOptions, self).__init__()
        self.isTrain = isTrain

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir',
                            type=str,
                            default=None,
                            required=True,
                            help='saves results here.')
        parser.add_argument('--need_profile', action='store_true')
        parser.add_argument('--num_test',
                            type=int,
                            default=float('inf'),
                            help='how many test images to run')
        parser.add_argument('--model',
                            type=str,
                            default='test',
                            help='which model do you want test')
        parser.add_argument('--restore_G_path',
                            type=str,
                            required=True,
                            help='the path to restore the generator')
        parser.add_argument('--netG',
                            type=str,
                            default='inception_9blocks',
                            choices=['inception_9blocks | inception_spade'],
                            help='specify generator architecture')
        parser.add_argument(
            '--ngf',
            type=int,
            default=64,
            help='the base number of filters of the student generator')
        parser.add_argument('--dropout_rate',
                            type=float,
                            default=0,
                            help='the dropout rate of the generator')
        parser.add_argument(
            '--channels',
            nargs='*',
            type=int,
            default=None,
            help='the list of channel numbers for different kernel sizes')
        parser.add_argument(
            '--channels_reduction_factor',
            type=int,
            default=1,
            help=
            'the reduction factor for channel numbers for different kernel sizes'
        )
        parser.add_argument('--kernel_sizes',
                            nargs='+',
                            type=int,
                            default=[3, 5, 7],
                            help='the list of kernel sizes')
        parser.add_argument('--norm_affine',
                            action='store_true',
                            help='set affine for the norm layer')
        parser.add_argument(
            '--norm_affine_D',
            action='store_true',
            help='set affine for the norm layer in discriminator')
        parser.add_argument('--norm_momentum',
                            type=float,
                            default=0.1,
                            help='the momentum for the norm layer')
        parser.add_argument('--norm_epsilon',
                            type=float,
                            default=1e-5,
                            help='the epsilon for the norm layer')
        parser.add_argument('--active_fn',
                            type=str,
                            default='nn.ReLU',
                            help='the activation function')
        parser.add_argument('--moving_average_decay',
                            type=float,
                            default=0.0,
                            help='the moving average decay for ema')
        parser.add_argument(
            '--moving_average_decay_adjust',
            action='store_true',
            help=
            'adjust the moving average decay for ema or not, default is False')
        parser.add_argument(
            '--moving_average_decay_base_batch',
            type=int,
            default=32,
            help='batch size of the moving average decay for ema')
        parser.add_argument('--no_fid', action='store_true')
        parser.add_argument(
            '--real_stat_path',
            type=str,
            required=None,
            help=
            'the path to load the groud-truth images information to compute FID.'
        )
        parser.add_argument('--no_mIoU', action='store_true')
        parser.add_argument(
            '--times',
            type=int,
            default=100,
            help='times of forwarding the data to test the latency')
        parser.set_defaults(phase='val',
                            serial_batches=True,
                            no_flip=True,
                            load_size=parser.get_default('crop_size'),
                            batch_size=1)
        return parser
