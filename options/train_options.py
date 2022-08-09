from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--val_epoch_freq', type=int, default=10, help='frequency of running validation')
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=10, help='# of iter at starting learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=5, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        # learning rate related
        parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine | increase]')
        
        self.isTrain = True
        return parser
