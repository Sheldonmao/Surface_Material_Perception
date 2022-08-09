'''
haven't change accordingly yet
'''
import torch,os
import torch.nn as nn

from . import networks
from .base_model import BaseModel
import itertools

class FeatureFusionNet(BaseModel):
    '''
    input x: torch(N,4,960,1280)
    if this is used as training on patches, be sure to train on patch size 32
    !! be careful this net is restricted to full imput of 4 channels
    note: in this net we change stride to padding layer
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--pretrain',  default='no', help='fix or finetune or no for pretraning')
        parser.add_argument('--ignore_weight', nargs='+', default='', help='ignore weights when loading')
        parser.add_argument('--dot_modal', default='dot_geo', help='dot modal includes')
        parser.add_argument('--dif_modal', default='dif_geo', help='dif modal includes')
        parser.add_argument('--gray_modal', default='gray', help='gray modal includes')
        parser.add_argument('--valid_modals', nargs='+', default=['dot','dif','gray'], help='what modals are valid')
        return parser

    def __init__(self, opt):
        """
        base model
        """
        super().__init__(opt)
        self.pad = 1
        self.pretrain=opt.pretrain
        self.ignore_weight = opt.ignore_weight
        self.inchannel=32
        self.valid_modals = opt.valid_modals
        
        self.dot_list=self.select_channels(opt,opt.dot_modal)
        self.dif_list=self.select_channels(opt,opt.dif_modal)
        self.gray_list=self.select_channels(opt,opt.gray_modal)

        self.netResDot = nn.Sequential(
            nn.Conv2d(len(self.dot_list), 32, kernel_size=3, stride=1, padding=self.pad, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
        ).to(self.device)

        self.netResDif = nn.Sequential(
            nn.Conv2d(len(self.dif_list), 32, kernel_size=3, stride=1, padding=self.pad, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
        ).to(self.device)

        self.netResGray = nn.Sequential(
            nn.Conv2d(len(self.gray_list), 32, kernel_size=3, stride=1, padding=self.pad, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
            nn.MaxPool2d(2,2),
            self.make_layer(networks.ResidualBlock, 32,  num_blocks=2, stride=1, padding=self.pad),
        ).to(self.device)


        self.inchannel = self.inchannel*len(self.valid_modals) # extend for two channels
        self.netFcn = self.make_layer(networks.FullyConvBlock, self.num_classes, in_size=opt.patch_size//8).to(self.device)
        
        self.model_names=['ResDot','ResDif','ResGray','Fcn']
        self.visual_names=['input_visual']

        if self.isTrain:
            # criterion
            self.criterionCls = nn.CrossEntropyLoss()
            self.loss_names=['cls']
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_cls = torch.optim.SGD(self.netFcn.parameters(), lr=opt.lr,momentum=0.8)
            if self.pretrain == 'finetune':
                self.optimizer_embed = torch.optim.SGD(itertools.chain(self.netResDot.parameters(),self.netResDif.parameters(),self.netResGray.parameters()), lr=opt.lr*0.01,momentum=0.8)
                self.optimizers=[self.optimizer_embed,self.optimizer_cls]
            elif self.pretrain == 'fix':
                self.optimizers=[self.optimizer_cls]
                self.set_requires_grad([self.netResDot,self.netResDif,self.netResGray],False)
            else:
                self.optimizer_embed = torch.optim.SGD(itertools.chain(self.netResDot.parameters(),self.netResDif.parameters(),self.netResGray.parameters()), lr=opt.lr,momentum=0.8)
                self.optimizers=[self.optimizer_embed,self.optimizer_cls]

    def set_input(self,data):
        self.x, self.y = data['data'].to(self.device),data['label'].to(self.device)
        self.dot = self.x[:,self.dot_list]
        self.dif = self.x[:,self.dif_list]
        self.gray = self.x[:,self.gray_list]
        # print('input feature shape:',self.x.shape)
        self.input_visual=[self.dot[0],self.y[0]]    # extrace the first example as visuals
        
    def forward(self,reduce=True):
        if self.pretrain == 'fix':
            self.netResDot.eval()
            self.netResDif.eval()
            self.netResGray.eval()
        self.embed_dot = self.netResDot(self.dot)
        self.embed_dif = self.netResDif(self.dif)
        self.embed_gray = self.netResGray(self.gray)
        self.embed = self.routing()
        self.out = self.netFcn(self.embed)
        if reduce:
            self.out = self.out.view(self.out.size(0), self.num_classes)
        # print('x:',self.x.shape,'embed:',self.embed.shape,'out:',self.out.shape)
        return self.out
    
    def routing(self):
        torch_list=[]
        for light in self.valid_modals:
            torch_list.append(getattr(self,'embed_%s'%light))
        return torch.cat(torch_list,1)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration
            only excuated when training
        """
        self.forward()
        # update classification networks
        if self.pretrain =='fix':
            self.optimizer_cls.zero_grad()  # set netFCN gradients to zero
            self.backward_cls()              # calcualte the gradient of clasification loss
            self.optimizer_cls.step()       # update parameters
        else:
            self.optimizer_cls.zero_grad()  # set netFCN gradients to zero
            self.optimizer_embed.zero_grad() # set netResDot netResDif netResGray gradients to zero
            self.backward_cls()              # calcualte the gradient of clasification loss
            self.optimizer_cls.step()       # update parameters
            self.optimizer_embed.step()

    def backward_cls(self):
        """Calculate classification loss"""
        self.loss_cls = self.criterionCls(self.out,self.y)
        self.loss_cls.backward()

    def get_outputs(self):
        output_dict={'out':self.out}
        return output_dict
    
    def get_target(self):
        return self.y

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str) and name not in self.ignore_weight:
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                    
                net.load_state_dict(state_dict)