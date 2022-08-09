import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.load_epoch + opt.n_epochs_decay - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1,padding=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.padding=padding
        self.stride=stride

    def forward(self, x):
        out = self.left(x)
        if self.padding==0:
            out += self.shortcut(x[:,:,1+self.stride:-1-self.stride,1+self.stride:-1-self.stride])
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class FullyConvBlock(nn.Module):
    """inchannel: the input channel
        outchannel: number of classes to be classified
        in_size: the size of input image"""
    def __init__(self,inchannel,outchannel,in_size,panels_layer1=512,panels_layer2=128):
        super(FullyConvBlock,self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(inchannel, panels_layer1, kernel_size=in_size,stride=1),
            nn.BatchNorm2d(panels_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(panels_layer1, panels_layer2, kernel_size=1),
            nn.BatchNorm2d(panels_layer2),
            nn.ReLU(inplace=True),
            nn.Conv2d(panels_layer2, outchannel, kernel_size=1),
            nn.Dropout2d(p=0.5),
        )
    def forward(self,x,depthfusion=None):
        out = self.fcn(x)
        return out

class InceptionBlock(nn.Module):
    """
    inchannel: the input channel
    midchannel: number of middle channel
    outchannel: the size of output channel
    """
    def __init__(self,inchannel,midchannel,outchannel):
        super(InceptionBlock,self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, midchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(inchannel, midchannel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(inchannel, midchannel, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(midchannel),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3*midchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        out = torch.cat([self.conv3(x),self.conv5(x),self.conv7(x)],dim=1)
        out = self.conv1(out)
        return out
