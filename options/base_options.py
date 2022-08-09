import argparse
import os
import torch
import json
import models
import data
from utils import utils
from options import config

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--outf', default=os.path.join(config.MODEL_DIR,'debug'), help='folder to output images and model checkpoints, i.e.../models/0609_nonsense/0805/')
        parser.add_argument('--name', default='debug_net', help="name of the net,just a flag")

        # patch dataset related
        parser.add_argument('--train_csv',default=config.DATASET_DIR+'/final_set/train/processed/np_files/data.csv')
        parser.add_argument('--val_csv',default=config.DATASET_DIR+'/final_set/train/processed/np_files/normal_data.csv')
        parser.add_argument('--test_csv',default=config.DATASET_DIR+'/final_set/test/processed/np_files/data.csv')
        # parser.add_argument('--material_type',default='ss')
        
        ## data transform
        parser.add_argument('--depth_preproc',default='inverse',help='the preprocessing method for depth channel')
        parser.add_argument('--patch_size',default=48,help='input size for data, to determine the best patch size',type=int)
        

        ## data loader related
        parser.add_argument('--batch_size',default=64,help='the batch_size of training',type=int)
        parser.add_argument('--num_workers',default=4,help='num of workers to load data',type=int)
        parser.add_argument('--preload',default=True,help='if preload the data into memory',type=str2bool)
        parser.add_argument('--data_expand',default=1,type=int,help='the repeat number of dataset ')
        ## net related
        parser.add_argument('--model',default='resnet')
        parser.add_argument('--load_dir', default='', help="name of the dir to be loaded, if continue train")
        
        ## train related
        parser.add_argument('--dataset_mode',default='patchBlender',help='determine the mode of training')
               
        ## gpu
        parser.add_argument('--gpu_id',default=0,type=int,help='the gpu to be used')

        ## additional options
        parser.add_argument('--load_epoch', type=int, default='0', help='iteration to be loaded')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)


        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt,save = True):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        file_name = os.path.join(opt.save_dir, 'opt.json')
        with open(file_name, 'wt') as opt_file:
            json.dump(sorted(vars(opt).items()),opt_file,ensure_ascii=False)
            opt_file.write('\n')

    def parse(self,opt=None):
        """Parse our options, and set up gpu device."""
        if not self.initialized:  # check if it has been initialized
            opt = self.gather_options()
        else: 
            opt = opt
        opt.isTrain = self.isTrain   # train or test

        ### set save_dir
        if opt.isTrain:
            opt.save_dir=os.path.join(opt.outf, opt.name)
        else: 
            opt.save_dir=opt.outf
        utils.mkdirs(opt.save_dir)

        self.print_options(opt)
        
        # set gpu id
        if opt.gpu_id == -1:
            opt.device=torch.device('cpu')
            #print('cpu is used')
        else:
            opt.device = torch.device(opt.gpu_id)

        # num of classes
        if opt.dataset_mode=='patchCapture':
            opt.num_classes = len(config.material_dict[opt.select_materials])

        self.opt = opt
        return self.opt

