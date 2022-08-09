from data.base_dataset import BaseDataset
import pandas as pd
import numpy as np
import cv2
import os
from data import data_transform as transforms

class PatchBlenderDataset(BaseDataset):
    """dataset for patch image generation created by blender simulation.
    combining channels in a fixed way: (dot,dif,real,depth_processed,normal_processed,radius_processed) altogether 6 channels

    Attributes:
        data_frame(pandas.DataFrame):   Data frame for indexing the file locations.
        transform(callable):   Optional transform to be applied on a sample.
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--select_plane',  default='all', help='select plane type from: [smooth,bump,all]')
        parser.add_argument('--channel_dict',default={'dot':0,'dif':1,'real':2,'depth':3,'angle':4,'radius':5,'gray':1},help='channel name:idx dictionary')
        parser.add_argument('--num_classes',  default=25, help='num of classes')
        return parser

    def __init__(self,opt, file):
        ## load csv file, _root_dir and samples_per_class
        super().__init__(opt)
        self.data_frame = pd.read_csv(file)
        self.data_frame = self.df_select(opt.select_plane)
        self.__root_dir=file.rsplit('/',1)[0]+'/'

        ## set transform
        self.transform = self.get_transform(opt)

        ## preload data related
        self.__data_list=[]
        self.expand = opt.data_expand   
        if opt.preload:     
            for idx in range(len(self.data_frame)):  # iterate over all instances
                data,label_idx = self.read_csv_item(idx)
                self.__data_list.append([data,label_idx])
            print('preload finished')
        self.device=opt.device
        


    def df_select(self,select_plane):
        data_frame = self.data_frame.copy()
        if select_plane == 'all':
            pass
        else:
            data_frame = data_frame[data_frame['plane']==select_plane]
        return data_frame

    def get_transform(self,opt):
        ## special get transform for blender rendered images formats
        transform_list = []
        # random crop 
        transform_list+= [transforms.transformRandomCropWithCheck(int(opt.patch_size),check_fun=lambda x:  x[:,:,3].max()<50)] #,opt.no_pad)]
        # depth preprocess
        transform_list+= [transforms.transformGeo(opt.depth_preproc,depth_idx=3)]
        if opt.isTrain:
            transform_list+= [transforms.transfromFlip(p=0.5)]    # random flip
        # convert to tensor  
        transform_list+= [transforms.transformToTensor()]
        
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data_frame)*self.expand

    def read_csv_item(self,idx):
        ir_file,geo_file,albedo,ss,plane = self.data_frame.iloc[idx]
        dst_dir = os.path.join(self.__root_dir,plane,"albedo%d__ss%d"%(albedo,ss)) 
        label_idx = albedo*5+ss
        ir_np = np.load(os.path.join(dst_dir,ir_file))
        ir_np[ir_np>1] = 1    ## threshold for value satuation
        geo_np = np.load(os.path.join(dst_dir,geo_file))
        geo_np = cv2.resize(geo_np,(1280,1024))
        data = np.concatenate([ir_np,geo_np],axis=-1)
        # data = {'ir':ir_np,'geo':geo_np}
        return data, label_idx

    def __getitem__(self, idx):
        if len(self.__data_list) == 0:
            data, label_idx = self.read_csv_item(idx//self.expand)
        else:
            data,label_idx = self.__data_list[idx//self.expand]
        # print(data.shape)
        data = self.transform(data)

        # formulate a data patch
        sample = {'data':data, 'label': label_idx}
        return sample

if __name__=='__main__':
    pass