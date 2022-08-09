import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def depthImgShow(window_name,depth_img,show=True):
    '''convert the raw depth to show version and call imshow'''
    min_depth, max_depth = depth_img.min(), depth_img.max()
    depth_show_img = (depth_img - min_depth) / (max_depth - min_depth)    #depth image normalization, dtype:float64
    if show:
        cv2.imshow(window_name,depth_show_img)
    return depth_show_img

def write_figure(writer,np_img,color_map='npy_spectral',tag='unnamed image',figsize=(10,10),vmin=0,vmax=1):
    fig=plt.figure(figsize=figsize,dpi=80)
    if color_map!='':
        plt.imshow(np_img,cmap=plt.get_cmap(color_map,vmax-vmin),vmin=vmin,vmax=vmax)
    else:
        plt.imshow(np_img)
    plt.colorbar()
    writer.add_figure(tag=tag,figure=fig)



def combine_results(predict,ir,show_depth,color):
    ir=cv2.resize(np.repeat(ir[:,:,np.newaxis], 3, axis=2),(640,480)).astype(np.uint8)
    show_depth=(cv2.resize(np.repeat(show_depth[:,:,np.newaxis], 3, axis=2),(640,480),interpolation=cv2.INTER_NEAREST)*256).astype(np.uint8)
    color=cv2.resize(color,(640,480)).astype(np.uint8)
    predict=cv2.resize(predict,(640,480),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    up=np.append(predict,color,axis=0)
    down=np.append(show_depth,ir,axis=0)
    combine_result=np.append(up,down,axis=1)
    return combine_result

def calc_norm(depth_img):
    '''
    input:depth_img: numpy(240,320) in the form of (H,W)
                    or numpy(N,1,234,320) in the form of (H,W)
    output: norm = torch(1,3,240,320) in the form of (N,C,H,W), channels:[ori_x,ori_y,ori_z]
            angle= torch(1,240,320) in the form of (N,H,W)
    '''
    #calculate the gradient and thus norm
    kernel_x = torch.FloatTensor([[-1,0,1],[-2,0,2],[-1,0,1]]).unsqueeze(0).unsqueeze(0)/8.0
    kernel_y = torch.FloatTensor([[-1,-2,-1],[0,0,0],[1,2,1]]).unsqueeze(0).unsqueeze(0)/8.0
    if len(depth_img.shape)==2:
        depth=torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0)
    else:
        depth=torch.from_numpy(depth_img)

    pad = nn.ReplicationPad2d(1)
    threshold = nn.Threshold(0,0.2)
    depth=threshold(depth)
    dz_dx=F.conv2d(pad(depth),kernel_x)
    dz_dy=F.conv2d(pad(depth),kernel_y)
    ori_x=-(570.34/2)*dz_dx/depth
    ori_y=-(570.34/2)*dz_dy/depth
    ori_z=torch.ones(depth.shape,dtype=(torch.float))
    norm=torch.cat((ori_x,ori_y,ori_z),dim=1)

    #load the predefined incidence orientation
    inc=config.inc

    #calculate the cosine of this two vectors
    normed_inc=((inc**2).sum(dim=1))**(1/2)
    normed_norm=((norm**2).sum(dim=1))**(1/2)
    angle=(inc*norm).sum(dim=1)/(normed_norm*normed_inc)

    return angle,norm

class Channel_Selector(object):
    ''' channel select an depth preprocess
    channel 0: IR (range 0,1)
    channel 1: depth (range 0,1+) inverse_squred (range -0.5,1)
    channel 2: norm (range -1,1)
    channel 3: blue (range 0,1)   represent grey when grey_flag==True
    channel 4: green (range 0,1)
    channel 5: red (range 0,1)
    channel 6: IR_B (range 0,1) if with_BC else mask (range 0,1)
    channel 7: IR_C (range 0,1) if with_BC
    channel 8: mask (range 0,1)
    '''
    def __init__(self,channels,with_BC=True):
        ''' hint: now only select data in models
        
        Args:
            channels(str):        name of the used channels
            depth_preproc(str):  depth process methods: ['inverse_square', 'inverse','none']
            blur_kernel(int):    bluring kernel of IR image, to be deprecated 
            with_BC(bool):       weather dataset have BC channel or not
        '''
        self.grey_flag = False
        self.channels = channels
        self.with_BC = with_BC
        self.direct_map={
            'ir':0,
            'depth':1,
            'norm':2,
            'blue':3,
            'green':4,
            'red':5,
            'B':6,
            'C':7,
            'grey':6,
        }

        self.secondary_map={
            'dn':['depth','norm'],
            'color':['blue','green','red'],
            'full':['ir','depth','norm','blue','green','red'],
        }
        if self.with_BC:
            self.secondary_map['full']=['ir','depth','norm','blue','green','red','B','C']
            self.direct_map['grey']=8
        self.__channel_selection__(self.channels)

    def select(self,data):
        ''' select data
        inputs: 
            data: torch tensor with N*D*H*W
        '''
        # verify the data dimension 
        if self.with_BC:
            assert(data.shape[1]==9 or data.shape[1]==8 )
        else:
            assert(data.shape[1]==7 or data.shape[1]==6 )
        if self.grey_flag == True: #grey case
            data[:,self.direct_map['grey']]=torch.mean(data[:,3:6],dim=1)
        return data[:,self.channel_list]

    def get_channel_list(self):
        return self.channel_list
    def get_channel_dict(self):
        return self.channel_dict
    def get_channel_len(self):
        length = len(self.channel_list)
        return length

    def __channel_selection__(self,channels):
        ''' generate channel_list and channel_dict and greey_flag

        return:
            channel_dict: a map from name to channel idx in previous data, like 'ir'->0
            channel_list: the channel list to be selected
            channel_dict_new: a map from name to channel idx in selected data, like 'grey'->3
        '''
        self.channel_dict={}
        # for signel channels, use direct map
        for name,idx_list in self.direct_map.items():
            if name in channels:
                self.channel_dict[name]=idx_list
                if name == 'grey':    self.grey_flag=True   # set grey flag
        # for combine channels, use secondary map
        for name,single_list in self.secondary_map.items():
            if name in channels:
                for single in single_list:
                    self.channel_dict[single]=self.direct_map[single]
        # generate channel list
        self.channel_list=[]
        for k,v in self.channel_dict.items():
            self.channel_list.append(v)
        self.channel_list=sorted(set(self.channel_list))
        # generate channel_dict_new
        self.channel_dict_new={}
        for idx,channel in enumerate(self.channel_list):
            print(idx,channel)
            self.channel_dict_new[[k for k, v in self.channel_dict.items() if v == channel][0]]=idx
        print('channel_list: %s , channel_dict: %s,channel_dict_new: %s ' %(self.channel_list,self.channel_dict,self.channel_dict_new) )
    
    

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    channel_selector = Channel_Selector('ir_dn_color',True)