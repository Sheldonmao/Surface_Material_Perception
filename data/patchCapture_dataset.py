from data.base_dataset import BaseDataset
import pandas as pd
import numpy as np
import cv2
from data import data_transform as transforms
from options import config
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from cv2.ximgproc import guidedFilter

# calibration parameters
calib_parameters = np.array([
    [9.50120074e-01, -8.69293641e-03,  2.20524367e+01, -2.12196076e+01],
    [1.14511592e-02,  9.41119908e-01,  6.30609363e+01, -4.57410650e+01],
])



#incidence orientation
ori_z=np.ones((240,320),dtype=(np.float32))
ori_x=np.ones((240,320),dtype=(np.float32))
ori_y=np.ones((240,320),dtype=(np.float32))
for i in range(240):
    for j in range(320):
        ori_x[i,j]=0.5543*(j-159.5)/159.5
        ori_y[i,j]=0.41420*(i-119.5)/119.5
inc = np.stack([ori_x,ori_y,ori_z],axis=-1)
inc = cv2.resize(inc,(640,480)).transpose(2,0,1)

class Registerer(object):
    '''class holding the camera setup and performing the actual depth registration'''

    def __init__(self, params):
        self.params = params
        self.pixel_grid = None

    def register(self, rgba_image, depth_image):
        '''this is where the magic happens

        inputs:
            rgba_np (np.ndarray)    -- raw RGBA image, shape=(960,1280,4), dtype=np.unit16
            depth_img (np.ndarray)  -- raw depth image with shape=(240,320), dtype=np.uint16
        outputs:
            registered_rgba_image (np.ndarray) -- registered RGBA image, shape=(960,1280,4), dtype=np.float32, range=[0,1]
        '''
        params = self.params
        HDepth,WDepth = depth_image.shape
        HRGBA,WRGBA,CRGBA = rgba_image.shape
        
        # generate the huge coordinate matrix only once
        if self.pixel_grid is None:
            # this is basically a 2d `range`
            self.pixel_grid = np.stack((
                np.array([np.arange(depth_image.shape[0]) for _ in range(depth_image.shape[1])]).T,
                np.array([np.arange(depth_image.shape[1]) for _ in range(depth_image.shape[0])])
                ), axis=2)


        # compute the exact usable (mapped to) size of the registered depth image wrt. the FOVs of the cameras to avoid
        # gaps in the registered depth image (columns/rows without values). later, the registered depth image gets
        # scaled to match the size of the rgb image.
        registered_rgba_image = np.zeros((HDepth,WDepth,CRGBA),dtype='uint8')

        # only consider pixels where actual depth values exist
        valid_depths = depth_image > 0
        valid_pixels = self.pixel_grid[valid_depths]
        # might seem a little nasty, but computes the registered depth numpy-efficiently
        # apply scaling and extrinsics to depth values
        zs = depth_image[valid_depths]
        # apply depth cam intrinsics, extrinsics, rgb cam intrinsics, scale down to (w, h)
        xs = params[0,0]*valid_pixels[:,1]*WRGBA/WDepth+params[0,1]*valid_pixels[:,0]*HRGBA/HDepth+params[0,2]+params[0,3]*1000/zs
        ys = params[1,0]*valid_pixels[:,1]*WRGBA/WDepth+params[1,1]*valid_pixels[:,0]*HRGBA/HDepth+params[1,2]+params[1,3]*1000/zs

        xs = (xs).astype(int)
        ys = (ys).astype(int)
        # # discard depth values unseen by rgb camera
        valid_positions = np.logical_and(np.logical_and(np.logical_and(ys >= 0, ys < rgba_image.shape[0]), xs >= 0), xs < rgba_image.shape[1])
        # registered_depth_image[ys[valid_positions], xs[valid_positions]] = zs[valid_positions]
        registered_rgba_image[valid_pixels[valid_positions,0], valid_pixels[valid_positions,1]] = rgba_image[ys[valid_positions], xs[valid_positions]]

        # scale up without smoothing to match rgb image
        registered_rgba_image = cv2.resize(registered_rgba_image, (rgba_image.shape[1], rgba_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        registered_rgba_image = np.clip(registered_rgba_image/255,0,1) # limit to range [0,1]

        return registered_rgba_image


def ir_process(ir_img,rgb_np):
    ''' 
    inputs: 
        ir_img (np.ndarray)     --raw IR image shape of (960,1280), dtype=np.uint16
        rgba_np (np.ndarray)    --registered RGBA image, shape of (960,1280,4), dtype=np.unit16
    outputs: 
        ir_np (np.ndarray)  --shape=(960,1280,2), dtype=np.float32, channels for dot/dif modals respectively, range=[0,1]
    '''
    dot = ir_img.astype(np.float32)

    dif=cv2.medianBlur(dot,5)
    dif = cv2.resize(dif,(320,240)).astype(np.float32)
    gray_img = cv2.resize(rgb_np[:,:,:3],(320,240)).mean(axis=-1).astype(np.float32)/255
    dif = guidedFilter(guide=gray_img, src=dif, radius=2, eps=0.0001, dDepth=-1)*4 # tends to normalize to (0,1)
    dif = cv2.resize(dif,(1280,960))
    
    data = np.stack([dot,dif],axis=-1)
    data = np.clip(data/65535,0,1)
    return data

def geo_process(depth_img):
    '''
    input:  
        depth_img (np.ndarray)  -- raw depth image with shape of (240,320), dtype=np.uint16
    output: 
        geo_np (np.ndarray) -- shape of (240,320,3), channels for depth/angle/radius respectively, dtype=np.float32
                                depth in range[0,+inf) in meter
                                angle in range[-1,1] in cosine value
                                radius in range[0,+inf) in meter
    '''
    #calculate the gradient and thus norm
    # depth_img = cv2.GaussianBlur(depth_img,(5,5),0)
    depth_img = cv2.bilateralFilter(depth_img.astype(np.float32),5,50,5)
    depth_img = cv2.resize(depth_img,(640,480)).astype(np.float32)/1000 # measure depth in meterf
    
    # develop: test if normal vecotr is calculated correctly 
    # depth_img = np.arange(0.68,1.32,0.001)
    # depth_img = np.repeat(depth_img[np.newaxis,:],480,axis=0).astype(np.float32)
    # depth_img = np.ones_like(depth_img)

    kernel_x = torch.FloatTensor([[-1,0,1],[-2,0,2],[-1,0,1]]).unsqueeze(0).unsqueeze(0)/8.0
    kernel_y = torch.FloatTensor([[-1,-2,-1],[0,0,0],[1,2,1]]).unsqueeze(0).unsqueeze(0)/8.0
    depth=torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0)

    pad = nn.ReplicationPad2d(1)
    threshold = nn.Threshold(0,0.2)
    depth=threshold(depth)
    dz_dx=F.conv2d(pad(depth),kernel_x)
    dz_dy=F.conv2d(pad(depth),kernel_y)
    ori_x=-(1156.6)*dz_dx/depth
    ori_y=-(1156.6)*dz_dy/depth
    ori_z=torch.ones(depth.shape,dtype=(torch.float))
    norm=torch.cat((ori_x,ori_y,ori_z),dim=1).squeeze().numpy()

    #calculate the cosine of this two vectors
    normed_inc=((inc**2).sum(axis=0))**(1/2)
    normed_norm=((norm**2).sum(axis=0))**(1/2)
    angle=(inc*norm).sum(axis=0)/(normed_norm*normed_inc)

    radius = normed_inc*depth_img
    angle = cv2.GaussianBlur(angle,(3,3),0)
    
    geo_np = np.stack([depth_img,angle,radius],axis=-1)
    geo_np = cv2.resize(geo_np,(1280,960))

    return geo_np

class PatchCaptureDataset(BaseDataset):
    """dataset for patch image generation captured in real-world.
    combining channels in a fixed way: (dot,dif,real,R,G,B,depth_processed,normal_processed,radius_processed) altogether 9 channels

    Attributes:
        file(string):      Path to the csv file with annotations.
        transform(callable):   Optional transform to be applied on a sample
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        input:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        output:
            the modified parser.
        """
        parser.add_argument('--select_plane',  default='all', help='select plane type from: [normal,deformed,all]')
        parser.add_argument('--select_materials',  default='40', help='select material list: 40/white_10/all')
        parser.add_argument('--channel_dict',default={'dot':0,'dif':1,'gray':2,'R':3,'G':4,'B':5,'A':6,'depth':7,'angle':8,'radius':9},help='channel name:idx dictionary')
        return parser

    def __init__(self,opt, file):
        """ Load csv file, root_dir
        """
        super().__init__(opt)
        self.data_frame = pd.read_csv(file)
        self.material_list = config.material_dict[opt.select_materials]  
        self.data_frame = self.df_select(materials=self.material_list,plane=opt.select_plane)
        self.root_dir=file.rsplit('/',1)[0]+'/'

        ## set transform
        self.transform = self.get_transform(opt)

        ## set register
        self.registerer = Registerer(params=calib_parameters)
        self.device=opt.device

        ## preload data related
        self.data_list=[]
        self.expand = opt.data_expand   
        if opt.preload:     
            for idx in tqdm(range(len(self.data_frame))):  # iterate over all instances
                data,label_idx = self.read_csv_item(idx)
                self.data_list.append([data,label_idx])
            print('preload finished')
        
        
    def df_select(self,materials,plane):
        ''' select by material_list and plane condition'''
        data_frame = self.data_frame.copy()
        if plane == 'all':
            pass
        else:
            data_frame = data_frame[data_frame['plane']==plane]
        
        data_frame = data_frame[data_frame['material'].isin(materials)]

        return data_frame

    def get_transform(self,opt):
        ## special get transform for blender rendered images formats
        transform_list = []
        # random crop 
        transform_list+= [transforms.transformRandomCropWithCheck(int(opt.patch_size),check_fun=lambda x: x[:,:,6].min()>0.5)]
        # depth preprocess
        transform_list+= [transforms.transformGeo(opt.depth_preproc,depth_idx=7)]
        if opt.isTrain:
            transform_list+= [transforms.transfromFlip(p=0.5)]    # random flip
        # convert to tensor  
        transform_list+= [transforms.transformToTensor()]
        
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data_frame)*self.expand

    def read_csv_item(self,idx):
        ''' read data from single csv item 

        input:  
            idx (int)   -- index of item in csv file, indicating the sample
        output: 
            data (np.ndarray) -- shape=(960,1280,10),dtype=np.float32, channels for dot/dif/gray/R/G/B/A/depth/angle/radius respectively
        '''
        ## data related
        material,distance,orientation,plane = self.data_frame.iloc[idx]
        origin_dir = pjoin(self.root_dir,material) 
                
        ir_img = cv2.imread(pjoin(origin_dir,"{}{}_{}_ir.png".format(distance,orientation,plane)),cv2.IMREAD_UNCHANGED)[:960,:1280]
        rgba_img = cv2.imread(pjoin(origin_dir,"{}{}_{}_mask.png".format(distance,orientation,plane)),cv2.IMREAD_UNCHANGED)[:960,:1280]
        geo_img = cv2.imread(pjoin(origin_dir,"{}{}_{}_depth.png".format(distance,orientation,plane)),cv2.IMREAD_UNCHANGED)
        if rgba_img.shape == (960,1280,3):  # if no alpha channel, it means all pixels are valid
            rgba_img = np.concatenate([rgba_img,np.ones((960,1280,1),dtype=np.uint8)*255],axis=-1)

        rgba_np = self.registerer.register(rgba_img,geo_img)
        ir_np = ir_process(ir_img,rgba_np)
        geo_np = geo_process(geo_img)
        gray_np = np.mean(rgba_np[:,:,:3],axis=-1)[:,:,np.newaxis]
        
        data = np.concatenate([ir_np,gray_np,rgba_np,geo_np],axis=-1)

        ## develop: visualize intermediate products
        # if not os.path.isdir(pjoin(self.root_dir,'processed',material)):
        #     os.mkdir(pjoin(self.root_dir,'processed',material))
        # tifffile.imwrite(pjoin(self.root_dir,'processed',material,"{}{}_{}_normal.tif".format(distance,orientation,plane)),geo_np[:,:,1])
        # tifffile.imwrite(pjoin(self.root_dir,'processed',material,"{}{}_{}_depth.tif".format(distance,orientation,plane)),geo_np[:,:,0])
        # tifffile.imwrite(pjoin(self.root_dir,'processed',material,"{}{}_{}_dif.tif".format(distance,orientation,plane)),ir_np[:,:,1])

        # plt.imshow(rgba_np,alpha=1)
        # plt.imshow(geo_np[:,:,1],alpha=0.3)
        # # plt.show()
        # plt.savefig(pjoin(self.root_dir,'processed',material,"{}{}_{}_align.tif".format(distance,orientation,plane)))
        # plt.cla()

        ## label related
        label_idx = self.material_list.index(material)
        return data, label_idx

    def __getitem__(self, idx):
        if len(self.data_list) == 0:
            data, label_idx = self.read_csv_item(idx//self.expand)
        else:
            data,label_idx = self.data_list[idx//self.expand]
        data = self.transform(data)

        # formulate a data patch
        sample = {'data':data, 'label': label_idx}
        return sample

if __name__=='__main__':
    pass
    