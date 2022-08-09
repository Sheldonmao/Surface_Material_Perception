import numpy as np
from skimage import transform
from torchvision import transforms

class transformRandomCrop(object):
    '''Crop randomly the images in a sample.'''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]
        new_h, new_w = self.output_size
        self.top = np.random.randint(0, h - new_h+1)
        self.left = np.random.randint(0, w - new_w+1)
        data = data[self.top: self.top + new_h, self.left: self.left + new_w]
        return data

    def get_location(self):
        return self.top,self.left


class transformRandomCropWithCheck(object):
    '''Crop randomly the image in a sample.'''
    def __init__(self, output_size,check_fun=lambda x:  x[:,:,3].max()<50):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.check_fun =check_fun

    def __call__(self, data):
        valid = False
        h, w = data.shape[:2]
        new_h, new_w = self.output_size
        while valid == False:
            self.top = np.random.randint(0, h - new_h+1)
            self.left = np.random.randint(0, w - new_w+1)
            new_data = data[self.top: self.top + new_h, self.left: self.left + new_w]
            valid = self.check_fun(new_data)
        return new_data

    def get_location(self):
        return self.top,self.left

class transformRandomCropWithMask(object):
    '''Crop randomly the image in a sample.'''
    def __init__(self, output_size,zero_padding=False,preload=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        valid = False
        h, w = data.shape[:2]
        new_h, new_w = self.output_size
        while valid == False:
            self.top = np.random.randint(0, h - new_h+1)
            self.left = np.random.randint(0, w - new_w+1)
            new_data = data[self.top: self.top + new_h, self.left: self.left + new_w]
            valid = self.check(new_data[:,:,3])
        return new_data
    
    def check(self,mask_img):
        if np.average(mask_img)<=0.2:
            return False
        else:
            return True

    def get_location(self):
        return self.top,self.left

class transfromFlip(object):
    ## data should be of the form H*W*C
    def __init__(self,p=0.5,vertical=True,horizontal=True):
        self.vertical = vertical
        self.horizontal = horizontal
        self.p=p

    def __call__(self,data):
        if self.vertical and np.random.binomial(1,self.p):
            data = data[::-1,:,:]
        if self.horizontal and np.random.binomial(1,self.p):
            data = data[:,::-1,:]
        return data

class transfromDropBehind(object):
    ## drop everything behind given idx
    def __init__(self,idx=1):
        self.idx=idx
    def __call__(self,data):
        ## data should be of the form H*W*C
        return data[:,:,:self.idx]

class transformToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1)).astype(np.float32)
        return data

class transformRandomRotate(object):
    '''Crop randomly the image in a sample.'''
    def __init__(self, angles):
        assert isinstance(angles, (int, tuple))
        if isinstance(angles, int):
            self.angles = (-angles, angles+1)
        else:
            assert len(angles) == 2
            self.angles = angles

    def __call__(self, data):
        angle=np.random.randint(self.angles[0],self.angles[1])
        data=transform.rotate(data,angle)
        return data

class transformGeo(object):
    ''' transform depth in another representation

    parameters:
        method: depth preprocessing method, range from ['inverse_squre','inverse','none']
    '''
    def __init__(self, method = 'inverse',depth_idx=1):
        self.method=method
        self.depth_idx = depth_idx
        self.rad_idx = depth_idx+2
        
    def depth_normalization(self,depth,method):
        '''translate original depth to different version'''
        #check_bin(depth.cpu())
        if method== 'inverse_square':
            depth=self.inverse_square(depth)
        elif method=='inverse':
            depth=self.inverse(depth)
        elif method == 'none':
            depth=self.none(depth)
        return depth
    def inverse_square(self,depth):
        depth[depth<=0.20] = 0.20
        depth=1/(5*depth)**2
        return depth
    def inverse(self,depth):
        depth[depth<=0.20] = 0.20
        depth=1/(5*depth)
        return depth
    def none(self,depth):
        return depth

    def __call__(self, data):
        """ transform
        """
        temp_data = data.copy()
        temp_data[:,:,self.depth_idx] = self.depth_normalization(temp_data[:,:,self.depth_idx],self.method) 
        temp_data[:,:,self.rad_idx] = self.inverse_square(temp_data[:,:,self.rad_idx])
        return temp_data

class Compose(transforms.Compose):
    pass