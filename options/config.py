import os
import numpy as np
import torch
import random
from collections import OrderedDict

HDD_DIR =  '/home/mm/IR-material/Dataset/'  # the root direction of the data
MODEL_DIR = os.path.join(HDD_DIR,'models')
RESULT_DIR = os.path.join(HDD_DIR,'Result')
DATASET_DIR = os.path.join(HDD_DIR,'Dataset')

material_list_white_10=['absplastic_white', 'absplasticrough_white', 'acrylic_white', 'aluminum_silver', 'hardpaper_white', 'felt_white', 'metal_white', 'sponge_white', 'towel_white', 'wovenbag_white']
material_list_40=sorted(['absplastic_white', 'absplasticrough_white', 'acrylic_white', 'aluminum_silver', 'artificialstone_yellow', 'calciumsilicate_white', 'carpet_blackstripe', 'carpet_green', 'carpet_yellow', 'carpet_yellowstripe', 'cloth_black', 'cloth_green', 'cloth_white', 'felt_white', 'foam_white', 'hardcloth_green', 'hardpaper_white', 'hardpaper_yellow', 'leather_black', 'leather_yellow', 'linen_fine', 'linen_rough', 'metal_white', 'oilcloth_green', 'paperposter_white', 'siliconefoam_red', 'sponge_black', 'sponge_white', 'towel_white', 'velvet_yellow', 'wall_white', 'wallpaperpvc_blue', 'wallpaperpvc_grey', 'wallpaperpvc_white', 'wallpaperpvc_wood', 'wood_white', 'woodbare_brown', 'woodbare_yellow', 'sponge_yellow', 'wovenbag_white'])
material_deformed_12=['velvet_yellow','sponge_white','siliconefoam_red','rubber_white','oilcloth_green','linen_rough','linen_fine','leather_yellow','leather_black','hardcloth_green','cloth_green','cloth_black']
material_dict={
    '40':material_list_40,
    'white10':material_list_white_10,
    'deform_12':material_deformed_12
}
material_group={
    'cloth_combine':['cloth_white','cloth_black','cloth_green'],
    'carpet_combine':['carpet_yellow','carpet_green','carpet_blackstripe','carpet_yellowstripe'],
    'wallpaperpvc_combine':['wallpaperpvc_blue','wallpaperpvc_grey','wallpaperpvc_white','wallpaperpvc_wood'],
    'ceramictile_combine':['ceramictile_white','ceramictile_grey'],
    'sponge_combine':['sponge_white','sponge_black','sponge_yellow'],
    'wovenbag_combine':['wovenbag_green','wovenbag_white'],
    # 'linen_combine':['linen_rough','linen_fine']
}

def origin2materialCapture(groups,origin_list):
    map_name_dict=OrderedDict()
    for old_name in origin_list:
        map_name_dict[old_name]=old_name
    for new_name,group_list in groups.items():
        for material_name in group_list:
            if material_name in origin_list: ## only names in groups are needed to change 
                map_name_dict[material_name]=new_name
    
    material_list = list(set(map_name_dict.values()))
    map_idx_dict=OrderedDict()
    for old_name,new_name in map_name_dict.items():
        map_idx_dict[origin_list.index(old_name)] = material_list.index(new_name)

    return np.vectorize(map_idx_dict.get),material_list

random.seed(10)
color_config=list()
for i in range(256):
    color_config.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))

colormap = np.zeros((256,1,3), np.uint8)
for i in range(255):
    colormap[i:i+1,:]=color_config[i]

#incidence orientation
ori_z=torch.ones((1,1,240,320),dtype=(torch.float))
ori_x=torch.ones((1,1,240,320),dtype=(torch.float))
ori_y=torch.ones((1,1,240,320),dtype=(torch.float))
for i in range(240):
    for j in range(320):
        ori_x[:,:,i,j]=0.5543*(j-159.5)/159.5
        ori_y[:,:,i,j]=0.41420*(i-119.5)/119.5
inc=torch.cat((ori_x,ori_y,ori_z),dim=1)

