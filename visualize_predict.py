import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import torch
import torch.nn as nn
from network import ViT,colorLoss
from utils.cvdObserver import cvdSimulateNet
from colour.blindness import matrix_cvd_Machado2009

prefix = 'vit_cn5'
pth_location = './Models/model_'+prefix+'.pth'
model = ViT('ColorViT', pretrained=False,image_size=512,patches=16,num_layers=6,num_heads=6,num_classes=1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()
model.load_state_dict(torch.load(pth_location, map_location='cpu'))
criterion = colorLoss()
cvd_process = cvdSimulateNet(cvd_type='deutan',cuda=False,batched_input=True)

# update at vit cn 5c
def classify_color(img_cvd:torch.tensor):
    model.eval()
    with torch.no_grad():
        outs = model(img_cvd.cuda()) 
    # print('Out shape:',outs.shape)  # debug
    outs = outs[0]  # 去掉batch维度
    # for vit cn5: give all index at once
    cls_index,_ = criterion.classification(outs,('Red',)*1024)  # the later parameter is used for fill blank, no meaning
    return cls_index

def cvd_simulate(img:np.ndarray,cvd_type='Deuteranomaly'):
    '''Given a 0-1 RGB image, simulate its appearance in CVD eyes to NT'''
    H_mat = matrix_cvd_Machado2009(cvd_type,1.)
    h,w,d = img.shape
    im1 = img.reshape(-1,d)
    im_dst1 = im1 @ H_mat.T
    # im_dst1 = cvd_simulation_tritran(im1)
    im_dst = im_dst1.reshape(h,w,d)
    im_dst[im_dst>1] = 1.
    im_dst[im_dst<0] = 0.
    return im_dst

def visualize_name(img:np.ndarray):
    '''Given a 0-1 RGB image, split it into 16x16 patches and show all the color names on it'''
    ori_shape = img.shape
    patches_y = max(1,ori_shape[0]//16)
    patches_x = max(1,ori_shape[1]//16)
    color_rep_value = {
            'Red': np.array((255, 0, 0)), 
            'Green': np.array((0, 255, 0)), 
            'Blue': np.array((0, 0, 255)), 
            'Black': np.array((0,0,0)),
            'White': np.array((255,255,255)),
            'Gray': np.array((192,192,192)),
            'Cyan': np.array((0,255,255)),
            'Yellow': np.array((255, 255, 0)), 
            'Orange': np.array((255, 165, 0)), 
            'Pink': np.array((255, 192, 203)), 
            'Purple': np.array((128, 0, 128)), 
            'Brown': np.array((139, 69, 19)) 
        }
    category_map = {
            'Red': 0,
            'Green': 1,
            'Blue': 2,
            'Black':3,
            'White':4,
            'Gray' :5,
            'Pink' :6,
            'Orange':7,
            'Purple':8,
            'Cyan':9,
            'Yellow':10,
            'Brown': 11
        }
    name_list = list(category_map.keys())
    # split into patches
    patches = []
    for patch_y_i in range(patches_y):
        for patch_x_i in range(patches_x):
            y_end = patch_y_i*16+16 if patch_y_i*16+16<ori_shape[0] else None
            x_end = patch_x_i*16+16 if patch_x_i*16+16<ori_shape[1] else None
            single_patch = img[patch_y_i*16:y_end,
                                 patch_x_i*16:x_end,
                                 :]
            single_patch_tensor = torch.from_numpy(single_patch).float().permute(2,0,1).unsqueeze(0)
            patch_cvd = cvd_process(single_patch_tensor)
            patches.append(patch_cvd)
    # calculate color names
    all_patch_name = []
    img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)
    img_cvd = cvd_process(img_tensor)
    patch_color_indexes = classify_color(img_cvd)
    print(patch_color_indexes.shape)    # debug
    for i in range(1024):
        patch_color_index = int(patch_color_indexes[i])
        patch_color_name = name_list[patch_color_index]
        all_patch_name.append(patch_color_name)
    # For vit cn4c and earlier version
    # for single_patch_cvd in patches:
    #     patch_color_index = classify_color(img_cvd,single_patch_cvd)
    #     patch_color_name = name_list[patch_color_index[0]]
    #     all_patch_name.append(patch_color_name)

    # show patches and name, 2 pixel grid
    patch_canvas = np.zeros((ori_shape[0]+(patches_y+1)*2,
                             ori_shape[1]+(patches_x+1)*2,
                             3))
    # cvd_rgb_image = cvd_simulate(img)
    cvd_rgb_image = img
    # put patches on canvas then label text name on them. #
    # Text name color should be the one in color_rep_value #
    for i, _ in enumerate(patches):
        patch_y_i, patch_x_i = divmod(i, patches_x)
        y_start = patch_y_i * 16
        x_start = patch_x_i * 16
        y_end = y_start+16 if y_start+16<ori_shape[0] else None
        x_end = x_start+16 if x_start+16<ori_shape[1] else None
        patch_cvd_rgb = cvd_rgb_image[y_start:y_end, x_start:x_end,:]
        y_start = patch_y_i * (16 + 2) + 2
        x_start = patch_x_i * (16 + 2) + 2
        y_end = y_start+patch_cvd_rgb.shape[0]
        x_end = x_start+patch_cvd_rgb.shape[1]
        patch_canvas[y_start:y_end, x_start:x_end,:] = patch_cvd_rgb
        
    # Add text labels
        name = all_patch_name[i]
        y_pos = patch_y_i * (16 + 2) + 12
        x_pos = patch_x_i * (16 + 2) + 4
        text_color = color_rep_value[name] /255
        # plt.text(x_pos+2, y_pos, name[0:3], color='white', fontsize=8)
        plt.text(x_pos, y_pos, '*', color=text_color, fontsize=12)
    patch_canvas = np.clip(patch_canvas,0,1)
    plt.imshow(patch_canvas)
    plt.axis('off')  # Turn off axis numbers
    plt.show()
    plt.savefig('predict_colors_on_image_cvd.png')

from PIL import Image
img = Image.open('apple-icon.png').convert('RGB').resize((512,512))
img = np.array(img)/255.
visualize_name(img)