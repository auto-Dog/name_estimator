import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import torch
import torch.nn as nn
from network import ViT,colorLoss, colorFilter
from utils.cvdObserver import cvdSimulateNet
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
# from colour.blindness import matrix_cvd_Machado2009

prefix = 'vit_cn7aD80'
cvd_type = 'deutan_80'
# 单张推理, 使用pic_name, 多张推理, 使用folder_name, 并在folder下直接新建一个文件夹存储原图、增强后以及其颜色命名结果的图像
batch_inference = True
ori_image_folder_name = "C:/Users/alphadu/OneDrive/CIE_CURVE/CVD_simulation/enhancement/human_model_eval/human_test/全部原图"
pth_location = './Models/model_'+prefix+'.pth'
pth_optim_location = './Models/model_vit_cn6b_optim_base.pth'
image_size = 240
patch_size = 10
category_names = ['Red','Green','Blue','Black','White','Gray',
                  'Pink','Orange','Purple','Cyan','Yellow','Brown']

model = ViT('ColorViT', pretrained=False,image_size=image_size,patches=patch_size,num_layers=6,num_heads=6,num_classes=1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()
model.load_state_dict(torch.load(pth_location, map_location='cpu'))

enhancemodel = colorFilter().cuda()
enhancemodel = nn.DataParallel(enhancemodel,device_ids=list(range(torch.cuda.device_count())))
enhancemodel = enhancemodel.cuda()
enhancemodel.load_state_dict(torch.load(pth_optim_location, map_location='cpu'))

criterion = colorLoss()
cvd_process = cvdSimulateNet(cvd_type=cvd_type,cuda=True,batched_input=True)

# update at vit cn 5c
def classify_color(img_cvd:torch.tensor):
    model.eval()
    with torch.no_grad():
        outs = model(img_cvd.cuda()) 
    # print('Out shape:',outs.shape)  # debug
    outs = outs[0]  # 去掉batch维度
    # for vit cn5: give all index at once
    cls_index,_ = criterion.classification(outs,('Red',)*24*24)  # the later parameter is used for fill blank, no meaning
    return cls_index

def single_enhancement(img:np.ndarray):
    image_sample = torch.tensor(img).float().permute(2,0,1).unsqueeze(0)
    image_sample = image_sample.cuda()
    img_out = image_sample.clone()
    # 一次性生成方案：
    enhancemodel.eval()
    with torch.no_grad():
        img_t = enhancemodel(img_out)    # 采用cnn变换改变色彩
    img_out_array = img_t.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    img_out_array = np.clip(img_out_array,0.0,1.0)
    return img_out_array

# 遍历df，根据文件名，从原始表中读取图片、目标label
df = pd.read_csv('C:/Users/alphadu/OneDrive/CIE_CURVE/CVD_simulation/enhancement/human_model_eval/human_test/questionnaire_shuffle/processing_results.csv')
df_out = pd.DataFrame(columns=['Path', 'Patch_ID', 'Color_ID', f'Predicted_Color_ID_{cvd_type}',
                               f'Enhanced_Predicted_Color_ID_{cvd_type}',
                               f'Predicted_Color_Name_{cvd_type}', 
                               f'Enhanced_Predicted_Color_Name_{cvd_type}'])
for index,row in df.iterrows():
    # 读取原图
    img_path:str = row['original_image_name']
    # 热修补，记录中的后缀名应该是第一列的后缀名
    img_experiment_path = row['processed_filename']
    img_ext = os.path.splitext(img_experiment_path)[-1]
    img_path = img_path.replace('.png', img_ext)  # 替换为实验图像的后缀名
    img_path = os.path.join(ori_image_folder_name, img_path)
    img = Image.open(img_path).convert('RGB').resize((image_size,image_size))
    img = np.array(img)/255.
    img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).cuda()

    # 读取目标位置和GT颜色ID
    patch_id = row['patch_id']
    color_id = row['color_index']

    # 模型预测
    img_cvd = cvd_process(img_tensor)
    patch_color_indexes = classify_color(img_cvd)
    predicted_patch_color_index = int(patch_color_indexes[patch_id].cpu().detach().numpy())
    prediceted_patch_color_name = category_names[predicted_patch_color_index]
    # 增强后预测
    img_enhanced = single_enhancement(img)
    img_enhanced = torch.from_numpy(img_enhanced).float().permute(2,0,1).unsqueeze(0).cuda()
    patch_color_indexes_enhanced = classify_color(cvd_process(img_enhanced))
    predicted_patch_color_index_enhanced = int(patch_color_indexes_enhanced[patch_id].cpu().detach().numpy())
    predicted_patch_color_name_enhanced = category_names[predicted_patch_color_index_enhanced]
    print('Processed: ',img_path,'\t',predicted_patch_color_index,'\t',predicted_patch_color_index_enhanced,'\t',color_id,)
    # 保存结果
    df_out.loc[index] = [img_path, patch_id, color_id, 
                         predicted_patch_color_index, predicted_patch_color_index_enhanced,
                         prediceted_patch_color_name, predicted_patch_color_name_enhanced]

# 保存结果到CSV文件
df_out.to_csv('C:/Users/alphadu/OneDrive/CIE_CURVE/CVD_simulation/enhancement/human_model_eval/human_test/questionnaire_shuffle/processing_results_predicted.csv', index=False)
