import argparse
import os
import time
import cv2
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.utils.data
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from dataloaders.CVDDS import CVDcifar,CVDImageNet,CVDPlace
from network import ViT,colorLoss
from utils.cvdObserver import cvdSimulateNet
import pandas as pd
import colour

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=8)
parser.add_argument('--size',type=int, default=256)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/work/mingjundu/imagenet100k/')
parser.add_argument("--cvd", type=str, default='deutan')
# C-Glow parameters
parser.add_argument("--x_bins", type=float, default=256.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=256.0)
parser.add_argument("--prefix", type=str, default='vit_cn5d')
args = parser.parse_args()
print(args) # show all parameters
### write model configs here
save_root = './run'
pth_location = './Models/model_'+args.prefix+'.pth'
model = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()
model.load_state_dict(torch.load(pth_location, map_location='cpu'))
model.eval()
criterion = colorLoss()
cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=False,batched_input=True) # 保证在同一个设备上进行全部运算

def render_ball(color=(0,0,0),):
    # 创建一个灰色背景的角落
    normalize_color = (color[0]/255., color[1]/255., color[2]/255.)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((60/255, 60/255, 60/255))
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    # # 创建三个灰色平面搭建角落
    # 二元函数定义域平面
    x = np.linspace(0, 12, 12)
    y = np.linspace(0, 12, 12)
    X, Y = np.meshgrid(x, y)
    # -------------------------------- 绘制 3D 图形 --------------------------------
    # 设置X、Y、Z面的背景是白色
    ax.xaxis.set_pane_color((0.4,0.4,0.4, 1.0))
    ax.yaxis.set_pane_color((0.8,0.8,0.8, 1.0))
    ax.zaxis.set_pane_color((0.3,0.3,0.3, 1.0))

    # 创建一个指定颜色的球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 5 * np.outer(np.cos(u), np.sin(v)) + 5  # 球体中心在(5, 5, 5)
    y = 5 * np.outer(np.sin(u), np.sin(v)) + 5
    z = 5 * np.outer(np.ones(np.size(u)), np.cos(v)) + 5
    ax.plot_surface(x, y, z, color=normalize_color, linewidth=0)  # 红色球体

    # 设置坐标轴标签
    ax.set_xlim([0,11])
    ax.set_ylim([0,11])
    ax.set_zlim([0,11])
    # ax.set_aspect(1)

    # 显示图形
    # plt.show()
    plt.savefig('tmp.png')
    plt.cla()
    plt.close("all")
    image_back = Image.open('tmp.png').convert('RGB')
    return np.array(image_back)

def render_patch(color=(0,0,0)):
    ''' 生成一个512x512的图像，中心色块为指定颜色。背景颜色为128,128,128'''
    image = np.full((512,512, 3), 128, dtype=np.uint8)

    # 获取色块的位置
    start_x = 200  # 中心位置向左偏移
    start_y = 200  # 中心位置向上偏移

    # 用户指定的色块RGB值
    color_rgb = color  # 示例红色

    # 在图像上绘制色块
    image[start_y:start_y+100, start_x:start_x+100,:] = color_rgb

    # 为色块添加2像素黑色边缘
    image[start_y:start_y+2, start_x:start_x+100,:] = (0, 0, 0)
    image[start_y+98:start_y+100, start_x:start_x+100,:] = (0, 0, 0)
    image[start_y:start_y+100, start_x:start_x+2,:] = (0, 0, 0)
    image[start_y:start_y+100, start_x+98:start_x+100,:] = (0, 0, 0)

    # # 显示图像
    # plt.imshow(image)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # pass
    return image, image[start_y+10:start_y+26, start_x+10:start_x+26,:]  # shape H,W,C
df = pd.read_excel('./name_table.xlsx',index_col='Colorname')  # 替换为您的文件路径
# 初始化字典
color_name = []
color_value = []
# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    # 获取颜色分类
    color_name.append(row['Classification'])
    # 将RGB字符串转换为数组
    rgb_array = [int(x) for x in row['RGB'].split(',')]
    color_value.append(rgb_array)

color_value_array = np.array(color_value)
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
category_names = list(category_map.keys())

def sRGB_to_Lab(rgb1):
    rgb_batch = np.float32(rgb1)
    # 重新调整输入数组的形状，使其成为 (n, 1, 3)，符合OpenCV的要求
    ori_shape = rgb_batch.shape
    rgb_batch = rgb_batch.reshape(-1, 1, 3)
    # 使用OpenCV的cvtColor函数转换RGB到Lab
    lab_batch = cv2.cvtColor(rgb_batch, cv2.COLOR_RGB2Lab)
    return lab_batch.reshape(ori_shape)  # 还原形状
color_value_array_lab = sRGB_to_Lab(color_value_array/255.)

def classify_color(rgb):
    # calculate norm as distance between input color and template colors
    ## use distance in RGB #
    # distances = np.linalg.norm(color_value_array - rgb, axis=1)
    ## or use distance in Lab #
    input_lab = sRGB_to_Lab(rgb/255.)
    distances = np.linalg.norm(color_value_array_lab - input_lab, axis=1)
    
    # # or use distance in HSV
    # color_value_array_hsv = colour.RGB_to_HSV(color_value_array/255.)
    # input_hsv = colour.RGB_to_HSV(rgb/255.)
    # distances = np.linalg.norm(color_value_array_hsv - input_hsv, axis=1)    
    # check if it is gray
    if(input_lab[0]>10 and input_lab[0]<90 and abs(input_lab[1])<5 and abs(input_lab[2])<5):
        return 5
    index = np.argmin(distances)
    return color_name[index]

# 产生指定颜色值
colors = []

# 定义色调
hues = ['5PB', '10B', '5B', '10BG', '5BG', '10G', '5G', '10GY', '5GY', '10Y', '5Y', '10YR', '5YR', '10R', '5R', '10RP', '5RP', '10P', '5P', '10PB']
# 定义明度等级
values = [3, 4, 5, 6, 7, 8, 9]

# 存储RGB颜色值
rgb_colors = []
df_mun = pd.read_csv('./munsell_xyY.csv')

def munsell_to_rgb(hue, value, chroma=6):
    ''' 用于转换Munsell颜色到RGB(0-255) '''
    max_srgb_exist = False
    while not max_srgb_exist:
        MRS_c = f'{hue} {value}/{chroma}'   # e.g. 4.2YR 8.1/5.3

        # The first step is to convert the *MRS* colour to *CIE xyY* 
        # colourspace.
        xyY = colour.munsell_colour_to_xyY(MRS_c)
        # We then perform conversion to *CIE xyY* tristimulus values.
        XYZ = colour.xyY_to_XYZ(xyY)

        # The last step will involve using the *Munsell Renotation System*
        # illuminant which is *CIE Illuminant C*:
        # http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/colorimetry/illuminants.ipynb#CIE-Illuminant-C
        # It is necessary in order to ensure white stays white when
        # converting to *sRGB* colourspace and its different whitepoint 
        # (*CIE Standard Illuminant D65*) by performing chromatic 
        # adaptation between the two different illuminant.
        # C = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C']
        RGB = colour.XYZ_to_sRGB(XYZ)*255
        if min(RGB)>=0 and max(RGB)<=255:
            max_srgb_exist = True
        chroma -= 2
    # print(f'{MRS_c}:{xyY}') # debug
    return RGB,MRS_c.replace('/','-')

def get_max_chroma(H: str, V: float) -> int:
    """
    根据色相 H 和明度 V，返回对应的最大彩度 Cmax。
    """
    # 查找特定 H 和 V 的数据行
    filtered_df = df_mun[(df_mun['h'] == H) & (df_mun['V'] == V)]
    
    # 如果找到了对应的数据，返回该条件下的最大彩度
    if not filtered_df.empty:
        Cmax = filtered_df['C'].max()  # 获取该条件下的最大彩度
        return Cmax
    else:
        raise ValueError(f"没有找到对应的 H: {H} 和 V: {V} 的数据")

colors = []
munsell_colors = []
# 生成颜色命名实验的颜色
for hue in hues:
    for value in values:
        max_chroma = get_max_chroma(hue,value)
        rgb,munsell_str = munsell_to_rgb(hue, value, max_chroma)
        colors.append(rgb)
        munsell_colors.append(munsell_str)

# # 生成deutan的混淆色
# for lg in range(0,256,50):
#     for lb in range(0,256,50):
#         colors.append((lg,lg,lb))

## 生成RGB空间中等间距的颜色
# for r in range(0, 256, 10):
#     for g in range(0, 256, 10):
#         for b in range(0, 256, 10):
#             colors.append((r, g, b))

label_list = []
pred_list = []
for rgb_value_ori in tqdm(colors):
    # 产生指定颜色值色块
    img,ci_patch = render_patch(rgb_value_ori)
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)/255.
    ci_patch = torch.from_numpy(ci_patch).permute(2,0,1).unsqueeze(0)/255.
    img = cvd_process(img)
    ci_patch = cvd_process(ci_patch)
    with torch.no_grad():
        outs = model(img.cuda())
    patch_color_name = classify_color(rgb_value_ori)
    outs = outs[:,512,:]    # 取最中间部分的输出embedding
    # print(outs.shape)   # debug
    pred,label = criterion.classification(outs,(patch_color_name,))
    # print('pred shape:',pred.shape,'label shape:',label.shape)  # debug
    label_list.extend(label.cpu().detach().tolist())
    pred_list.extend(pred.cpu().detach().tolist())

cls_report = classification_report(label_list, pred_list, output_dict=True, zero_division=0)
acc = accuracy_score(label_list, pred_list)
print(cls_report)   # all class information
results_out = np.hstack([label_list,pred_list])
np.savetxt( "colormap_sphere.csv", results_out, delimiter=",") # 保存结果

# for munsell experiment
print(pred_list) # debug
pred_mat = np.array(pred_list,dtype=int).flatten().reshape(-1,7).T
category_names = np.array(category_names,dtype='U10')
category_names_mat = category_names[pred_mat]
np.savetxt( "colormap_munsell.csv", category_names_mat, delimiter=",",fmt="%s") # 保存结果