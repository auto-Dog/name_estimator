import argparse
import os
import time

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

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=16)
parser.add_argument('--size',type=int, default=512)
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
parser.add_argument("--prefix", type=str, default='vit_cn2')
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

def classify_color(rgb):
    # calculate norm as distance between input color and template colors
    distances = np.linalg.norm(color_value_array - rgb, axis=1)
    index = np.argmin(distances)
    return color_name[index]

# 产生指定颜色值
colors = []
for r in range(0, 256, 10):
    for g in range(0, 256, 10):
        for b in range(0, 256, 10):
            colors.append((r, g, b))
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
        outs = model(img.cuda(),ci_patch.cuda())
    patch_color_name = classify_color(rgb_value_ori)
    pred,label = criterion.classification(outs,(patch_color_name,))
    label_list.extend(label.cpu().detach().tolist())
    pred_list.extend(pred.cpu().detach().tolist())

cls_report = classification_report(label_list, pred_list, output_dict=True, zero_division=0)
acc = accuracy_score(label_list, pred_list)
print(cls_report)   # all class information
results_out = np.hstack([label_list,pred_list])
np.savetxt( "colormap_sphere.csv", results_out, delimiter=",") # 保存结果