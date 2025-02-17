import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,ImageNet,ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import os
from typing import Any, Callable, Optional, Tuple
from collections import Counter
from PIL import Image
import os
import pandas as pd
import json
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
import argparse
import colour

parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--dataset', type=str, default='/work/mingjundu/imagenet100k/')
args = parser.parse_args()

df = pd.read_excel('../name_table.xlsx',index_col='Colorname')  # 替换为您的文件路径
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

dataset_path = args.dataset
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
    return category_map[color_name[index]]

class CVDImageNet(ImageFolder):
    def __init__(self, root: str, split: str = "train", patch_size=16, img_size=512, cvd='deutan',**kwargs: Any) -> None:
        target_path = os.path.join(root,split)
        super().__init__(target_path, **kwargs)
        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.image_size,self.image_size)),
            ]
        )
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]  # names form ImageNet -> ImageFolder -> DatasetFolder
        sample = self.loader(path)
        img = self.my_transform(sample)
        return img, path

trainset = CVDImageNet(dataset_path,split='imagenet_subtrain',patch_size=16,img_size=32,cvd='deutan')
valset = CVDImageNet(dataset_path,split='imagenet_subval',patch_size=16,img_size=32,cvd='deutan')
trainloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle = True,num_workers=8)
valloader = torch.utils.data.DataLoader(valset,batch_size=1,shuffle = True,num_workers=8)

def make_data_label(loader,filename):
    X_train = []
    Y_train = []
    for img,path in tqdm(loader):
        for i in range(20):  # 每张图采样十个颜色点，实际可能不会全部采纳
            color_names = np.array([-1,-1,-1, -1,-1,-1, -1,-1,-1])
            x_index = np.random.randint(1,30)
            y_index = np.random.randint(1,30)
            surroundings = img[0, :, x_index-1:x_index+2, y_index-1:y_index+2].reshape(3,-1)
            for j in range(9):
                color_names[j] = classify_color(surroundings[:,j].flatten().numpy()*255)
            target = color_names[4]
            target_patch_id = x_index*32+y_index
            mask = (color_names==target)
            if np.sum(mask)>=5: # voting,至少有5/9个像素支持当前颜色
                X_train.append((path[0],target_patch_id))
                Y_train.append(target)
        # if len(Y_train)>1000:   # debug
        #     time.sleep(0.5)
        #     break
            
    under_sampler = RandomUnderSampler(random_state=1337)
    X_resampled, Y_resampled = under_sampler.fit_resample(X_train,Y_train)
    print('Resample result:',Counter(Y_resampled))
    all_path,all_patch_id = zip(*X_resampled)
    df_data = {'Path':all_path,'Patch_ID':all_patch_id,'Color_ID':Y_resampled}
    df_train = pd.DataFrame(data=df_data)
    df_train.to_csv(filename)

make_data_label(trainloader,'train_label.csv')
make_data_label(valloader,'val_label.csv')