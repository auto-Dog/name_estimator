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

trainset = CVDImageNet('/work/mingjundu/imagenet100k/',split='imagenet_subtrain',patch_size=16,img_size=32,cvd='deutan')
valset = CVDImageNet('/work/mingjundu/imagenet100k/',split='imagenet_subtest',patch_size=16,img_size=32,cvd='deutan')
trainloader = torch.utils.data.DataLoader(trainset,batch_size=1,shuffle = True,num_workers=4)
valloader = torch.utils.data.DataLoader(valset,batch_size=1,shuffle = True,num_workers=4)

X_train = []
Y_train = []
row_index = 0
for img,path in tqdm(trainloader):
    for i in range(5):  # 每张图采样五个颜色点，实际可能不会全部采纳
        color_names = [-1,-1,-1, -1,-1,-1, -1,-1,-1]
        x_index = torch.randint(1,30)
        y_index = torch.randint(1,30)
        surroundings = img[0, :, x_index-1:x_index+2, y_index-1:y_index+2].reshape(3,-1)
        for j in range(9):
            color_names[j] = classify_color(surroundings[:,j].flatten().numpy()*255)
        target = color_names[4]
        target_patch_id = x_index*32+y_index
        mask = (color_names==target)
        if np.sum(mask)>=3: # voting,至少有3/9个像素支持当前颜色
            X_train.append((path,target_patch_id))
            Y_train.append(target)
        
under_sampler = RandomUnderSampler(random_state=1337)
X_resampled, Y_resampled = under_sampler.fit_resample(X_train,Y_train)
print('Resample result:',Counter(Y_resampled))
all_path,all_patch_id = zip(*X_resampled)
df_data = {'Path':all_path,'Patch_ID':all_patch_id,'Color_ID':Y_resampled}
df_train = pd.DataFrame(data=df_data)
df_train.to_csv('train_label.csv')