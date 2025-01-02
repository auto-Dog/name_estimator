import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,ImageNet,ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import os
from typing import Any, Callable, Optional, Tuple
# import sys
# sys.path.append("..")   # debug
from utils.cvdObserver import cvdSimulateNet
from PIL import Image
import os
import pandas as pd
import json

class CVDcifar(CIFAR10):
    def __init__(        
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        patch_size = 16,
        img_size = 512,
        cvd = 'deutan'
    ) -> None:
        super().__init__(root,train,transform,target_transform,download)

        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Resize(self.image_size),
            ]
        )
        self.cvd_observer = cvdSimulateNet(cvd)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index] # names from CIFAR10

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = self.my_transform(img)
        img_target = img.clone()
        random_index = torch.randint(0,self.image_size//self.patch_size,size=(2,))
        patch_target = img[:, random_index[0]*self.patch_size:random_index[0]*self.patch_size+self.patch_size, 
                           random_index[1]*self.patch_size:random_index[1]*self.patch_size+self.patch_size]
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target # CVD image, CVD patch, image target, patch target
    
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
        self.cvd_observer = cvdSimulateNet(cvd)
        self.color_name_embeddings = pd.read_csv('basic_color_embeddings.csv',index_col='Name')
        df = pd.read_excel('name_table.xlsx',index_col='Colorname')  # 替换为您的文件路径
        # 初始化字典
        self.color_names = []
        color_value = []
        # 遍历DataFrame中的每一行
        for index, row in df.iterrows():
            # 获取颜色分类
            self.color_names.append(row['Classification'])
            # 将RGB字符串转换为数组
            rgb_array = [int(x) for x in row['RGB'].split(',')]
            color_value.append(rgb_array)

        self.color_value_array = np.array(color_value)
        
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
        img_target = img.clone()
        random_index = torch.randint(0,self.image_size//self.patch_size,size=(2,))
        patch_target = img[:, random_index[0]*self.patch_size:random_index[0]*self.patch_size+self.patch_size, 
                           random_index[1]*self.patch_size:random_index[1]*self.patch_size+self.patch_size]
        patch_color_embedding,patch_color_name = self.getEmbedding(patch_target) # get color names   # debug
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target, patch_color_name, patch_color_embedding # CVD image, CVD patch, image target, patch target
    
    def getEmbedding(self,color_patch):
        '''Given a color patch, return its color type number and embedding'''
        def classify_color(rgb):
            rgb = rgb.numpy()  # RGB tensor to numpy
            # calculate norm as distance between input color and template colors
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
            distances = np.linalg.norm(self.color_value_array - rgb, axis=1)
            index = np.argmin(distances)
            return self.color_names[index], category_map[self.color_names[index]]   # return color words and index
        
        color_patch_mean = torch.mean(color_patch,dim=[1,2])*255
        color_name,color_index = classify_color(color_patch_mean)
        color_embedding = self.color_name_embeddings.loc[color_name].to_numpy()
        color_embedding = torch.from_numpy(color_embedding)
        # color_index = torch.tensor(color_index,dtype=torch.long)
        return color_embedding, color_name

    
class CVDPlace(CVDImageNet):
    def __init__(self, root: str, split: str = "train", patch_size=4, img_size=64,**kwargs: Any) -> None:
        super().__init__(root, split, patch_size, img_size, cvd = 'deutan',**kwargs)

if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    CVD_set = CVDImageNet('/work/mingjundu/imagenet100k/',split='imagenet_subtrain')
    CVD_loader = torch.utils.data.DataLoader(CVD_set,batch_size=10,shuffle = True)
    for outputs in CVD_loader:
        img_ori = outputs[2]
        patch_ori = outputs[3]
        patch_ori_name = outputs[4]
        patch_ori_emb = outputs[5]
        show = ToPILImage()
        show(img_ori[0]).save('img_ori.png')
        show(patch_ori[0]).save('patch_ori.png')
        print('color name:',patch_ori_name)
        break
    

