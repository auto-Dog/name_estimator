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
        self.root = root
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
        self.category_map = {
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
        self.category_names = list(self.category_map.keys())
        try:
            self.data_label_df = pd.read_csv('dataloaders/'+split+'_label.csv',index_col=0)
        except:
            raise IOError('Label file does not exist. Run dataloaders/make_data.py')

    def __len__(self):
        return len(self.data_label_df)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        [path, patch_id, target] = self.data_label_df.loc[index].to_list()  # names form ImageNet -> ImageFolder -> DatasetFolder
        sample = self.loader(os.path.join(self.root,path))

        img = self.my_transform(sample)
        img_target = img.clone()
        patch_target = img[:, 
                           patch_id//32:patch_id//32+self.patch_size, 
                           patch_id%32:patch_id%32+self.patch_size]
        patch_color_embedding,patch_color_name = self.getEmbedding(target) # get color names   # debug
        patch = self.cvd_observer(patch_target)
        img = self.cvd_observer(img)

        return img, patch, img_target, patch_target, patch_color_name, patch_color_embedding # CVD image, CVD patch, image target, patch target
    
    def getEmbedding(self,target):
        '''Given a patch's color type number, return embedding and name'''
        color_name = self.category_names[target]
        color_embedding = self.color_name_embeddings.loc[color_name].to_numpy()
        color_embedding = torch.from_numpy(color_embedding)
        # color_index = torch.tensor(color_index,dtype=torch.long)
        return color_embedding, color_name
    
    def loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
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
    

