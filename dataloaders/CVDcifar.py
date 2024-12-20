import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,ImageNet,ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import os
from typing import Any, Callable, Optional, Tuple
from utils.cvdObserver import cvdSimulateNet
from PIL import Image
import os
import pandas as pd

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
        self.color_name_embeddings = pd.read_csv('basic_color_embeddings.csv')

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
        
        # pre defined color names and their typical values
        color_categories = {
            'Red': np.array([(255, 0, 0), (255, 99, 71), (255, 69, 0)]), 
            'Green': np.array([(0, 255, 0), (34, 139, 34), (0, 128, 0)]), 
            'Blue': np.array([(0, 0, 255), (30, 144, 255), (70, 130, 180)]), 
            'Black': np.array([(0,0,0)]),
            'White': np.array([(255,255,255)]),
            'Gray': np.array([(192,192,192)]),
            'Cyan': np.array([(0,255,255)]),
            'Yellow': np.array([(255, 255, 0), (255, 255, 102), (255, 255, 51)]), 
            'Orange': np.array([(255, 165, 0), (255, 99, 71), (255, 140, 0)]), 
            'Pink': np.array([(255, 192, 203), (255, 105, 180), (255, 20, 147)]), 
            'Purple': np.array([(128, 0, 128), (160, 32, 240), (138, 43, 226)]), 
            'Brown': np.array([(139, 69, 19), (160, 82, 45), (165, 42, 42)]) 
        }

        def classify_color(rgb):
            # rgb = torch.tensor(rgb)  # RGB tuple to tensor
            min_distance = float('inf')
            closest_category = None
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
            # iterate all color types
            for category, color_list in color_categories.items():
                # calculate norm as distance between input color and template colors
                distances = torch.linalg.norm(color_list - rgb, dim=1)
                min_distance_for_category = np.min(distances)
                
                if min_distance_for_category < min_distance:
                    min_distance = min_distance_for_category
                    closest_category = category

            return closest_category,category_map.get(closest_category, -1)  # return color words and index
        
        color_patch_mean = torch.mean(color_patch,dim=[1,2])
        color_name,color_index = classify_color(color_patch_mean)
        color_embedding = self.color_name_embeddings.iloc[color_name].to_numpy()
        color_embedding = torch.from_numpy(color_embedding)

        return color_embedding, color_name

    
class CVDPlace(CVDImageNet):
    def __init__(self, root: str, split: str = "train", patch_size=4, img_size=64,**kwargs: Any) -> None:
        super().__init__(root, split, patch_size, img_size, cvd = 'deutan',**kwargs)

if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    CVD_set = CVDImageNet('/work/mingjundu/imagenet100k/')
    CVD_loader = torch.utils.data.DataLoader(CVD_set,batch_size=1,shuffle = True)
    for outputs in CVD_loader:
        img_ori = outputs[2]
        patch_ori = outputs[3]
        patch_ori_name = outputs[4]
        patch_ori_emb = outputs[5]
        show = ToPILImage()
        show(img_ori).save('img_ori.png')
        show(patch_ori).save('patch_ori.png')
        print('color name:',patch_ori_name)
        break
    

