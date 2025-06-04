import shutil
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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
from PIL import Image, ImageDraw
from torchvision.transforms import ToPILImage
import torch

parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--dataset', type=str, default='/data/mingjundu/imagenet100k/')
parser.add_argument('--split',type=str, default='imagenet_subval')
parser.add_argument('--export_folder',type=str, default='imagetnet_samples')
parser.add_argument('--patch',type=int, default=10)
parser.add_argument('--size',type=int, default=240)
parser.add_argument("--cvd", type=str, default='deutan')
args = parser.parse_args()

df = pd.read_excel('name_table.xlsx',index_col='Colorname')  # 替换为您的文件路径
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
    # 'Cyan':9,
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
    input_lab = sRGB_to_Lab(rgb/255.)
    distances = np.linalg.norm(color_value_array_lab - input_lab, axis=1)
    if(input_lab[0]>10 and input_lab[0]<90 and abs(input_lab[1])<5 and abs(input_lab[2])<5):
        return 5
    index = np.argmin(distances)
    return category_map[color_name[index]]

class CVDImageNet(ImageFolder):
    def __init__(self, root: str, split: str = "train", patch_size=16, img_size=512,**kwargs: Any) -> None:
        target_path = os.path.join(root,split)
        super().__init__(target_path, **kwargs)
        self.image_size = img_size
        self.patch_size = patch_size
        self.my_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size), # resize short side as this length
                transforms.RandomCrop(self.image_size)  # to a square output
            ]
        )
        self.my_thumbnail_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size//self.patch_size),
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
        img_thumbnail = self.my_thumbnail_transform(img)
        return img,img_thumbnail,path


testset = CVDImageNet(dataset_path,split=args.split,patch_size=args.patch,img_size=args.size)
testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle = True,num_workers=1)

def make_data_label(loader,filename):
    """Given a dataloader, random sample color patch and name them. 
    Use under_sampling strategy to balance the samples in each class. 
    """
    X_train = []
    Y_train = []
    os.makedirs(args.export_folder, exist_ok=True)
    os.makedirs(args.export_folder+'_tmp', exist_ok=True)
    img_size = args.size//args.patch
    for img_full,img,path in tqdm(loader):
        to_pil_obj = ToPILImage()
        # save the 240x240 size image
        img_to_save = to_pil_obj(img_full[0])
        path = path[0] # tuple->str
        single_name = os.path.basename(path)
        path = os.path.join(args.export_folder+'_tmp', single_name)
        img_to_save.save(path)
        for i in range(50):  # 每张图采样十个颜色点，实际可能不会全部采纳
            color_names = np.array([-1,-1,-1, -1,-1,-1, -1,-1,-1])
            y_index = np.random.randint(1,img_size-1)
            x_index = np.random.randint(1,img_size-1)
            surroundings = img[0, :, y_index-1:y_index+2, x_index-1:x_index+2].reshape(3,-1)
            for j in range(9):
                color_names[j] = classify_color(surroundings[:,j].flatten().numpy()*255)
            target = color_names[4]
            target_patch_id = y_index*img_size+x_index  # patch_id = y*n_w+x
            mask = (color_names==target)
            if np.sum(mask)>=5: # voting,至少有5/9个像素(24*24分辨率上)支持当前颜色
                X_train.append((path,target_patch_id))
                Y_train.append(target)
        # if len(Y_train)>1000:   # debug
        #     time.sleep(0.5)
        #     break
    # 对数据进行抽样，以保证各类别样本数平衡       
    under_sampler = RandomUnderSampler(random_state=1337,sampling_strategy={category_map[i]:20 for i in category_map.keys()})
    X_resampled, Y_resampled = under_sampler.fit_resample(X_train,Y_train)
    print('Resample Statistic: ',Counter(Y_resampled))
    all_path,all_patch_id = zip(*X_resampled)
    # export all files in all_path to image folder
    all_path_new = []
    # copy file from original path to new folder
    for single_path in all_path:
        single_name = os.path.basename(single_path)
        new_file_path = os.path.join(args.export_folder, single_name)
        all_path_new.append(new_file_path)
        if not os.path.exists(new_file_path):
            shutil.copy(single_path, new_file_path)
    shutil.rmtree(args.export_folder+'_tmp')
    df_data = {'Path': all_path_new, 'Patch_ID': all_patch_id, 'Color_ID': Y_resampled}
    df_train = pd.DataFrame(data=df_data)
    df_train.to_csv(filename)
    return df_train

def get_patch_coordinates(patch_id):
    """Convert patch_id to coordinates
    patch_id = y*n_w+x
    """
    # Calculate top-left corner coordinates
    y = (patch_id // (args.size//args.patch)) * args.patch
    x = (patch_id % (args.size//args.patch)) * args.patch
    return x, y

def draw_anchor(label_file_path,target_img_folder):
    """With image & anchor information in label_file_path(csv), 
    draw anchor on the image and save them into new folder.  
    
    The label_file should be like:
    ```
    |Path|Patch_ID|Color_ID|  
    |---|---|---|  
    |/kaggle/ILSVRC2012_val_00017369.JPEG|75|0|  
    ```
    """
    # read df
    df = pd.read_csv(label_file_path)
    os.makedirs(target_img_folder,exist_ok=True)
    # retrive image_paths, patch_id, color_id for each item
    for i in range(len(df)):
        image_path = df.iloc[i]['Path']
        patch_id = df.iloc[i]['Patch_ID']
        color_id = df.iloc[i]['Color_ID']
        
        # read image, shold have size 240x240x3
        image = Image.open(image_path)
        # Calculate patch coordinates
        x, y = get_patch_coordinates(patch_id)
        
        # Create a copy of the image to draw on
        image_with_anchor = image.copy()
        draw = ImageDraw.Draw(image_with_anchor)
        
        # Draw anchor box (10x10 pixels)
        box_size = 10
        if color_id!=3:
            draw.rectangle([(x, y), (x + box_size, y + box_size)], outline="black", width=2)
        else:
            draw.rectangle([(x, y), (x + box_size, y + box_size)], outline="white", width=2)
        
        # Create patch preview (zoom in on the selected area)
        patch_preview = image.crop((x, y, x + box_size, y + box_size))
        patch_preview = patch_preview.resize((120, 120), Image.NEAREST)

        # Save the image with anchor and patch preview
        ori_path,ext = os.path.splitext(image_path)
        image_with_anchor_path = os.path.join(target_img_folder, f"{os.path.basename(ori_path)}"+f'_{patch_id}'+ext)
        patch_preview_path = os.path.join(target_img_folder, f"{os.path.basename(ori_path)}"+f'_{patch_id}_patch'+f'_{category_names[color_id]}'+ext)
        image_with_anchor.save(image_with_anchor_path)
        patch_preview.save(patch_preview_path)


df = make_data_label(testloader,os.path.join(args.export_folder,'test_label.csv'))
draw_anchor(os.path.join(args.export_folder,'test_label.csv'),args.export_folder+'_labeled')