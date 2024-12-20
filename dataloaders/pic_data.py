import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os


# 构建训练集、验证集;
class ImgDataset(Dataset):
    def __init__(self, split='train',workspace_dir='./dataset',input_size=224):
        self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.ToTensor(),
                                            ])
        self.input_size0 = input_size
        self.input_size1 = input_size
        if split=='train':
            self.x,self.y = self.readfile(os.path.join(workspace_dir, "training"))
            # label is required to be a LongTensor
            self.y = torch.LongTensor(self.y)
            print("Size of training data = {}".format(len(self.x)))
        else:
            self.x,self.y = self.readfile(os.path.join(workspace_dir, "testing"))
            # label is required to be a LongTensor
            self.y = torch.LongTensor(self.y)        
            print("Size of testing data = {}".format(len(self.x)))    


    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        X = self.transform(X)
        Y = self.y[index]
        return X, Y
    # 读取数据集文件(黑白);要使用彩色请看注释
    # 文件名格式为id，要放在对应类别的文件夹下(0,1,2...)
    def readfile(self,path):
        j=0
        file_count=0
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                file_count=file_count+1
        class_folders = sorted(os.listdir(path))
        print("\n Total %d classes and %d images "%(len(class_folders),file_count))
        # x = np.zeros((file_count, self.input_size0, self.input_size1), dtype=np.uint8)    # 黑白图片
        x = np.zeros((file_count, self.input_size0, self.input_size1,3), dtype=np.uint8)    # 彩色图片
        y = np.zeros((file_count), dtype=np.uint8)
        for sub_folders in class_folders:
            image_dir=os.listdir(os.path.join(path,sub_folders))
            # image_dir.sort(key= lambda x:int(x[:-4]))
            for i, file_name in enumerate(image_dir):
                # img = Image.open(os.path.join(path,sub_folders,file_name)).convert('L')   # 或使用opencv                
                # x[j+i, :] = np.array(img.resize((self.input_size0, self.input_size1)),dtype=np.uint8) # 或使用opencv

                # img = cv2.imread(os.path.join(path,sub_folders,file_name),0)    # 黑白图片
                # x[j+i, :] = cv2.resize(img, (self.input_size0, self.input_size1))    # 黑白图片

                img = cv2.imread(os.path.join(path,sub_folders,file_name))    # 彩色图片
                x[j+i, :, :] = cv2.resize(img, (self.input_size0, self.input_size1))   # 彩色图片
                y[j+i] = int(eval(sub_folders))
            j+=(i+1)
        print(y.shape)
        return x, y