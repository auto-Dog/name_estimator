import numpy as np
import sys
import torch
import torch.nn as nn
# import colour
from PIL import Image
import matplotlib.pyplot as plt

class cvdObserver:
    '''一个色觉障碍人士的模拟观察'''
    def __init__(self,severe = 0.1,cvd_type='PRO') -> None:
        self.CVDCMF = []
        beta_L, beta_M, beta_S = (0.63, 0.32, 0.05)
        pass
    
    def getCMF(self)->np.ndarray:
        '''返回色弱人士的CMF曲线'''

    def getLMSmetric(self,center_color:np.ndarray)->np.ndarray:
        '''返回指定aLMS颜色的色差椭球参数'''

    def cvd_sRGB_to_LMS(self,image_pixels)->np.ndarray:
        '''将sRGB颜色转换为CVD的aLMS精确值'''

    def quantify_LMS(self,lms_image_pixels)->np.ndarray:
        '''根据色差椭球参数，将精确aLMS值转为感知aLMS值，使得观察混淆色对计算机也混淆'''


class cvdSimulateNet(nn.Module):
    ''' 将输入图像转成CVD眼中的图像，目前只模拟色盲'''
    def __init__(self, cvd_type='protan', cuda=False, batched_input=False) -> None:
        super().__init__()
        self.cvd_type = cvd_type
        self.cuda_flag = cuda
        self.batched_input = batched_input
        beta_L, beta_M, beta_S = (0.63, 0.32, 0.05)
        self.beta_L = beta_L
        self.beta_M = beta_M
        self.beta_S = beta_S
        # lms_to_xyz_mat = np.array([[1.93986443, 1.34664359, 0.43044935, ],
        #                 [0.69283932, 0.34967567, 0, ],
        #                 [0, 0, 2.14687945]])    # 2006 10 degree
        # xyz_to_lms_mat = np.linalg.inv(lms_to_xyz_mat)
        # print(xyz_to_lms_mat)   # debug
        # xyz_to_lms_mat = (np.diag([beta_L,beta_M,beta_S]))@ xyz_to_lms_mat  # 根据视锥细胞比例更新变换矩阵
        xyz_to_lms_mat = np.array([[0.34*beta_L, 0.69*beta_L, -0.076*beta_L ],
                        [-0.39*beta_M, 1.17*beta_M, 0.049*beta_M ],
                        [0.038*beta_S, -0.043*beta_S, 0.48*beta_S]])
        self.xyz_to_lms_mat = torch.from_numpy(xyz_to_lms_mat).float()
        rgb_to_xyz_mat = np.array([[0.4124,0.3576,0.1805],
                                   [0.2126,0.7152,0.0722],
                                   [0.0193,0.1192,0.9505]])   # BT 709.2
        self.rgb_to_xyz_mat = torch.from_numpy(rgb_to_xyz_mat).float()

    def einsum_dot_tensor(self,batched_image,matrix):    # input BCHW
        return torch.einsum('vi,biju->bvju',matrix,batched_image)
    
    def sRGB_to_alms(self,image_sRGB:torch.tensor):
        if self.cuda_flag:
            mask_srgb = (image_sRGB<=0.04045)
            mask_srgb= mask_srgb.cuda()
            self.xyz_to_lms_mat=self.xyz_to_lms_mat.cuda()
            self.rgb_to_xyz_mat=self.rgb_to_xyz_mat.cuda()
            linear_RGB = torch.zeros_like(image_sRGB).cuda()
            image_xyz = torch.zeros_like(image_sRGB).cuda()
            image_alms = torch.zeros_like(image_sRGB).cuda()
            linear_RGB[mask_srgb] = image_sRGB[mask_srgb]/12.92
            linear_RGB[~mask_srgb] = torch.pow((image_sRGB[~mask_srgb]+0.055)/1.055,torch.tensor(2.4).cuda())# decode to linear
        else:
            mask_srgb = (image_sRGB<=0.04045)
            linear_RGB = torch.zeros_like(image_sRGB)
            image_xyz = torch.zeros_like(image_sRGB)
            image_alms = torch.zeros_like(image_sRGB)
            linear_RGB[mask_srgb] = image_sRGB[mask_srgb]/12.92
            linear_RGB[~mask_srgb] = torch.pow((image_sRGB[~mask_srgb]+0.055)/1.055,torch.tensor(2.4))# decode to linear
        # print(f'Shape {linear_RGB.shape}')   # debug
        image_xyz = self.einsum_dot_tensor(linear_RGB,self.rgb_to_xyz_mat,)
        image_alms = self.einsum_dot_tensor(image_xyz,self.xyz_to_lms_mat,)
        return image_alms
    
    def lRGB_to_alms(self,linear_RGB:torch.tensor):
        # print(f'Shape {linear_RGB.shape}')   # debug
        image_xyz = self.einsum_dot_tensor(linear_RGB,self.rgb_to_xyz_mat,)
        image_alms = self.einsum_dot_tensor(image_xyz,self.xyz_to_lms_mat,)
        return image_alms


    # def quantify_alms(self,image_alms:torch.tensor):
    #     ''' image_alms：视锥细胞响应，值域0-1.  
    #     假设环境光最大光子数1e5，等步长量化，即采用aLMS均为1e4点的色差椭球，其轴长为1*1e2/2e5'''
    #     channel_max=(0.6,0.6,0.025)
    #     bins=256
    #     quantify_out = image_alms
    #     quantify_out[:,0,:,:] = image_alms[:,0,:,:]/channel_max[0]
    #     quantify_out[:,1,:,:] = image_alms[:,1,:,:]/channel_max[1]
    #     quantify_out[:,2,:,:] = image_alms[:,2,:,:]/channel_max[2]*bins
    #     ori_noise /= bins
    #     out = input+ori_noise
    #     return out

    def add_noise(self,image_alms:torch.tensor,Y_max=2e4):
        ''' image_alms：视锥细胞响应，值域0-1. 噪声阈值为取值的sqrt倍. 
        Fonseca, María da, and Inés Samengo. 
        Derivation of Human Chromatic Discrimination Ability from an Information-Theoretical Notion of Distance in Color Space.
        https://doi.org/10.1162/NECO_A_00903.
        '''
        image_alms_denormalized = image_alms*Y_max
        noise_std = torch.sqrt(image_alms_denormalized)
        noise = torch.randn_like(image_alms_denormalized) * noise_std
        noisy_image = image_alms_denormalized + noise
        # print('Noise stds:', noise_std/Y_max)  # debug
        image_alms = noisy_image / Y_max  # 归一化回0-1范围
        return image_alms.clamp(0, 1)  # 确保值域在0-1之间

    def forward(self,image_sRGB):
        ''' image RGB: 未经处理的原始图像，值域0-1'''
        if not self.batched_input:
            image_sRGB = image_sRGB.unsqueeze(0)
        lms_image = self.sRGB_to_alms(image_sRGB)
        lms_image_cvd = lms_image
        if self.cvd_type == 'protan':
            lms_image_cvd[:,0,:,:] = lms_image_cvd[:,1,:,:]/self.beta_M*self.beta_L # 缺L
        elif self.cvd_type == 'deutan':
            lms_image_cvd[:,1,:,:] = lms_image_cvd[:,0,:,:]/self.beta_L*self.beta_M # 缺M
        elif 'protan_' in self.cvd_type:   # protan_xxx形式
            degree = int(self.cvd_type.split('_')[1])
            lms_image_cvd[:,0,:,:] = degree/100.*lms_image_cvd[:,1,:,:]/self.beta_M*self.beta_L +\
                                    (1-degree/100.)*lms_image_cvd[:,0,:,:] # 部分缺L
        elif 'deutan_' in self.cvd_type:   # deutan_xxx形式
            degree = int(self.cvd_type.split('_')[1])
            lms_image_cvd[:,1,:,:] = degree/100.*lms_image_cvd[:,0,:,:]/self.beta_L*self.beta_M +\
                                    (1-degree/100.)*lms_image_cvd[:,1,:,:] # 部分缺M
        if not self.batched_input:
            lms_image_cvd = lms_image_cvd.squeeze(0)
        lms_image_cvd = self.add_noise(lms_image_cvd)  # 添加噪声，模拟人眼对颜色的感知 y=Hx+n
        return lms_image_cvd
    
if __name__ == '__main__':
    myobserver = cvdSimulateNet(cvd_type='protan_50',cuda=False,batched_input=True)
    image_sample_ori = Image.open('../../../CVD_test.png').convert('RGB')
    image_sample_ori = torch.tensor(np.array(image_sample_ori)).permute(2,0,1).unsqueeze(0)/255.
    # image_sample_ori = torch.tensor(np.array([[[0.35003133, 0.83144364, 0.6597791],
    #                                            [0.35003133, 0.83144364, 0.6597791],
    #                                            [0.58516089, 0.77587424, 0.66080132],
    #                                            [0.95678832, 0.56522484, 0.66385579],
    #                                            [1.0,1.0,1.0]]])).permute(2,0,1).unsqueeze(0).float()
    image_sample = image_sample_ori.clone()
    print(image_sample.is_leaf)
    # Test loss backward ability
    image_sample.requires_grad = True
    # image_sample = image_sample.cuda()
    image_sample_o = myobserver(image_sample)
    # image_loss = torch.mean(torch.exp(-image_sample_o))
    # image_loss.backward()
    # print(image_sample.grad)

    image_array = image_sample_o.squeeze(0).permute(1,2,0).detach().numpy()
    print(image_array)  # debug
    # print(np.min(image_array),np.max(image_array))  # debug
    image_array = image_array/np.array([0.58,0.26,0.024])   # 归一化，方便查看
    plt.imshow(image_array)
    plt.show()  # 如果看到一只鹿，证明变换有效
