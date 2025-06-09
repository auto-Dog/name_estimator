import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import pathlib
from scipy.io import loadmat
from torch import tensor as to_tensor
from torchvision.transforms.functional import pil_to_tensor

class ColorNaming():
    def __init__(self, matrix_path="w2c.mat",device='cuda'):
        """ Van de Weijer et al. (2009) Color Naming model python implementation.
        Van De Weijer, J. et al. Learning color names for real-world applications. IEEE Transactions on Image Processing
        The class is based on the MATLAB implementation by Van de Weijer et al. (2009) and it needs the w2c.mat original
        file. The input RGB image is converted to a set (6 or 11) color naming probability maps.
        """
        self.matrix = to_tensor(loadmat(matrix_path)['w2c']).to(device)
        self.device = device
        # self.top_k = top_k

    def __call__(self, input_tensor):
        """Converts an RGB image to a color naming image.

        Args:
        input_tensor: batch of RGB images (B x 3)

        Returns:
            torch.tensor: Color naming image. (B x 11)
        """
        # Reconvert image to [0-255] range
        input_tensor = torch.clamp(input_tensor, 0, 1)
        img = (input_tensor * 255).int()

        index_tensor = torch.floor(
            img[:, 0, ...].view(img.shape[0], -1) / 8).long() + 32 * torch.floor(
            img[:, 1, ...].view(img.shape[0], -1) / 8).long() + 32 * 32 * torch.floor(
            img[:, 2, ...].view(img.shape[0], -1) / 8).long()

        prob_maps = []
        for w2cM in self.matrix.permute(*torch.arange(self.matrix.ndim-1, -1, -1)):
            out = w2cM[index_tensor]
            prob_maps.append(out)
        prob_maps = torch.stack(prob_maps, dim=1).squeeze(2)
        # # set probability to 0 except top_k colors
        # k_largest_value = torch.topk(prob_maps, self.top_k, dim=1)[0][:, -1]
        # prob_maps[prob_maps<k_largest_value] = 0.
        # prob_maps = F.softmax(prob_maps, dim=1)
        return prob_maps

class colorLossEnhance(nn.Module):
    def __init__(self,tau=0.95,device='cuda'):
        super().__init__()
        self.color_namer = ColorNaming()
        self.KL_loss = nn.KLDivLoss(reduction='none')
        # store dataframe into dict
        df = pd.read_csv('basic_color_embeddings.csv',index_col='Name')
        self.color_name_embeddings_dict = {}
        all_embeddings = []
        self.all_names = {}
        self.tau = tau
        for i,(index, row) in enumerate(df.iterrows()):
            single_row = row.to_numpy() / np.linalg.norm(row.to_numpy())    # IMPORTANT: embedding module is around 10, should undergo L2 NORM
            self.color_name_embeddings_dict[index] = torch.tensor(single_row).float().to(device)    
            # print(index,np.linalg.norm(single_row)) # debug
            all_embeddings.append(single_row)
            self.all_names[index] = torch.tensor(i,dtype=torch.long).to(device) 
        self.name_to_index = {name: i for i, name in enumerate(self.color_name_embeddings_dict.keys())}
        self.name_lists = list(self.color_name_embeddings_dict.keys())
        # drop "Cyan" at index 9
        self.name_lists.pop(9)
        self.all_embeddings_list = all_embeddings
        self.all_embeddings = torch.tensor(np.array(all_embeddings)).float().to(device)    # M colors, M x 768

    def forward(self,x:torch.Tensor,x_patch:torch.Tensor):
        average_rgb = torch.mean(x_patch,dim=(2,3))   # output Nx3
        color_logits = self.color_namer(average_rgb)    # output Nx11
        color_logits = self.interleave(color_logits)    
        # print('color_logits:',color_logits)  # debug
        predict_logits = self.get_logits(x) # output Nx11
        # print('predict_logits:',predict_logits)  # debug
        # KL loss
        loss = self.KL_loss(color_logits.log(),predict_logits)  # y dist. and x dist.
        return loss.mean()
    
    def get_logits(self,x:torch.Tensor):
        '''given N embeddings, return their cloest color type in index form
        Also return GT index from color names
        '''
        x = x / x.norm(p=2, dim=-1, keepdim=True)   # L2 norm for cosine similarity
        all_similarity = torch.matmul(x,self.all_embeddings.T)  # B x classes
        # drop "Cyan" at index 9
        all_similarity = torch.cat((all_similarity[:,:9],all_similarity[:,10:]),dim=1)
        logits = F.softmax(all_similarity/self.tau,dim=1)
        return logits
    
    def interleave(self, x):
        """Logits from color namer have name orders:
        black, blue, brown, gray, green, orange, pink, purple, red, white, yellow
        Whereas ours are stored in self.name_lists
        Interleave the logits order in x to match the order of self.name_lists
        Args:
            x (torch.tensor): Input tensor. (B x 11)
        Returns:
            torch.tensor: Interleaved tensor. (B x 11)
        """
        # Create a tensor with the indices of the desired order
        cname_list = ['Black', 'Blue', 'Brown', 'Gray', 
                    'Green', 'Orange', 'Pink', 'Purple', 
                    'Red', 'White', 'Yellow']
        indices = torch.tensor([cname_list.index(name) for name in self.name_lists],dtype=torch.long)
        # Use advanced indexing to reorder the tensor
        return x[:, indices]

if __name__ == '__main__':
    criteria = colorLossEnhance(tau=0.3)
    x = criteria.all_embeddings_list[2]  # blue
    input_tensor_cname = torch.tensor([1,2,3,4,5,6,7,8,9,10,11]).float().repeat(2,1).cuda()
    output_tensor_cname = criteria.interleave(input_tensor_cname)
    print(output_tensor_cname)
    # x = (x+criteria.all_embeddings_list[0])/2
    x[10:100] = 0.
    x = torch.tensor(x).float().repeat(2,1).cuda() # 2x768
    # x = 10*torch.randn(2,768).float()
    color_patch = torch.tensor([[0,92,95],[95,158,160]]).cuda().reshape(2,3,1,1)/255.
    loss = criteria(x,color_patch)
    print(loss)
