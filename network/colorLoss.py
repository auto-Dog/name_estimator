import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class colorLoss(nn.Module):
    def __init__(self,tau=0.95):
        super().__init__()
        self.tau = tau
        # store dataframe into dict
        df = pd.read_csv('basic_color_embeddings.csv',index_col='Name')
        self.color_name_embeddings_dict = {}
        all_embeddings = []
        self.all_names = []
        for index, row in df.iterrows():
            single_row = row.to_numpy() / np.linalg.norm(row.to_numpy())    # IMPORTANT: embedding module is around 10, should undergo L2 NORM
            self.color_name_embeddings_dict[index] = torch.tensor(single_row).float().cuda()    
            # print(index,np.linalg.norm(single_row)) # debug
            all_embeddings.append(single_row)
            self.all_names.append(index)
        self.all_embeddings_list = all_embeddings
        self.all_embeddings = torch.tensor(np.array(all_embeddings)).float().cuda()   # M colors, M x 768

    def forward(self,x:torch.Tensor,x_names:torch.Tensor):
        # # use str names
        # embedding_gt = [self.color_name_embeddings_dict[x_name_i] for i,x_name_i in enumerate(x_names)]  
        # embedding_gt = torch.vstack(embedding_gt)    # N x 768
        # OR use index
        embedding_gt = self.all_embeddings[x_names]     # N x 768
        # print('X shape:',x.shape)    # debug
        # print('Nx768 shape:',embedding_gt.shape)    # debug
        # print('Mx768 shape:',self.all_embeddings.shape)    # debug
        all_similarity = torch.matmul(x,self.all_embeddings.T)
        all_similarity = torch.sum(torch.exp(all_similarity)/self.tau,dim=1)    # Nx1
        
        def tensor_row_dot(tensor1,tensor2):
            '''dot multiply on each row vector, whose indexes are the same'''
            # Convert each row into a 1x768/768x1 matrix
            tensor1 = tensor1.unsqueeze(1)  # Become nx1x768
            tensor2 = tensor2.unsqueeze(2)  # Become nx768x1

            # Perform batch matrix multiplication
            result = torch.bmm(tensor1, tensor2)  # Result is nx1x1

            # Squeeze the result into a nx1 vector
            result = result.squeeze()  # Become nx1
            return result

        numerator_similarity = torch.exp(tensor_row_dot(x,embedding_gt)/self.tau)  # Nx1
        # print(numerator_similarity,all_similarity)  # debug
        total_loss = -torch.log(numerator_similarity/all_similarity)
        total_loss = total_loss.mean()
        return total_loss

    def classification(self,x:torch.Tensor,x_names:torch.Tensor):
        '''given N embeddings, return their cloest color type in index form'''
        all_similarity = torch.matmul(x,self.all_embeddings.T)  # B x classes
        val,class_index = torch.max(torch.exp(all_similarity),dim=1)    # Nx1
        # # use str names
        # class_index_gt = [self.all_names.index(x_name_i) for i,x_name_i in enumerate(x_names)]   # get GT index of color
        # class_index_gt = torch.tensor(class_index_gt,dtype=torch.long, device='cuda')
        # OR use index
        class_index_gt = x_names
        return class_index,class_index_gt

if __name__ == '__main__':
    criteria = colorLoss()
    x = criteria.all_embeddings_list[2]  # blue
    x[10:100] = 0.
    x = torch.tensor(x).float().cuda().unsqueeze(0) # 1x768
    colors = torch.tensor([2])
    loss = criteria(x,colors)
    print('loss B-B',loss)
    colors = torch.tensor([0])
    loss = criteria(x,colors)
    print('loss B-R',loss)
