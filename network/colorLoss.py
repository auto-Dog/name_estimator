import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class colorLoss(nn.Module):
    def __init__(self,tau=0.95):
        super().__init__()
        self.tau = tau
        self.mseLoss = nn.MSELoss()
        # store dataframe into dict
        df = pd.read_csv('basic_color_embeddings.csv',index_col='Name')
        self.color_name_embeddings_dict = {}
        all_embeddings = []
        self.all_names = {}
        for i,(index, row) in enumerate(df.iterrows()):
            single_row = row.to_numpy() / np.linalg.norm(row.to_numpy())    # IMPORTANT: embedding module is around 10, should undergo L2 NORM
            self.color_name_embeddings_dict[index] = torch.tensor(single_row).float().cuda()    
            # print(index,np.linalg.norm(single_row)) # debug
            all_embeddings.append(single_row)
            self.all_names[index] = torch.tensor(i,dtype=torch.long).cuda()
        self.all_embeddings_list = all_embeddings
        self.all_embeddings = torch.tensor(np.array(all_embeddings)).float().cuda()   # M colors, M x 768

    def infoNCELoss(self,x:torch.Tensor,x_names:tuple):
        x = x / x.norm(p=2, dim=-1, keepdim=True)   # L2 norm for cosine similarity
        # use str names
        embedding_gt = [self.color_name_embeddings_dict[x_name_i] for i,x_name_i in enumerate(x_names)]  
        embedding_gt = torch.vstack(embedding_gt)    # N x 768
        # # OR use index
        # embedding_gt = self.all_embeddings[x_names]     # N x 768

        # print('X shape:',x.shape)    # debug
        # print('Nx768 shape:',embedding_gt.shape)    # debug
        # print('Mx768 shape:',self.all_embeddings.shape)    # debug
        all_similarity = torch.matmul(x,self.all_embeddings.T)
        # print('dominator:',torch.exp(all_similarity/self.tau))  # debug
        all_similarity = torch.sum(torch.exp(all_similarity/self.tau),dim=1)    # Nx1
        
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
        # print('numerator:',numerator_similarity)  # debug
        contras_loss = -torch.log(numerator_similarity/all_similarity)
        return contras_loss,embedding_gt
    
    def forward(self,x:torch.Tensor,x_names:tuple):
        contras_loss,embedding_gt = self.infoNCELoss(x,x_names)
        mse_loss = self.mseLoss(x,embedding_gt)*x.shape[0]
        # total_loss = mse_loss
        total_loss = contras_loss.mean() + mse_loss
        return total_loss

    def classification(self,x:torch.Tensor,x_names:tuple):
        '''given N embeddings, return their cloest color type in index form'''
        x = x / x.norm(p=2, dim=-1, keepdim=True)   # L2 norm for cosine similarity
        all_similarity = torch.matmul(x,self.all_embeddings.T)  # B x classes
        val,class_index = torch.max(torch.exp(all_similarity),dim=1,keepdim=True)    # Nx1
        # use str names
        class_index_gt = [self.all_names[x_name_i] for i,x_name_i in enumerate(x_names)]   # get GT index of color
        class_index_gt = torch.vstack(class_index_gt)
        # # OR use index
        # class_index_gt = x_names
        return class_index,class_index_gt

if __name__ == '__main__':
    criteria = colorLoss()
    x = criteria.all_embeddings_list[2]  # blue
    # x = (x+criteria.all_embeddings_list[0])/2
    # x[10:100] = 0.
    x = torch.tensor(x).float().cuda().repeat(2,1) # 2x768
    # x = 10*torch.randn(2,768).float().cuda()
    colors = ('Blue','Blue')    # 2.45
    loss = criteria(x,colors)
    print('loss B-B',loss)
    colors = ('Red','Blue')  # 2.47
    loss = criteria(x,colors)
    print('loss B-R',loss)
