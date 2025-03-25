# We should train the enhancement model after two classifiers are trained
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
# from sklearn.model_selection import StratifiedGroupKFold
from utils.logger import Logger
from tqdm import tqdm
from dataloaders.CVDDS import CVDcifar,CVDImageNet,CVDPlace,CVDImageNetRand
from network import ViT,colorLoss, colorFilter,criticNet,SSIMLoss
from utils.cvdObserver import cvdSimulateNet
from utils.conditionP import conditionP
from utils.utility import patch_split,patch_compose
from torch.autograd import Variable
from torch import autograd

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=8)
parser.add_argument('--size',type=int, default=256)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/data/mingjundu/imagenet100k/')
parser.add_argument("--cvd", type=str, default='deutan')
parser.add_argument("--tau", type=float, default=0.3)
parser.add_argument("--n_critic", type=int, default=10)
parser.add_argument("--x_bins", type=float, default=128.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=128.0)
parser.add_argument("--prefix", type=str, default='vit_cn5')
parser.add_argument('--from_check_point',type=str,default='')
args = parser.parse_args()
print(args) # show all parameters
save_root = './run'
pth_location = './Models/model_'+args.prefix+'.pth'
pth_nt_location = './Models/model_'+args.prefix+'_nt.pth'
pth_optim_location = './Models/model_'+args.prefix+'_optim'+'.pth'
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
train_val_percent = 0.8

trainset = CVDImageNet(args.dataset,split='imagenet_subtrain',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
valset = CVDImageNet(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # cvd模拟器
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True,num_workers=8)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True,num_workers=8)
model = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()

model_critic = criticNet().cuda()
model_critic = nn.DataParallel(model_critic,device_ids=list(range(torch.cuda.device_count())))
model_critic = model_critic.cuda()

model_enhance = colorFilter().cuda()
model_enhance = nn.DataParallel(model_enhance,device_ids=list(range(torch.cuda.device_count())))
model_enhance = model_enhance.cuda()

criterion = colorLoss(args.tau)
criterion_L1 = nn.L1Loss()
criterion_ssim = SSIMLoss()
# optimizer_critic = torch.optim.RMSprop(model_critic.parameters(), lr=args.lr, weight_decay=5e-5)    # update WGAN-CP optimizer, same as original paper
# optimizer_enhance = torch.optim.RMSprop(model_enhance.parameters(), lr=args.lr, weight_decay=5e-5)

optimizer_critic = torch.optim.Adam(model_critic.parameters(), lr=args.lr, betas=(0.5,0.999))    # update WGAN-GP optimizer, same as original paper
optimizer_enhance = torch.optim.Adam(model_enhance.parameters(), lr=args.lr, betas=(0.5,0.999))

lr_lambda = lambda epoch: min(1.0, (epoch + 1)/5.)  # noqa
lrsch = torch.optim.lr_scheduler.LambdaLR(optimizer_enhance, lr_lambda=lr_lambda)

def calculate_gradient_penalty(real_tuple, fake_tuple, critic):
    """
    Calculate the gradient penalty for WGAN-GP.
    Inputs are two tuples of tensors (real and fake), each containing three tensors.
    """
    # Unpack the tuples
    real_images, real_labels, real_embeddings = real_tuple
    fake_images, fake_labels, fake_embeddings, = fake_tuple

    # Generate random factors for interpolation
    eta = torch.FloatTensor(real_images.size(0), 1, 1, 1).uniform_(0, 1).cuda()

    # Interpolate between real and fake for all three components
    interpolated_images = eta * real_images + (1 - eta) * fake_images
    interpolated_embeddings = eta * real_embeddings + (1 - eta) * fake_embeddings
    interpolated_labels = eta * real_labels + (1 - eta) * fake_labels

    # Ensure gradients can be computed
    interpolated_images = Variable(interpolated_images, requires_grad=True)
    interpolated_embeddings = Variable(interpolated_embeddings, requires_grad=True)
    interpolated_labels = Variable(interpolated_labels, requires_grad=True)

    # Calculate critic output for interpolated inputs
    prob_interpolated = critic((interpolated_images, interpolated_labels, interpolated_embeddings))

    # Compute gradients of critic output with respect to interpolated inputs
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=[interpolated_images, interpolated_embeddings, interpolated_labels],
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
    )

    # Flatten and combine gradients for all components
    gradients_images = gradients[0].view(gradients[0].size(0), -1)
    gradients_embeddings = gradients[1].view(gradients[1].size(0), -1)
    gradients_labels = gradients[2].view(gradients[2].size(0), -1)
    combined_gradients = torch.cat([gradients_images, gradients_embeddings, gradients_labels], dim=1)

    # Compute gradient penalty
    grad_penalty = ((combined_gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # lambda_term = 10
    return grad_penalty

def wgan_train(x:torch.Tensor,patch_id,labels,
               classifier,enhancement,enhancement_optimizer,
               critic,critic_optimizer,iter_num):
    ''' Train conditional Wasserstein GAN for one step '''
    one = torch.FloatTensor([1]).cuda()
    mone = -1*one

    # random_seed = torch.rand(x.shape[0],1).cuda()
    # limit the gradient, clamp parameters to a cube

    # WGAN-clip algorithm
    # for p in critic.parameters():
    #     p.data.clamp_(-0.01,0.01)

    x1 = cvd_process(x)
    y1 = classifier(x1)
    y1_all = topk_sample(x,patch_id,y1,labels)
    real_validity = critic(y1_all)
    # critic_loss = torch.mean(real_validity)
    # critic_loss.backward()

    xe = enhancement(x)
    x2 = cvd_process(xe)
    y2 = classifier(x2)
    y2_all = sample(xe,patch_id,y2,labels)  # 0225 update: use xe rather that x
    
    fake_validity = critic(y2_all)

    # WGAN GP algorithm  
    gradient_penalty = calculate_gradient_penalty(y1_all, y2_all, critic)
    # train critic function
    set_requires_grad(critic,True)
    set_requires_grad(enhancement,False)
    critic_loss = -torch.mean(real_validity)+torch.mean(fake_validity)+gradient_penalty
    critic_loss.backward()

    num_accumlate = max(1,128//args.batchsize)
    if iter_num % num_accumlate == 0:
        critic_optimizer.step()
        critic_optimizer.zero_grad()
    # train enhancement model, equals to generator
    iter_lambda = max(iter_num-10000,0) # control ssim
    if iter_num % args.n_critic == 0:
        set_requires_grad(critic,False)
        set_requires_grad(enhancement,True)
        enhancement_optimizer.zero_grad()
        xe = enhancement(x)
        x2 = cvd_process(xe)
        y2 = classifier(x2)
        y2_all = sample(xe,patch_id,y2,labels,k=args.batchsize) # 0225 update: use xe rather that x
        fake_validity = critic(y2_all)
        # enhancement_loss = -torch.mean(fake_validity) + criterion_L1(xe,x)
        enhancement_loss = -torch.mean(fake_validity) + iter_lambda*criterion_ssim(xe,x)    #criterion_L1(xe,x) + criterion(y2_all[2],labels)    # 0303 update: add infoNCE loss
        enhancement_loss.backward()
        enhancement_optimizer.step()
    return y2,critic_loss

def sample(img,patch_id,embedding_patch,gts,k=11):
    ori_shape = img.shape
    # extract embeddings at patch_id
    batch_index = torch.arange(ori_shape[0],dtype=torch.long)   # 配合第二维度索引使用
    embedding_patch = embedding_patch[batch_index,patch_id]
    if ori_shape[0]<=k:
        return img,patch_id,embedding_patch
    random_start = np.random.randint(0,ori_shape[0]-k)
    return img[random_start:random_start+k,...],patch_id[random_start:random_start+k,...],embedding_patch[random_start:random_start+k,...]

def topk_sample(img,patch_id,embedding_patch,gts,top_k=11):
    '''calculate the correctness of embeddings, take top results on each class'''
    ori_shape = img.shape
    batch_index = torch.arange(ori_shape[0],dtype=torch.long)   # 配合第二维度索引使用
    embedding_patch = embedding_patch[batch_index,patch_id]
    if ori_shape[0]<=top_k:
        return img,patch_id,embedding_patch
    top_k_per_cls = top_k//11
    loss_batch,_ = criterion.infoNCELoss(embedding_patch,gts)
    # sample ones with minimun loss for each class
    color_types = set(gts)
    # sort the samples' loss and name by loss value (ascending)
    loss_batch,indices = torch.sort(loss_batch.flatten())
    indices = indices.cpu().numpy()
    
    # choose top colors for each type
    gts_np = np.array(gts, dtype=str)
    out_indices = []
    for i, color in enumerate(color_types):
        cls_indices = indices[gts_np[indices] == color]
        cls_needed = min(top_k_per_cls,len(cls_indices))
        selected = cls_indices[:cls_needed]
        out_indices.extend(selected.tolist())

    out_indices = torch.tensor(out_indices, device=img.device)   # original indices for a given sorted index
    return img[out_indices,...],patch_id[out_indices,...],embedding_patch[out_indices,...]

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train(trainloader, model, criterion, optimizer, lrsch, logger, args, phase='train'):
    logger.update_step()
    model.eval()
    model_enhance.train()
    model_critic.train()
    loss_logger = 0.
    label_list = []
    pred_list  = []
    iter_num = 0
    for img, ci_patch, img_ori, _,patch_color_name, patch_id in tqdm(trainloader,ascii=True,ncols=60):
        img = img.cuda()
        img_ori = img_ori.cuda()
        patch_id = patch_id.cuda()
        outs,critic_loss = wgan_train(img_ori,patch_id,patch_color_name,
                                      model,model_enhance,optimizer_enhance,
                                      model_critic,optimizer_critic,iter_num)

        batch_index = torch.arange(len(outs),dtype=torch.long)   # 配合第二维度索引使用
        outs = outs[batch_index,patch_id] # 取出目标位置的颜色embedding
        # print('Model use:',time.time()-st_time) # debug
        # print('Loss Func. use:',time.time()-st_time)    # debug
        pred,label = criterion.classification(outs,patch_color_name)
        # print('Classification use:',time.time()-st_time)    # debug
        label_list.extend(label.cpu().detach().tolist())
        pred_list.extend(pred.cpu().detach().tolist())
        loss_logger += critic_loss.item()
        iter_num+=1
    lrsch.step()

    loss_logger /= len(trainloader)
    print("Train Optim loss:",loss_logger)
    log_metric('Train Optim',logger,loss_logger,label_list,pred_list)

def validate(testloader, model, criterion, optimizer, lrsch, logger, args):
    model_enhance.eval()
    model.eval()
    loss_logger = 0.
    label_list = []
    pred_list  = []
    for img, ci_patch, img_ori, _,patch_color_name, patch_id in tqdm(testloader,ascii=True,ncols=60):

        with torch.no_grad():
            img_ori = img_ori.cuda()
            img_t = model_enhance(img_ori)
            img = cvd_process(img_t)
            outs = model(img.cuda()) 
        # ci_rgb = ci_rgb.cuda()
        # img_target = img_target.cuda()
        # print("label:",label)
        batch_index = torch.arange(len(outs),dtype=torch.long)   # 配合第二维度索引使用
        outs = outs[batch_index,patch_id] # 取出目标位置的颜色embedding
        loss_batch = criterion(outs,patch_color_name)
        loss_logger += loss_batch.item()    # 显示全部loss
        pred,label = criterion.classification(outs,patch_color_name)
        label_list.extend(label.cpu().detach().tolist())
        pred_list.extend(pred.cpu().detach().tolist())
    loss_logger /= len(testloader)
    print("Val Optim loss:",loss_logger)
    acc = log_metric('Val Optim', logger,loss_logger,label_list,pred_list)
    return acc, model_enhance.state_dict()

def log_metric(prefix, logger, loss, target, pred):
    cls_report = classification_report(target, pred, output_dict=True, zero_division=0)
    acc = accuracy_score(target, pred)
    print(cls_report)   # all class information
    # auc = roc_auc_score(target, prob)
    logger.log_scalar(prefix+'/loss',loss,print=False)
    # logger.log_scalar(prefix+'/AUC',auc,print=True)
    logger.log_scalar(prefix+'/'+'Acc', acc, print= True)
    logger.log_scalar(prefix+'/'+'Pos_precision', cls_report['weighted avg']['precision'], print=False)
    # logger.log_scalar(prefix+'/'+'Neg_precision', cls_report['0']['precision'], print= True)
    logger.log_scalar(prefix+'/'+'Pos_recall', cls_report['weighted avg']['recall'], print=False)
    # logger.log_scalar(prefix+'/'+'Neg_recall', cls_report['0']['recall'], print= True)
    logger.log_scalar(prefix+'/'+'Pos_F1', cls_report['weighted avg']['f1-score'], print=False)
    logger.log_scalar(prefix+'/loss',loss,print=False)
    return acc   # 越大越好

def sample_enhancement(model,inferenceloader,epoch,args):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    '''
    model.eval()
    cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # cvd模拟器，保证在同一个设备上进行全部运算
    temploader =  CVDImageNetRand(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)   # 只利用其中的颜色命名模块
    image_sample = Image.open('apple.png').convert('RGB')
    # image_sample_big = np.array(image_sample)/255.   # 缓存大图
    image_sample = image_sample.resize((args.size,args.size))
    image_sample = np.array(image_sample)
    patch_names = []
    for patch_y_i in range(args.size//args.patch):
        for patch_x_i in range(args.size//args.patch):
            y_end = patch_y_i*args.patch+args.patch
            x_end = patch_x_i*args.patch+args.patch
            single_patch = image_sample[patch_y_i*16:y_end,patch_x_i*16:x_end,:]
            # calculate color names
            patch_rgb = np.mean(single_patch,axis=(0,1))
            patch_color_name,_ = temploader.classify_color(torch.tensor(patch_rgb)) # classify_color接收tensor输入
            patch_names.append(patch_color_name)

    image_sample = torch.tensor(image_sample).permute(2,0,1).unsqueeze(0)/255.
    image_sample = image_sample.cuda()
    img_out = image_sample.clone()

    # 一次性生成方案：
    model_enhance.eval()
    img_t = model_enhance(img_out)    # 采用cnn变换改变色彩
    img_cvd = cvd_process(img_t)
    outs = model(img_cvd)
    outs = outs[0]  # 去掉batch维度

    ori_out_array = img_out.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    img_out_array = img_t.clone()
    img_out_array = img_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()

    img_diff = (img_out_array - ori_out_array)*10.0
    img_all_array = np.clip(np.hstack([ori_out_array,img_out_array,img_diff]),0.0,1.0)
    plt.imshow(img_all_array)
    plt.savefig('./run/'+f'sample_{args.prefix}_e{epoch}.png')

testing = validate
best_score = 0

if args.test == True:
    finaltestset =  CVDImageNetRand(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = True,num_workers=4)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    model_enhance.load_state_dict(torch.load(pth_optim_location, map_location='cpu'))
    # sample_enhancement(model,None,-1,args)  # test optimization
    testing(valloader,model,criterion,None,lrsch,logger,args)    # test performance on dataset
else:
    model.load_state_dict(torch.load(pth_location))
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        # if i==0:
        #     sample_enhancement(model,None,i,args) # debug
        # train(trainloader, model,criterion,optimizer,lrsch,logger,args,'train')
        train(trainloader, model,criterion,optimizer_enhance,lrsch,logger,args)
        score_optim, model_optim_save = validate(valloader,model,criterion,None,lrsch,logger,args)
        sample_enhancement(model,None,i,args)
        if score_optim > best_score:
            torch.save(model_optim_save, pth_optim_location)
            best_score = score_optim