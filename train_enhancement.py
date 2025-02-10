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
from network import ViT,colorLoss, colorFilter,criticNet
from utils.cvdObserver import cvdSimulateNet
from utils.conditionP import conditionP
from utils.utility import patch_split,patch_compose

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=16)
parser.add_argument('--size',type=int, default=512)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/work/mingjundu/imagenet100k/')
parser.add_argument("--cvd", type=str, default='deutan')
parser.add_argument("--tau", type=float, default=0.3)
parser.add_argument("--n_critic", type=int, default=5)
parser.add_argument("--x_bins", type=float, default=256.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=256.0)
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

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True,num_workers=4)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True,num_workers=4)
model = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()

model_nt = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model_nt = nn.DataParallel(model_nt,device_ids=list(range(torch.cuda.device_count())))
model_nt = model_nt.cuda()

model_critic = criticNet().cuda()
model_critic = nn.DataParallel(model_critic,device_ids=list(range(torch.cuda.device_count())))
model_critic = model_critic.cuda()

model_enhance = colorFilter().cuda()
model_enhance = nn.DataParallel(model_enhance,device_ids=list(range(torch.cuda.device_count())))
model_enhance = model_enhance.cuda()

criterion = colorLoss(args.tau)
criterion_L1 = nn.L1Loss()
optimizer_critic = torch.optim.Adamax(model_critic.parameters(), lr=args.lr, weight_decay=5e-5)
optimizer_enhance = torch.optim.Adamax(model_enhance.parameters(), lr=args.lr, weight_decay=5e-5)
lr_lambda = lambda epoch: min(1.0, (epoch + 1)/5.)  # noqa
lrsch = torch.optim.lr_scheduler.LambdaLR(optimizer_enhance, lr_lambda=lr_lambda)

def wgan_train(classifier1,classifier2,enhancement,enhancement_optimizer,critic,critic_optimizer,x:torch.Tensor,iter_num):
    ''' Train conditional Wasserstein GAN for one step '''
    # one = torch.FloatTensor(1).cuda()
    # mone = -1*one
    y1 = classifier1(x)
    # random_seed = torch.rand(x.shape[0],1).cuda()
    # limit the gradient, clamp parameters to a cube
    for p in critic.parameters():
        p.data.clamp_(-0.01,0.01)
    x2 = enhancement(x)
    x2 = cvd_process(x2)
    y2 = classifier2(x2)
    # train critic function
    critic_optimizer.zero_grad()
    real_validity = critic(y1,x)
    fake_validity = critic(y2,x)
    critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

    critic_loss.backward()
    critic_optimizer.step()
    
    # train enhancement model, equals to generator
    if iter_num % args.n_critic == 0:
        enhancement_optimizer.zero_grad()
        x2 = enhancement(x)
        x2 = cvd_process(x2)
        y2 = classifier2(x2)
        fake_validity = critic(y2,x)
        enhancement_loss = -torch.mean(fake_validity) + criterion_L1(x2,x)
        enhancement_loss.backward()
        enhancement_optimizer.step()
    return y2,critic_loss

def train(trainloader, model, criterion, optimizer, lrsch, logger, args, phase='train'):
    logger.update_step()
    model.eval()
    model_nt.eval()
    model_enhance.train()
    model_critic.train()
    loss_logger = 0.
    label_list = []
    pred_list  = []
    iter_num = 0
    for img, ci_patch, img_ori, _,patch_color_name, patch_id in tqdm(trainloader,ascii=True,ncols=60):
        img = img.cuda()
        img_ori = img_ori.cuda()
        outs,critic_loss = wgan_train(model,model_nt,model_enhance,optimizer_enhance,
                   model_critic,optimizer_critic,img_ori,iter_num)

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
    model_nt.load_state_dict(torch.load(pth_nt_location))
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        # if i==0:
        #     sample_enhancement(model,None,i,args) # debug
        # train(trainloader, model,criterion,optimizer,lrsch,logger,args,'train')
        train(trainloader, model,criterion,optimizer_enhance,lrsch,logger,args)
        score_optim, model_optim_save = validate(valloader,model,criterion,None,lrsch,logger,args)
        # sample_enhancement(model,None,i,args)
        if score_optim > best_score:
            torch.save(model_optim_save, pth_optim_location)
            best_score = score_optim