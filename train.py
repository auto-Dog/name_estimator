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
from dataloaders.pic_data import ImgDataset
from dataloaders.CVDcifar import CVDcifar,CVDImageNet,CVDPlace
from network import ViT,colorLoss
from utils.cvdObserver import cvdSimulateNet
from utils.conditionP import conditionP
from utils.utility import patch_split,patch_compose

# hugface官方实现
# from transformers import ViTImageProcessor, ViTForImageClassification
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits

dataset = 'local'
num_classes = 6

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=4)
parser.add_argument('--size',type=int, default=32)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/work/mingjundu/imagenet100k/')
parser.add_argument("--cvd", type=str, default='deutan')
# C-Glow parameters
parser.add_argument("--x_size", type=str, default="(3,32,32)")
parser.add_argument("--y_size", type=str, default="(3,32,32)")
parser.add_argument("--x_hidden_channels", type=int, default=128)
parser.add_argument("--x_hidden_size", type=int, default=32)
parser.add_argument("--y_hidden_channels", type=int, default=256)
parser.add_argument("-K", "--flow_depth", type=int, default=8)
parser.add_argument("-L", "--num_levels", type=int, default=3)
parser.add_argument("--learn_top", type=bool, default=False)
parser.add_argument("--x_bins", type=float, default=256.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=256.0)
parser.add_argument("--prefix", type=str, default='K32_b64')
args = parser.parse_args()

args.x_size = eval(args.x_size)
args.y_size = eval(args.y_size)
print(args) # show all parameters
### write model configs here
save_root = './run'
pth_location = './Models/model_'+args.prefix+'.pth'
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
train_val_percent = 0.8
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# skf = StratifiedGroupKFold(n_splits=n_splits)

# trainset = CVDcifar('./',train=True,download=True,patch_size=args.patch,img_size=args.size,cvd=args.cvd)
# testset = CVDcifar('./',train=False,download=True,patch_size=args.patch,img_size=args.size,cvd=args.cvd)
trainset = CVDImageNet(args.dataset,split='imagenet_subtrain',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
valset = CVDImageNet(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
# trainset = CVDPlace('/work/mingjundu/place_dataset/places365_standard/',split='train',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
# valset = CVDPlace('/work/mingjundu/place_dataset/places365_standard/',split='val',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
# inferenceset = CIFAR10('./',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),]))

# train_size = int(len(trainset) * train_val_percent)   # not suitable for ImageNet subset
# val_size = len(trainset) - train_size
# trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize*4,shuffle = True)
# testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)
# inferenceloader = torch.utils.data.DataLoader(inferenceset,batch_size=args.batchsize,shuffle = False,)
# trainval_loader = {'train' : trainloader, 'valid' : validloader}

# model = ViT('ColorViT', pretrained=False,image_size=32,patches=4,num_layers=6,num_heads=6,num_classes=4*4*3)
model = ViT(args)
# model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()

criterion = colorLoss()
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

# Update 11.15
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=5e-5)
lr_lambda = lambda epoch: min(1.0, (epoch + 1)/5.)  # noqa
lrsch = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=0.3)
logger.auto_backup('./')

def add_input_noise(input:torch.Tensor,channel_max=(0.6,0.6,0.025),bins=256):
    ori_noise = torch.rand_like(input)
    ori_noise[:,0,:,:] *= channel_max[0]
    ori_noise[:,1,:,:] *= channel_max[1]
    ori_noise[:,2,:,:] *= channel_max[2]
    ori_noise /= bins
    out = input+ori_noise
    return out

def train(trainloader, model, criterion, optimizer, lrsch, logger, args, epoch):
    model.train()
    loss_logger = 0.
    logger.update_step()
    for img, ci_patch, img_target, _,patch_color_name, _  in tqdm(trainloader,ascii=True,ncols=60):
        optimizer.zero_grad()
        img = img.cuda()
        ci_patch = ci_patch.cuda()
        outs = model(add_input_noise(img,bins=args.x_bins),
                         add_input_noise(ci_patch,bins=args.y_bins))
        loss_batch = criterion(outs,patch_color_name)
        # img_target = img_target.cuda()
        # print("opt tensor:",out)
        # ci_rgb = ci_rgb.cuda()

        # if epoch>30:
        #     # 冻结部分层
        #     for name, param in model.named_parameters():
        #         if ("transformer" in name):
        #             param.requires_grad = False
        # loss_batch = criterion(outs,img_target)
        loss_batch.backward()
        loss_logger += loss_batch.item()    # 显示全部loss
        optimizer.step()
    lrsch.step()

    loss_logger /= len(trainloader)
    print("Train loss:",loss_logger)
    log_metric('Train', logger,loss_logger)
    if not (logger.global_step % args.save_interval):
        logger.save(model,optimizer, lrsch, criterion)
        
def validate(testloader, model, criterion, optimizer, lrsch, logger, args):
    model.eval()
    loss_logger = 0.

    for img, ci_patch, img_target, _,patch_color_name, _ in tqdm(testloader,ascii=True,ncols=60):
        with torch.no_grad():
            outs,nll = model(img.cuda(),ci_patch.cuda()) 
        # ci_rgb = ci_rgb.cuda()
        # img_target = img_target.cuda()
        # print("label:",label)
        loss_batch = criterion(outs,patch_color_name)
        loss_logger += loss_batch.item()    # 显示全部loss

    loss_logger /= len(testloader)
    print("Val loss:",loss_logger)

    acc = log_metric('Val', logger,loss_logger)

    return acc, model.state_dict()

# def sample_enhancement(model,inferenceloader,epoch,args):
#     ''' 根据给定的图片，进行颜色优化

#     目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

#     '''
#     model.eval()
#     sample_num = 30 # 目标输出采样数
#     cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # 保证在同一个设备上进行全部运算
#     # for img,_ in inferenceloader:
#     #     img = img.cuda()
#     #     img_cvd = cvd_process(img)
#     #     img_cvd:torch.Tensor = img_cvd[0,...].unsqueeze(0)  # shape C,H,W
#     #     img_t:torch.Tensor = img[0,...].unsqueeze(0)        # shape C,H,W
#     #     break   # 只要第一张
#     image_sample = Image.open('flowers.PNG').convert('RGB')
#     image_sample_big = np.array(image_sample)/255.   # 缓存大图
#     image_sample = image_sample.resize((args.size,args.size))
#     image_sample = torch.tensor(np.array(image_sample)).permute(2,0,1).unsqueeze(0)/255.
#     image_sample = image_sample.cuda()
#     # img_cvd = cvd_process(image_sample)
#     # img_cvd:torch.Tensor = img_cvd[0,...].unsqueeze(0)  # shape C,H,W
#     img_t:torch.Tensor = image_sample[0,...].unsqueeze(0)        # shape C,H,W

#     img_out = img_t.clone()
#     img_out_delta = img_out.repeat(sample_num,1,1,1).contiguous()  # 保持跟采样数一致
#     img_bias = torch.rand(sample_num, 3, 1, 1).cuda() * 0.2 - 0.1  # 生成 -0.1 到 0.1 之间的随机偏移量
#     img_out_delta = img_out_delta + img_bias  # 将偏移量加到 img_out_delta 上，维度自动广播
#     img_out_delta[img_out_delta<0] = 0.
#     # inference_criterion = nn.MSELoss()
#     img_t.requires_grad = True
#     # inference_optimizer = torch.optim.SGD(params=[img_t],lr=3e-3)   # 对输入图像进行梯度下降
#     # inference_optimizer = torch.optim.SGD(params=[img_t],lr=3e-3,momentum=0.3) # 对输入图像进行梯度下降
#     inference_optimizer = torch.optim.Adam(params=[img_t],lr=1e-2)   # 对输入图像进行梯度下降
#     # for iter in range(100):
#     #     inference_optimizer.zero_grad()
#     #     img_cvd = cvd_process(img_t)
#     #     img_cvd_batch = img_cvd.repeat(sample_num,1,1,1).contiguous()  # 保持跟采样数一致
#     #     out_z,loss = model(img_cvd_batch,img_out_delta)  # 相当于Σ-log p(img_ori|img_cvd(t))
#     #     loss = torch.mean(loss)
#     #     # loss = inference_criterion(out,img_out)   
#     #     loss.backward()
#     #     inference_optimizer.step()
#     #     if iter%10 == 0:
#     #         print(f'Mean Absolute grad: {torch.mean(torch.abs(img_t.grad))}, nll:{loss.item()}')
#     img_cvd = cvd_process(img_t)
#     out_z,loss = model(img_cvd,img_t)  # temp debug
#     out,nll = model(img_cvd,out_z,reverse=True)   # 存在问题，逆向生成大概率上色不准

#     ori_out_array = img_out.squeeze(0).permute(1,2,0).cpu().detach().numpy()

#     recolor_out_array = out.clone()
#     recolor_out_array = recolor_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()
#     # recolor_out_array_big = apply_color_transfer(ori_out_array,recolor_out_array,image_sample_big)  # 将小图变换应用到大图

#     img_out_array = img_t.clone()
#     img_out_array = img_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()
#     img_out_array_big = apply_color_transfer(ori_out_array,img_out_array,image_sample_big)

#     img_diff = (img_out_array - ori_out_array)*10.0
#     img_diff_big = (img_out_array_big - image_sample_big)*10.0
#     img_all_array = np.clip(np.hstack([ori_out_array,recolor_out_array,img_out_array,img_diff]),0.0,1.0)
#     img_all_array_big = np.clip(np.hstack([image_sample_big,img_out_array_big,img_diff_big]),0.0,1.0)
#     plt.imshow(img_all_array)
#     plt.savefig('./run/'+f'sample_{args.prefix}_e{epoch}.png')
#     plt.cla()
#     plt.imshow(img_all_array_big)
#     plt.savefig('./run/'+f'highres_sample_{args.prefix}_e{epoch}.png')

def log_metric(prefix, logger, loss):
    logger.log_scalar(prefix+'/loss',loss,print=False)
    return 1/loss   # 越大越好

testing = validate
auc = 0

if args.test == True:
    # finaltestset = CVDcifar('./',train=False,download=True)
    # finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = False,num_workers=8)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    # sample_enhancement(model,None,-1,args)
    # testing(finaltestloader,model,criterion,optimizer,lrsch,logger,args)
else:
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        # if i==0:
        #     sample_enhancement(model,None,i,args) # debug
        train(trainloader, model,criterion,optimizer,lrsch,logger,args,i)
        score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args)
        # sample_enhancement(model,None,i,args)
        if score > auc:
            auc = score
            torch.save(model_save, pth_location)
