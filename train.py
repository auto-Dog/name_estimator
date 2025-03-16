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
from network import ViT,colorLoss, colorFilter
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
parser.add_argument('--patch',type=int, default=8)
parser.add_argument('--size',type=int, default=256)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/data/mingjundu/imagenet100k/')
parser.add_argument('--test_split', type=str, default='imagenet_subval')
parser.add_argument("--cvd", type=str, default='deutan')
parser.add_argument("--tau", type=float, default=0.3)
parser.add_argument("--x_bins", type=float, default=128.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=128.0)
parser.add_argument("--prefix", type=str, default='vit_cn5')
parser.add_argument('--from_check_point',type=str,default='')
args = parser.parse_args()

print(args) # show all parameters
### write model configs here
save_root = './run'
pth_location = './Models/model_'+args.prefix+'.pth'
pth_optim_location = './Models/model_'+args.prefix+'_optim_base'+'.pth'
ckp_location = './Models/'+args.from_check_point
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
cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # cvd模拟器
# trainset = CVDPlace('/work/mingjundu/place_dataset/places365_standard/',split='train',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
# valset = CVDPlace('/work/mingjundu/place_dataset/places365_standard/',split='val',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
# inferenceset = CIFAR10('./',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),]))

# train_size = int(len(trainset) * train_val_percent)   # not suitable for ImageNet subset
# val_size = len(trainset) - train_size
# trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True,num_workers=8)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True,num_workers=8)
# testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)
# inferenceloader = torch.utils.data.DataLoader(inferenceset,batch_size=args.batchsize,shuffle = False,)
# trainval_loader = {'train' : trainloader, 'valid' : validloader}

model = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()
filtermodel = colorFilter().cuda()
filtermodel = nn.DataParallel(filtermodel,device_ids=list(range(torch.cuda.device_count())))
filtermodel = filtermodel.cuda()

criterion = colorLoss(args.tau)
criterion2 = nn.MSELoss()
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

# Update 11.15
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=5e-5)
optimizer_optim = torch.optim.Adamax(filtermodel.parameters(), lr=args.lr, weight_decay=5e-5)
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

import time
def train(trainloader, model, criterion, optimizer, lrsch, logger, args, phase='train', optim_model=None):
    if phase=='train':
        model.train()
        logger.update_step()
        loss_logger = 0.
        label_list = []
        pred_list  = []
        for img, ci_patch, img_ori, _,patch_color_name, patch_id  in tqdm(trainloader,ascii=True,ncols=60):
            optimizer.zero_grad()
            img = img.cuda()
            ci_patch = ci_patch.cuda()
            # st_time = time.time()   # debug
            # print('Timer now')  # debug

            outs = model(add_input_noise(img,bins=args.x_bins))
            batch_index = torch.arange(len(outs),dtype=torch.long)   # 配合第二维度索引使用
            outs = outs[batch_index,patch_id] # 取出目标位置的颜色embedding
            # print('Model use:',time.time()-st_time) # debug
            loss_batch = criterion(outs,patch_color_name)
            # print('Loss Func. use:',time.time()-st_time)    # debug
            pred,label = criterion.classification(outs,patch_color_name)
            # print('Classification use:',time.time()-st_time)    # debug
            label_list.extend(label.cpu().detach().tolist())
            pred_list.extend(pred.cpu().detach().tolist())
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
        log_metric('Train',logger,loss_logger,label_list,pred_list)
        if not (logger.global_step % args.save_interval):
            logger.save(model,optimizer, lrsch, criterion)
    
    if phase=='optim':
        model.eval()
        optim_model.train()
        loss_logger = 0.
        label_list = []
        pred_list  = []
        for img, ci_patch, img_ori, _,patch_color_name, patch_id  in tqdm(trainloader,ascii=True,ncols=60):
            optimizer.zero_grad()
            img = img.cuda()
            img_ori = img_ori.cuda()
            img_t = optim_model(img_ori)
            img = cvd_process(img_t)
            outs = model(img)
            batch_index = torch.arange(len(outs),dtype=torch.long)   # 配合第二维度索引使用
            outs = outs[batch_index,patch_id] # 取出目标位置的颜色embedding
            # print('Model use:',time.time()-st_time) # debug
            loss_batch = 5*criterion(outs,patch_color_name)+criterion2(img_t,img_ori)
            # print('Loss Func. use:',time.time()-st_time)    # debug
            pred,label = criterion.classification(outs,patch_color_name)
            # print('Classification use:',time.time()-st_time)    # debug
            label_list.extend(label.cpu().detach().tolist())
            pred_list.extend(pred.cpu().detach().tolist())
            loss_batch.backward()
            loss_logger += loss_batch.item()    # 显示全部loss
            optimizer.step()
        # lrsch.step()

        loss_logger /= len(trainloader)
        print("Train Optim loss:",loss_logger)
        log_metric('Train Optim',logger,loss_logger,label_list,pred_list)

def validate(testloader, model, criterion, optimizer, lrsch, logger, args, phase='eval', optim_model=None):
    model.eval()
    optim_model.eval()
    loss_logger = 0.
    label_list = []
    pred_list  = []
    for img, ci_patch, img_ori, _,patch_color_name, patch_id in tqdm(testloader,ascii=True,ncols=60):
        if phase == 'eval':
            with torch.no_grad():
                outs = model(img.cuda()) 
        elif phase == 'optim':
            with torch.no_grad():
                img_ori = img_ori.cuda()
                img_t = optim_model(img_ori)
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
    if phase == 'eval':
        print("Val loss:",loss_logger)
        acc = log_metric('Val', logger,loss_logger,label_list,pred_list)
        return acc, model.state_dict()
    elif phase == 'optim':
        print("Val Optim loss:",loss_logger)
        acc = log_metric('Val Optim', logger,loss_logger,label_list,pred_list)
        return acc, optim_model.state_dict()
    
def sample_enhancement(model,inferenceloader,epoch,args):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    '''
    model.eval()
    cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # cvd模拟器，保证在同一个设备上进行全部运算
    # temploader =  CVDImageNetRand(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)   # 只利用其中的颜色命名模块
    image_sample = Image.open('apple.png').convert('RGB')
    # image_sample_big = np.array(image_sample)/255.   # 缓存大图
    image_sample = image_sample.resize((args.size,args.size))
    image_sample = np.array(image_sample)
    # patch_names = []
    # for patch_y_i in range(args.size//args.patch):
    #     for patch_x_i in range(args.size//args.patch):
    #         y_end = patch_y_i*args.patch+args.patch
    #         x_end = patch_x_i*args.patch+args.patch
    #         single_patch = image_sample[patch_y_i*16:y_end,patch_x_i*16:x_end,:]
    #         # calculate color names
    #         patch_rgb = np.mean(single_patch,axis=(0,1))
    #         patch_color_name,_ = temploader.classify_color(torch.tensor(patch_rgb)) # classify_color接收tensor输入
    #         patch_names.append(patch_color_name)

    image_sample = torch.tensor(image_sample).permute(2,0,1).unsqueeze(0)/255.
    image_sample = image_sample.cuda()
    img_out = image_sample.clone()

    # 一次性生成方案：
    filtermodel.eval()
    img_t = filtermodel(img_out)    # 采用cnn变换改变色彩
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
    finaltestset =  CVDImageNetRand(args.dataset,split=args.test_split,patch_size=args.patch,img_size=args.size,cvd=args.cvd)
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = True,num_workers=4)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    filtermodel.load_state_dict(torch.load(pth_optim_location, map_location='cpu'))
    # sample_enhancement(model,None,-1,args)  # test optimization
    testing(finaltestloader,model,criterion,optimizer,lrsch,logger,args,'optim',filtermodel)    # test performance on dataset
else:
    if args.from_check_point != '':
        model.load_state_dict(torch.load(ckp_location))
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        # if i==0:
        #     sample_enhancement(model,None,i,args) # debug
        # train(trainloader, model,criterion,optimizer,lrsch,logger,args,'train',filtermodel)
        # score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'eval',filtermodel)
        # if score > best_score:
        #     best_score = score
        #     torch.save(model_save, pth_location)

        # if (i+1)%5 == 0:
        #     train(trainloader, model,criterion,optimizer_optim,lrsch,logger,args,'optim',filtermodel)
        #     score_optim, model_optim_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'optim',filtermodel)
        #     sample_enhancement(model,None,i,args)
        #     if score_optim > score:
        #         torch.save(model_optim_save, pth_optim_location)

        train(trainloader, model,criterion,optimizer_optim,lrsch,logger,args,'optim',filtermodel)
        score_optim, model_optim_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args,'optim',filtermodel)
        sample_enhancement(model,None,i,args)
        if score_optim > best_score:
            best_score = score_optim
            torch.save(model_optim_save, pth_optim_location)

