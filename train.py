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
parser.add_argument("--x_bins", type=float, default=256.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=256.0)
parser.add_argument("--prefix", type=str, default='vit_cn1')
args = parser.parse_args()

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

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True,num_workers=4)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True,num_workers=4)
# testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)
# inferenceloader = torch.utils.data.DataLoader(inferenceset,batch_size=args.batchsize,shuffle = False,)
# trainval_loader = {'train' : trainloader, 'valid' : validloader}

model = ViT('ColorViT', pretrained=False,image_size=args.size,patches=args.patch,num_layers=6,num_heads=6,num_classes = 1000)
model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()

criterion = colorLoss(args.tau)

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

import time
def train(trainloader, model, criterion, optimizer, lrsch, logger, args, epoch):
    model.train()
    loss_logger = 0.
    logger.update_step()
    label_list = []
    pred_list  = []
    for img, ci_patch, img_target, _,patch_color_name, _  in tqdm(trainloader,ascii=True,ncols=60):
        optimizer.zero_grad()
        img = img.cuda()
        ci_patch = ci_patch.cuda()
        # st_time = time.time()   # debug
        # print('Timer now')  # debug
        outs = model(add_input_noise(img,bins=args.x_bins),
                         add_input_noise(ci_patch,bins=args.y_bins))
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
        
def validate(testloader, model, criterion, optimizer, lrsch, logger, args):
    model.eval()
    loss_logger = 0.
    label_list = []
    pred_list  = []
    for img, ci_patch, img_target, _,patch_color_name, _ in tqdm(testloader,ascii=True,ncols=60):
        with torch.no_grad():
            outs = model(img.cuda(),ci_patch.cuda()) 
        # ci_rgb = ci_rgb.cuda()
        # img_target = img_target.cuda()
        # print("label:",label)
        loss_batch = criterion(outs,patch_color_name)
        loss_logger += loss_batch.item()    # 显示全部loss
        pred,label = criterion.classification(outs,patch_color_name)
        label_list.extend(label.cpu().detach().tolist())
        pred_list.extend(pred.cpu().detach().tolist())
    loss_logger /= len(testloader)
    print("Val loss:",loss_logger)

    acc = log_metric('Val', logger,loss_logger,label_list,pred_list)

    return acc, model.state_dict()

def sample_enhancement(model,inferenceloader,epoch,args):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    '''
    model.eval()
    cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # 保证在同一个设备上进行全部运算
    temploader =  CVDImageNetRand(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
    image_sample = Image.open('flowers.PNG').convert('RGB')
    image_sample_big = np.array(image_sample)/255.   # 缓存大图
    image_sample = image_sample.resize((args.size,args.size))
    image_sample = np.array(image_sample)
    patch_names = []
    for patch_y_i in range(args.size//args.patch):
        for patch_x_i in range(args.size//args.patch):
            y_end = patch_y_i*args.patch+args.patch
            x_end = patch_x_i*args.patch+args.patch
            single_patch = image_sample[patch_y_i*16:y_end,patch_x_i*16:x_end,:]*255
            # calculate color names
            patch_rgb = np.mean(single_patch,axis=(0,1))
            patch_color_name,_ = temploader.classify_color(patch_rgb)
            patch_names.append(patch_color_name)

    image_sample = torch.tensor(image_sample).permute(2,0,1).unsqueeze(0)/255.
    image_sample = image_sample.cuda()
    # img_cvd = cvd_process(image_sample)
    # img_cvd:torch.Tensor = img_cvd[0,...].unsqueeze(0)  # shape C,H,W
    img_t:torch.Tensor = image_sample[0,...].unsqueeze(0)        # shape C,H,W
    img_out = img_t.clone()
    inference_criterion = colorLoss(tau=0.2)
    img_t.requires_grad = True
    # inference_optimizer = torch.optim.SGD(params=[img_t],lr=3e-3)   # 对输入图像进行梯度下降
    # inference_optimizer = torch.optim.SGD(params=[img_t],lr=3e-3,momentum=0.3) # 对输入图像进行梯度下降
    inference_optimizer = torch.optim.Adam(params=[img_t],lr=1e-2)   # 对输入图像进行梯度下降
    for iter in range(100):
        inference_optimizer.zero_grad()
        img_cvd = cvd_process(img_t)
        outs = model(img_cvd)
        loss = inference_criterion(outs,patch_names)
        # loss = inference_criterion(out,img_out)   
        loss.backward()
        inference_optimizer.step()
        if iter%10 == 0:
            print(f'Mean Absolute grad: {torch.mean(torch.abs(img_t.grad))}, loss:{loss.item()}')

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
    finaltestset =  CVDImageNetRand(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = True,num_workers=4)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    # sample_enhancement(model,None,-1,args)
    testing(finaltestloader,model,criterion,optimizer,lrsch,logger,args)
else:
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        # if i==0:
        #     sample_enhancement(model,None,i,args) # debug
        train(trainloader, model,criterion,optimizer,lrsch,logger,args,i)
        score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args)
        # sample_enhancement(model,None,i,args)
        if score > best_score:
            best_score = score
            torch.save(model_save, pth_location)
