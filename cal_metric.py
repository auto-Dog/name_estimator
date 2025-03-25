import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse
import random
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1896)
from dataloaders.CVDDS import CVDImageNetRand
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def RGB_to_sRGB(RGB):
    '''RGB to sRGB, value 0.0-1.0(NOT 0-255)'''
    sRGB = np.ones_like(RGB)
    mask = RGB > 0.0031308
    sRGB[~mask] = 12.92*RGB[~mask]
    sRGB[mask] = 1.055 * RGB[mask]**(1 / 2.4) - 0.055
    return sRGB

def sRGB_to_RGB(srgb_img):
    ''' Gamma correction of sRGB photo from camera  
        value 0.0-1.0(NOT 0-255)
     Ref: http://brucelindbloom.com/Eqn_RGB_to_XYZ.html 
    '''
    RGB = np.ones_like(srgb_img)
    mask = srgb_img < 0.04045
    RGB[mask] = srgb_img[mask]/12.92
    RGB[~mask] = ((srgb_img[~mask]+0.055)/1.055)**2.4
    return RGB

def sRGB_to_Lab(rgb1):
    rgb1 = rgb1.numpy()
    rgb_batch = np.float32(rgb1)
    # 重新调整输入数组的形状，使其成为 (n, 1, 3)，符合OpenCV的要求
    ori_shape = rgb_batch.shape
    rgb_batch = rgb_batch.reshape(-1, 1, 3)
    # 使用OpenCV的cvtColor函数转换RGB到Lab
    lab_batch = cv2.cvtColor(rgb_batch, cv2.COLOR_RGB2Lab)
    return torch.tensor(lab_batch.reshape(ori_shape)).cuda()  # 还原形状

def cvd_simulate(img:np.ndarray,cvd_type='Deuteranomaly'):
    '''Given a 0-1 RGB image, simulate its appearance in CVD eyes to NT'''
    H_mat = matrix_cvd_Machado2009(cvd_type,1.)
    h,w,d = img.shape
    im1 = img.reshape(-1,d)
    im1 = sRGB_to_RGB(im1)
    im_dst1 = im1 @ H_mat.T
    im_dst1 = RGB_to_sRGB(im_dst1)
    # im_dst1 = cvd_simulation_tritran(im1)
    im_dst = im_dst1.reshape(h,w,d)
    im_dst[im_dst>1] = 1.
    im_dst[im_dst<0] = 0.
    return im_dst

def select_filter(input, win, stride=1):
    r""" Windowly select input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            if stride == 1:
                out = conv(out, weight=win.transpose(2 + i, -1), stride=stride, padding=0, groups=C)
            else:
                S = [(stride, 1), (1, stride)]
                out = conv(out, weight=win.transpose(2 + i, -1), stride=S[i], padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def _contrast(X:torch.Tensor, Y:torch.Tensor, win:torch.Tensor=None):
    r""" Calculate contrast index for X and Y
    Args:
        X (torch.Tensor): images in CIE La*b* space
        Y (torch.Tensor): images in CIE La*b* space
        win (torch.Tensor): 1-D gauss kernel. Actually we donot set window here, but generate 11 one-hot windows win_size=11, win_sigma=1.

        return (torch.Tensor): ssim results.
    """
    # win = win.to(X.device, dtype=X.dtype)
    #
    # X = X[:, 1:, :, :]
    # Y = Y[:, 1:, :, :]

    # mu1 = gaussian_filter(X, win)  # 即mean
    # mu2 = gaussian_filter(Y, win)
    mu1 = X[:, :, 5:-5, 5:-5]
    mu2 = Y[:, :, 5:-5, 5:-5]

    win_list = []
    for i in range(11):
        win_list.append(torch.zeros([1, 1, 1, 11]).to(X.device, dtype=X.dtype)) 
        win_list[i][0, 0, 0, i] = 1 # 选择窗内的一个元素, 横向窗
    x_list = []
    dis = 0
    for win1 in win_list:
        win1 = win1.repeat([X.shape[1]] + [1] * (len(X.shape) - 1)).to(X.device, dtype=X.dtype)
        x = select_filter(X, win1)
        y = select_filter(Y, win1)
        x_list.append(x)
        pdist = torch.nn.PairwiseDistance(p=2)
        # dis1 = pdist(x.permute(0, 2, 3, 1), mu1.permute(0, 2, 3, 1))
        # dis2 = pdist(y.permute(0, 2, 3, 1), mu2.permute(0, 2, 3, 1))
        dis1 = pdist(x, mu1)
        dis2 = pdist(y, mu2)
        # dis += torch.relu(dis1 - dis2)
        dis += torch.abs(dis1 - dis2)
        # dis += (dis1 - dis2).pow(2)
    dis = dis / 11
    return dis.mean()

def LCD_metric(X, Y, size_average=True):
    ''' Calculate the LCD metric on batched images, enhanced image and CVD sim. on enhanced image '''
    X = sRGB_to_Lab(((X + 1) / 2.0).clamp(0, 1))
    Y = sRGB_to_Lab(((Y + 1) / 2.0).clamp(0, 1))
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    cpr = _contrast(X, Y)
    if size_average:
        return cpr.mean()
    else:
        return cpr.mean(1)

def CD_metric(X,Y):
    ''' Calculate the CD metric on batched images, enhanced image and original image '''
    X = sRGB_to_Lab(((X + 1) / 2.0).clamp(0, 1))
    Y = sRGB_to_Lab(((Y + 1) / 2.0).clamp(0, 1))
    X[:,0,:,:] *= 0 # L通道置零
    Y[:,0,:,:] *= 0
    pdist = torch.nn.PairwiseDistance(p=2)
    dis = pdist(X,Y)
    return dis.mean()

# def acc_metric(X,Y):
#     ''' X: test image, Y: reference image'''
#     model.eval()
#     with torch.no_grad():
#         X = cvd_process(X.cuda())
#         outs_X = model(X)        
#         Y = cvd_process(Y.cuda())
#         outs_X = model(Y)
#     # print('Out shape:',outs.shape)  # debug
#     outs = outs[0]  # 去掉batch维度
#     # for vit cn5: give all index at once
#     cls_index,_ = criterion.classification(outs,('Red',)*1024)  # the later parameter is used for fill blank, no meaning
#     return cls_index

# def testing(testloader, method, args):
#     model.eval()
#     optim_model.eval()
#     loss_logger = 0.
#     label_list = []
#     pred_list  = []
#     for img, ci_patch, img_ori, _,patch_color_name, patch_id in tqdm(testloader,ascii=True,ncols=60):
#         if phase == 'eval':
#             with torch.no_grad():
#                 outs = model(img.cuda()) 
#         elif phase == 'optim':
#             with torch.no_grad():
#                 img_ori = img_ori.cuda()
#                 img_t = optim_model(img_ori)
#                 img = cvd_process(img_t)
#                 outs = model(img.cuda()) 
#         # ci_rgb = ci_rgb.cuda()
#         # img_target = img_target.cuda()
#         # print("label:",label)
#         batch_index = torch.arange(len(outs),dtype=torch.long)   # 配合第二维度索引使用
#         outs = outs[batch_index,patch_id] # 取出目标位置的颜色embedding
#         loss_batch = criterion(outs,patch_color_name)
#         loss_logger += loss_batch.item()    # 显示全部loss
#         pred,label = criterion.classification(outs,patch_color_name)
#         label_list.extend(label.cpu().detach().tolist())
#         pred_list.extend(pred.cpu().detach().tolist())
#     loss_logger /= len(testloader)
#     if phase == 'eval':
#         print("Val loss:",loss_logger)
#         acc = log_metric('Val', logger,loss_logger,label_list,pred_list)
#         return acc, model.state_dict()
#     elif phase == 'optim':
#         print("Val Optim loss:",loss_logger)
#         acc = log_metric('Val Optim', logger,loss_logger,label_list,pred_list)
#         return acc, optim_model.state_dict()
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
if __name__ == '__main__':
    from other_methods import AndriodDaltonizer
    from colour.blindness import matrix_cvd_Machado2009
    from PIL import Image, ImageDraw
    from torchvision import transforms

    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    # if args.test == True:
    #     finaltestset =  CVDImageNetRand(args.dataset,split=args.test_split,patch_size=args.patch,img_size=args.size,cvd=args.cvd)
    #     finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = True,num_workers=4)
    #     model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    #     filtermodel.load_state_dict(torch.load(pth_optim_location, map_location='cpu'))
    #     # sample_enhancement(model,None,-1,args)  # test optimization
    #     testing(finaltestloader,method='WGAN',args)    # test performance on dataset
    img = Image.open("C:\\Users\\alphadu\\OneDrive\\CIE_CURVE\\CVD_simulation\\test_img\\test2.PNG").convert('RGB').resize((256,256))
    img = np.array(img)/255.
    basic_enhance = AndriodDaltonizer('Deuteranomaly')
    img_enhance = basic_enhance.forward(img)
    # img_enhance = Image.open("C:\\Users\\alphadu\\OneDrive\\CIE_CURVE\\CVD_simulation\\test_img\\test2_ours.PNG").convert('RGB').resize((256,256))
    # img_enhance = np.array(img_enhance)/255.
    img_enhance_cvd = cvd_simulate(img_enhance)
    metric1 = LCD_metric(transform1(img_enhance).unsqueeze(0),transform1(img_enhance_cvd).unsqueeze(0))
    metric2 = CD_metric(transform1(img_enhance).unsqueeze(0),transform1(img).unsqueeze(0))
    print(metric1,metric2)
