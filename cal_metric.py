import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import cv2

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

if __name__ == '__main__':
    from other_methods import AndriodDaltonizer
    from colour.blindness import matrix_cvd_Machado2009
    from PIL import Image, ImageDraw
    from torchvision import transforms

    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ]
    )
    img = Image.open('apple.png').convert('RGB')
    img = np.array(img)/255.
    basic_enhance = AndriodDaltonizer('Deuteranomaly')
    img_enhance = basic_enhance.forward(img)
    img_enhance_cvd = cvd_simulate(img_enhance)
    metric1 = LCD_metric(transform1(img_enhance).unsqueeze(0),transform1(img_enhance_cvd).unsqueeze(0))
    metric2 = CD_metric(transform1(img_enhance).unsqueeze(0),transform1(img).unsqueeze(0))
    print(metric1,metric2)