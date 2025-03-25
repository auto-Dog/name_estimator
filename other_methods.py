import numpy as np

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

def im_dot(H_mat,im):
    '''input: h*w*3, 0-1. np array, with Gamma recovering'''
    h,w,d = im.shape
    im1 = im.reshape(-1,d)
    im1 = sRGB_to_RGB(im1)
    im_dst1 = im1 @ H_mat.T
    # im_dst1 = cvd_simulation_tritran(im1)
    im_dst1 = RGB_to_sRGB(im_dst1)
    im_dst = im_dst1.reshape(h,w,d)
    im_dst[im_dst>1] = 1.
    im_dst[im_dst<0] = 0.
    return im_dst

 # 以deuteranopia为例
class AndriodDaltonizer:   # cpp实现中所有矩阵都是x^TA, 我们需要改成A^Tx, 源代码中所有矩阵需要转置
    def __init__(self,mType,mMode='Correction',mLevel=0.8):
        self.mDirty = True
        self.mType = mType # 'NoneType','Protanomaly', 'Deuteranomaly', 'Tritanomaly'
        self.mMode = mMode   # 'Correction', 'Simulation'
        self.mLevel = mLevel   # 0.0-1.0
        self.mColorTransform = np.identity(4)
        self.update()

    def update(self):
        if self.mType == 'NoneType':
            self.mColorTransform = np.identity(4)
            return

        rgb2xyz = np.array([
            [0.4124, 0.2126, 0.0193, 0],
            [0.3576, 0.7152, 0.1192, 0],
            [0.1805, 0.0722, 0.9505, 0],
            [0, 0, 0, 1]
        ]).T

        xyz2lms = np.array([
            [0.7328, -0.7036, 0.0030, 0],
            [0.4296, 1.6975, 0.0136, 0],
            [-0.1624, 0.0061, 0.9834, 0],
            [0, 0, 0, 1]
        ]).T

        rgb2lms = xyz2lms @ rgb2xyz
        lms2rgb = np.linalg.inv(rgb2lms)
        lms_r = (rgb2lms @ np.array([1, 0, 0, 0]).reshape(4,1)).flatten()[:3]
        lms_b = (rgb2lms @ np.array([0, 0, 1, 0]).reshape(4,1)).flatten()[:3]
        lms_w = (rgb2lms @ np.array([1, 1, 1, 0]).reshape(4,1)).flatten()[:3]

        p0 = np.cross(lms_w, lms_b)
        p1 = np.cross(lms_w, lms_r)

        lms2lmsp = np.array([
            [0, 0, 0, 0],
            [-p0[1] / p0[0], 1, 0, 0],
            [-p0[2] / p0[0], 0, 1, 0],
            [0, 0, 0, 1]
        ]).T

        lms2lmsd = np.array([
            [1, -p0[0] / p0[1], 0, 0],
            [0, 0, 0, 0],
            [0, -p0[2] / p0[1], 1, 0],
            [0, 0, 0, 1]
        ]).T

        lms2lmst = np.array([
            [1, 0, -p1[0] / p1[2], 0],
            [0, 1, -p1[1] / p1[2], 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]).T

        errp = np.array([
            [1, self.mLevel, self.mLevel, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).T

        errd = np.array([
            [1, 0, 0, 0],
            [self.mLevel, 1, self.mLevel, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).T

        errt = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [self.mLevel, self.mLevel, 1, 0],
            [0, 0, 0, 1]
        ]).T

        simulation = np.identity(4)
        correction = np.identity(4)

        if self.mType == 'Protanomaly':
            simulation = lms2lmsp
            if self.mMode == 'Correction':
                correction = errp
        elif self.mType == 'Deuteranomaly':
            simulation = lms2lmsd
            if self.mMode == 'Correction':
                correction = errd
        elif self.mType == 'Tritanomaly':
            simulation = lms2lmst
            if self.mMode == 'Correction':
                correction = errt

        self.mColorTransform = lms2rgb @ (simulation @ rgb2lms + correction @ (rgb2lms - simulation @ rgb2lms))

    def forward(self,image):
        ''' image: np 0.-1. image '''
        self.update()
        out = im_dot(self.mColorTransform[:3,:3],image)
        return out

# daltonizer = AndriodDaltonizer('Deuteranomaly')
# daltonizer.forward(img)
