import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import measure
import math
import os
import datetime
from glob import glob


# Gaussian blur
def GaussianBlur(img):
    dst = cv2.GaussianBlur(img, (21, 21), 0)
    return dst

# Gaussian motion blur
def motion_blur(img):
    degree = 15
    angle = 45
    image = np.array(img)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

# Gaussian noise
def GaussianNoise(img):
    mu = 0.0
    sigma = 0.05
    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise

# salt and pepper noise
def SPnoise(img):
    s_vs_p = 0.2
    amount = 0.02
    noisy_img = np.copy(img)
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1], :] = [255, 255, 255]
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1], :] = [0, 0, 0]
    return noisy_img

# Contrast adjustment-enhancement
def hightContrast(img):
    res1 = np.uint8(np.clip((cv2.add(1.5 * img, 0)), 0, 255))
    return res1

# Contrast adjustment-weakening
def weakContrast(img):
    res1 = np.uint8(np.clip((cv2.add(0.5 * img, 0)), 0, 255))
    return res1

# Brightness contrast adjustment-enhancement
def hightLightContrast(img):
    res1 = np.uint8(np.clip((cv2.add(1.5 * img, 30)), 0, 255))
    return res1

# Brightness contrast adjustment-weakening
def weakLightContrast(img):
    res1 = np.uint8(np.clip((cv2.add(0.5 * img, -30)), 0, 255))
    return res1

# Pass in the string variable, lines is the read list variable
def changeYoloTxtLine(lines):
    ResLines = []
    for line in lines:
        Rline = line.split(' ')
        Rline[1] = str(1.0-float(Rline[1]))
        Rline[2] = str(1.0-float(Rline[2]))
        ResLines.append(Rline[0] +' ' +Rline[1] +' ' +Rline[2]+' '+Rline[3])
    return ResLines


# Image flipped horizontally
def changeYoloTxtLine_W(lines):
    ResLines = []
    for line in lines:
        
        Rline = line.split(' ')
        Rline[1] = str(1.0-float(Rline[1]))
        Rline[2] = Rline[2]
        ResLines.append(Rline[0] +' ' +Rline[1] +' ' +Rline[2]+' '+Rline[3])
    return ResLines

# Image flipped vertically
def changeYoloTxtLine_H(lines):
    ResLines = []
    for line in lines:
        Rline = line.split(' ')
        Rline[1] = Rline[1]
        Rline[2] = str(1.0-float(Rline[2]))
        ResLines.append(Rline[0] +' ' +Rline[1] +' ' +Rline[2]+' '+Rline[3])
    return ResLines

def DataAugmentation(inputPathImg,outputPathImg,index,inputPathYolo,outputPathYolo):
    img = cv2.imread(inputPathImg, 1)
    GBImg = GaussianBlur(img.copy())
    GNImg = GaussianNoise(img.copy())
    SPImg = SPnoise(img.copy())
    mbImg = motion_blur(img.copy())
    HCImg = hightContrast(img.copy())
    WCImg = weakContrast(img.copy())
    HLCImg = hightLightContrast(img.copy())
    WLCImg = weakLightContrast(img.copy())

    Yolo = open(inputPathYolo, encoding='utf-8')
    YoloTxt = Yolo.readlines()
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--img.JPG'), img)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--GBImg.JPG'), GBImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--GNImg.JPG'), GNImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--SPImg.JPG'), SPImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--mbImg.JPG'), mbImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--HCImg.JPG'), HCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--WCImg.JPG'), WCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--HLCImg.JPG'), HLCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--WLCImg.JPG'), WLCImg)

    with open(os.path.join(outputPathYolo, str(index)+'--img.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--GBImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--GNImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--SPImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--mbImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--HCImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--WCImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--HLCImg.txt'), 'w') as f:
        f.writelines(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--WLCImg.txt'), 'w') as f:
        f.writelines(YoloTxt)

    # Perform horizontal flipping, and then perform data enhancement again
    v_h_img = None
    v_h_img = cv2.flip(img.copy(), 1, v_h_img)
    GBImg = GaussianBlur(v_h_img.copy())
    GNImg = GaussianNoise(v_h_img.copy())
    SPImg = SPnoise(v_h_img.copy())
    mbImg = motion_blur(v_h_img.copy())
    HCImg = hightContrast(v_h_img.copy())
    WCImg = weakContrast(v_h_img.copy())
    HLCImg = hightLightContrast(v_h_img.copy())
    WLCImg = weakLightContrast(v_h_img.copy())

    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--v_img.JPG'), v_h_img)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--v_GBImg.JPG'), GBImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--v_GNImg.JPG'), GNImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--v_SPImg.JPG'), SPImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--v_mbImg.JPG'), mbImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--v_HCImg.JPG'), HCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--v_WCImg.JPG'), WCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--v_HLCImg.JPG'), HLCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--v_WLCImg.JPG'), WLCImg)

    v_h_YoloTxt = changeYoloTxtLine_W(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_img.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_GBImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_GNImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_SPImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_mbImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_HCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_WCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_HLCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--v_WLCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)

    # Perform vertical flip, and then perform data enhancement again
    v_h_img = None
    v_h_img = cv2.flip(img.copy(), 0, v_h_img)
    GBImg = GaussianBlur(v_h_img.copy())
    GNImg = GaussianNoise(v_h_img.copy())
    SPImg = SPnoise(v_h_img.copy())
    mbImg = motion_blur(v_h_img.copy())
    HCImg = hightContrast(v_h_img.copy())
    WCImg = weakContrast(v_h_img.copy())
    HLCImg = hightLightContrast(v_h_img.copy())
    WLCImg = weakLightContrast(v_h_img.copy())

    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_img.JPG'), v_h_img)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_GBImg.JPG'), GBImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_GNImg.JPG'), GNImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_SPImg.JPG'), SPImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_mbImg.JPG'), mbImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_HCImg.JPG'), HCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_WCImg.JPG'), WCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_HLCImg.JPG'), HLCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_WLCImg.JPG'), WLCImg)

    v_h_YoloTxt = changeYoloTxtLine_H(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index) + '--h_img.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_GBImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_GNImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_SPImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_mbImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_HCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_WCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_HLCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_WLCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)

    # Perform horizontal and vertical flipping, and then perform data enhancement again
    v_h_img = None
    v_h_img = cv2.flip(img.copy(), -1, v_h_img)
    GBImg = GaussianBlur(v_h_img.copy())
    GNImg = GaussianNoise(v_h_img.copy())
    SPImg = SPnoise(v_h_img.copy())
    mbImg = motion_blur(v_h_img.copy())
    HCImg = hightContrast(v_h_img.copy())
    WCImg = weakContrast(v_h_img.copy())
    HLCImg = hightLightContrast(v_h_img.copy())
    WLCImg = weakLightContrast(v_h_img.copy())

    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_v_img.JPG'), v_h_img)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_v_GBImg.JPG'), GBImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_v_GNImg.JPG'), GNImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_v_SPImg.JPG'), SPImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_v_mbImg.JPG'), mbImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_v_HCImg.JPG'), HCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_v_WCImg.JPG'), WCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index)+'--h_v_HLCImg.JPG'), HLCImg)
    cv2.imwrite(os.path.join(outputPathImg, str(index) + '--h_v_WLCImg.JPG'), WLCImg)

    v_h_YoloTxt = changeYoloTxtLine(YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_img.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_GBImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_GNImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_SPImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_mbImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_HCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_WCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_HLCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    with open(os.path.join(outputPathYolo, str(index)+'--h_v_WLCImg.txt'), 'w') as f:
        f.writelines(v_h_YoloTxt)
    return 0

if __name__ == "__main__":
    start = datetime.datetime.now()
    inputPathImg = r'.\carDataset\CARKP\DA-origin\train'
    outputPathImg = r'.\carDataset\CARKP\DA-after\train'
    inputPathYolo = r'.\carDataset\CARKP\DA-origin\train'
    outputPathYolo = r'.\carDataset\CARKP\DA-after\train'

    files = sorted(glob(os.path.join(inputPathImg, '*.JPG')))
    for filename in files:
        portion=os.path.splitext(os.path.basename(filename))
        inputPathI = os.path.join(inputPathImg,filename)
        inputPathY = os.path.join(inputPathYolo,portion[0] +'.txt')
        print("Process pictures：" +inputPathI)
        DataAugmentation(inputPathI,outputPathImg,portion[0],inputPathY,outputPathYolo)
    end = datetime.datetime.now()
    print("calculating time：" + str((end - start)))

    start = datetime.datetime.now()
    inputPathImg = r'.\carDataset\CARKP\DA-origin\val'
    outputPathImg = r'.\carDataset\CARKP\DA-after\val'
    inputPathYolo = r'.\carDataset\CARKP\DA-origin\val'
    outputPathYolo = r'.\carDataset\CARKP\DA-after\val'
    files = sorted(glob(os.path.join(inputPathImg, '*.JPG')))
    for filename in files:
        portion=os.path.splitext(os.path.basename(filename))
        inputPathI = os.path.join(inputPathImg,filename)
        inputPathY = os.path.join(inputPathYolo,portion[0] +'.txt')
        print("Process pictures：" +inputPathI)
        DataAugmentation(inputPathI,outputPathImg,portion[0],inputPathY,outputPathYolo)
    end = datetime.datetime.now()
    print("calculating time：" + str((end - start) ))
