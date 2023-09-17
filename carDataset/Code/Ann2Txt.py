import cv2
import numpy as np
import os
import re
import argparse
from PIL import Image
import scipy

def changeLine(lines,w,h):
    lines = lines.split(' ')
    x1 = int(lines[0])
    y1 = int(lines[1])
    x2 = int(lines[2])
    y2 = int(lines[3])

    Kx = (x1+x2)/2
    ky = (y1+y2)/2
    return str( 1 ) +' ' + str(Kx/w) +' ' + str(ky/h) + ' \n'

txt_data = r'/home/chuanzhi/mnt_3T/zyt/CARPKandPUCPR/PUCPR+_devkit/data/txttxt' # Save as required yolo annotation file
ann_data = r'/home/chuanzhi/mnt_3T/zyt/CARPKandPUCPR/PUCPR+_devkit/data/Annotations' # box marks the file path

files = os.listdir(ann_data)

for file in files:
    ann_path = os.path.join(ann_data,file)
    img_path = ann_path.replace('Annotations','Images').replace('.txt','.jpg')
    img = cv2.imread(img_path, 1)
    h,w = img.shape[0],img.shape[1]
    ann = open(ann_path, encoding='utf-8')
    annLines = ann.readlines()
    result = []
    for lines in annLines:
        resultL = changeLine(lines,w,h)
        result.append(resultL)
    print(result)
    # result stores the txt file that needs to be saved
    with open(os.path.join(txt_data, file), 'w') as f:
        f.writelines(result)
