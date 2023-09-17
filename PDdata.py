from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse
from matplotlib import pyplot as plt

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    img = plt.imread(im_path)
    im_h,im_w = img.shape[0],img.shape[1]
    gtData = open(im_path.replace('.JPG','.txt'), 'r').readlines()
    points=[]
    for gtline in gtData:
        a = gtline.split(' ')
        points.append([round(float(a[1]),6)*float(im_w), round(float(a[2]),6)*float(im_h)])
    points = np.array(points).astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(img)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points

def find_dis(point):
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--origin-dir', default=r'.\carDataset\CARKP\DA-after',
                        help='original data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    min_size = 256
    max_size = 10000

    for phase in ['train', 'val']:
        dir = os.path.join(args.origin_dir, phase)
        files = os.listdir(dir)
        img_paths = []
        for filename in files:
            portion = os.path.splitext(filename)
            if portion[1] == '.JPG':
                img_paths.append(os.path.join(dir, filename))
        for im_path in img_paths:
            im, points = generate_data(im_path)
            im.save(im_path)
            if phase == 'train':
                gd_save_path = im_path.replace('JPG', 'npy')
                dis = find_dis(points)
                points = np.concatenate((points, dis), axis=1)
                np.save(gd_save_path, points)
            else:
                gd_save_path = im_path.replace('JPG', 'npy')
                np.save(gd_save_path, points)

