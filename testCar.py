import torch
import os
import numpy as np
import pandas as pd
from datasets.crowd_fish import Crowd
from models.LDNet import vgg19
import argparse
import time
import cv2
args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default='test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./result/CARPK',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu


    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_val.pth'), device))
    epoch_minus = []


    df = pd.DataFrame(columns=['imgNum','AverageMAE','RMSE', 'Accuracy', 'times'])
    dfIMG = pd.DataFrame(columns=['imageName', 'trueNum', 'predictNum','Accuracy','times'])
    imgIndex = 0
    start = time.time()
    totalAcc = 0
    datasets = Crowd(args.data_dir, 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    RMSESum = 0
    epoch_minus=[]
    i = 0
    for inputs, name, imgPath in dataloader:
        startIMG = time.time()
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            print(imgPath)
            txtPath = imgPath[0].replace('.JPG','.txt')
            Yolo = open(txtPath, encoding='utf-8')
            YoloTxt = Yolo.readlines()
            GT = len(YoloTxt)
            temp_minu = GT - torch.sum(outputs).item()
            print(name[0], temp_minu, GT, torch.sum(outputs).item())
            epoch_minus.append(temp_minu)
        RMSESum = (GT - torch.sum(outputs).item())*(GT - torch.sum(outputs).item()) + RMSESum
        i=i+1
        endIMG = time.time()
        pre = torch.sum(outputs).item()
        acc = 1-abs(pre-GT)/GT
        totalAcc = acc + totalAcc
        new = [name[0], GT, pre, acc , endIMG - startIMG]
        dfIMG.loc[imgIndex] = new
        imgIndex = imgIndex + 1
        # dm = outputs.squeeze().detach().cpu().numpy()
        # plt.imshow(dm, cmap=CM.jet)
        #
        # plt.savefig(imgPath[0].replace('test','testD'), bbox_inches="tight", pad_inches=0.0, dpi=96)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
    end = time.time()
    AMAE = mae
    Accuracy = totalAcc / i
    RMSE = (RMSESum/i)**0.5
    new = [i, AMAE, RMSE ,Accuracy, (end - start)/i]
    df.loc[0] = new

df.to_csv(os.path.join(args.save_dir , "OriginTest.csv"), index=False)  # CSV文件路径
dfIMG.to_csv(os.path.join(args.save_dir ,"OriginTestDetial.csv"), index=False)

