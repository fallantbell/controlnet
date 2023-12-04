import json
import cv2
import numpy as np
import os
import random

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,max_time=100,mode=None):
        self.video = []

        self.max_time = max_time    #* 兩張圖最大的時間間隔 

        if mode == "train":
            self.root_path = "../../../disk2/icchiu/acid_dataset/video/train"
        elif mode == "validation":
            self.root_path = "../../../disk2/icchiu/acid_dataset/video/validation"
        else:
            print("Wrong dataset path!")
            raise ValueError("Wrong dataset path!")

        self.prompt = "nature scene,  a professional, detailed, high-quality image"

        for video in os.listdir(self.root_path):
            self.video.append(video)

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        # item = self.data[idx]
        video_name = self.video[idx]
        video_path = f"{self.root_path}/{video_name}"

        # print(video_path)

        t_range = random.randint(3,self.max_time)      #* 隨機取一個區間大小
        begin_t = random.randint(1,self.max_time-t_range+1)   #* 隨機取一個起始的frame

        end_t = begin_t+(t_range-1)     #* end image index

        # print(self.max_time)
        begin_name = "{:03d}".format(begin_t)       #* 轉成image 名稱格式: 001 002 003 ...
        begin_name = "{:05d}".format(begin_t)       #! inf v2 dataset
        begin_path = f"{video_path}/{begin_name}.png"
        begin_img = cv2.imread(begin_path)
        begin_img = cv2.cvtColor(begin_img, cv2.COLOR_BGR2RGB)
        h,w,c = begin_img.shape
        crop_size = int((4/6)*h)
        w1 = int((w-crop_size)//2)
        h1 = int((1/6)*h)
        cropped_begin_img = begin_img[h1:h1+crop_size,w1:w1+crop_size]

        end_name = "{:03d}".format(end_t)
        end_name = "{:05d}".format(end_t)   #! inf v2 dataset
        end_path = f"{video_path}/{end_name}.png"
        end_img = cv2.imread(end_path)
        end_img = cv2.cvtColor(end_img, cv2.COLOR_BGR2RGB)
        cropped_end_img = end_img[h1:h1+crop_size,w1:w1+crop_size]

        #* 取中間的image
        mid_t = (begin_t+end_t)//2
        mid_name = "{:03d}".format(mid_t)
        mid_name = "{:05d}".format(mid_t)   #! inf v2 dataset
        mid_path = f"{video_path}/{mid_name}.png"
        mid_img = cv2.imread(mid_path)
        mid_img = cv2.cvtColor(mid_img, cv2.COLOR_BGR2RGB)
        cropped_mid_img = mid_img[h1:h1+crop_size,w1:w1+crop_size]

        cv2.imwrite("test_img/crop_begin.png",cropped_begin_img)
        cv2.imwrite("test_img/crop_mid.png",cropped_mid_img)
        cv2.imwrite("test_img/crop_end.png",cropped_end_img)
        cv2.imwrite("test_img/begin.png",begin_img)
        cv2.imwrite("test_img/mid.png",mid_img)
        cv2.imwrite("test_img/end.png",end_img)
        
        
        begin_img = cv2.resize(cropped_begin_img,(512,512))
        end_img = cv2.resize(cropped_end_img,(512,512))
        mid_img = cv2.resize(cropped_mid_img,(512,512))

        cv2.imwrite("test_img/big_begin.png",begin_img)
        cv2.imwrite("test_img/big_mid.png",mid_img)
        cv2.imwrite("test_img/big_end.png",end_img)


        # Normalize source images to [0, 1].
        begin_img = begin_img.astype(np.float32) / 255.0
        end_img = end_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        mid_img = (mid_img.astype(np.float32) / 127.5) - 1.0


        se3_path = f"{video_path}/se3.npy"
        se3_list = np.load(se3_path)
        begin_w2c = se3_list[begin_t-1].astype(np.float32)
        mid_w2c = se3_list[mid_t-1].astype(np.float32)
        end_w2c = se3_list[end_t-1].astype(np.float32)
        se3_all = []
        se3_all.append(begin_w2c)
        se3_all.append(mid_w2c)
        se3_all.append(end_w2c)

        return dict(jpg=mid_img, txt=self.prompt, hint=begin_img,\
                    begin = begin_img,end = end_img, range = t_range,se3 = se3_all)

