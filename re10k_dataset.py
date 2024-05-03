import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange
import re

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

def custom_sort(file_name):
    # 提取文件名中的數字部分
    num_part = re.findall(r'\d+', file_name)
    if num_part:
        return int(num_part[0])  # 將數字部分轉換為整數進行排序
    else:
        return file_name

class Re10k_dataset(Dataset):
    def __init__(self,data_root,mode,max_interval=5,infer_len = 20):
        assert mode == 'train' or mode == 'test' or mode == 'validation'

        self.mode = mode

        self.inform_root = '{}/RealEstate10K/{}'.format(data_root, mode)
        self.image_root = '{}/realestate/{}'.format(data_root, mode)
        self.image_root = '{}/realestate_4fps/{}'.format(data_root, mode)

        self.transform = default_transform

        self.max_interval = max_interval   #* 兩張圖片最大的間隔
        
        self.infer_len = infer_len  #* inference 時要生成的影片長度

        self.video_dirs = []
        self.image_dirs = []
        self.inform_dirs = []
        self.total_img = 0

        #* 原始圖像大小
        H = 360
        W = 640

        #* 256 x 256 縮放版本
        # H = 256
        # W = 455

        #* 64 x 64 縮放版本
        # H = 64
        # W = 114

        self.H = H
        self.W = W

        self.sd_H = 512
        self.sd_W = 512

        self.square_crop = True     #* 是否有做 center crop 

        xscale = W / min(H, W)      #* crop 之後 cx cy 的縮放比例
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale


        for video_dir in sorted(os.listdir(self.image_root)):
            npz_file_path = f"{self.image_root}/{video_dir}/data.npz"
            if os.path.isfile(npz_file_path):
                self.video_dirs.append(video_dir)

        print(f"video num: {len(self.video_dirs)}")
        print(f"load {mode} data finish")
        print(f"-------------------------------------------")


    def __len__(self):
        return len(self.video_dirs) 
    
    def get_image(self,index):

        #* 選哪一個video
        video_idx = index

        #* 讀取video 每個frame 的檔名
        frame_namelist = []
        video_dir = self.video_dirs[video_idx]
        npz_file_path = f"{self.image_root}/{video_dir}/data.npz"
        if os.path.isfile(npz_file_path) == False:
            return None, None, None, None, None, False
        npz_file = np.load(npz_file_path)

        for file_name in sorted(npz_file.files, key=custom_sort):
            frame_namelist.append(file_name)

        if len(frame_namelist) <= self.max_interval:
            return None, None, None, None, None, False
        
        if self.mode=="test" and len(frame_namelist) < self.infer_len:   #* inference 時影片長度小於要生成的長度
            return None, None, None, None, None, False


        #* 隨機取間距
        interval_len = np.random.randint(self.max_interval) + 1

        #* 隨機取origin frame
        frame_idx = np.random.randint(len(frame_namelist)-interval_len)

        image_seq = []
        frame_idxs = [frame_idx, frame_idx+interval_len]  #* 兩張圖片, 一個origin 一個target

        if self.mode == "test":     #* 做 inference 取 infer_len 張圖片，用來做比較
            frame_idxs = np.arange(self.infer_len)

        cnt = 0
        for idx in frame_idxs:
            frame_name = frame_namelist[idx]
            img_np = npz_file[frame_name]
            # print(f"img ori shape:{img_np.shape}")
            img = Image.fromarray(img_np)
            img = img.resize((self.W,self.H))
            img = self.crop_image(img)
            img = img.resize((self.sd_W,self.sd_H))
            # if cnt == 0:
            #     src_img_numpy = np.array(img)
            #     src_img_numpy = src_img_numpy / 255.0 
            #     src_img = self.midas_transform({"image": src_img_numpy})["image"]
            #     src_img_tensor = torch.from_numpy(src_img)

            img_tensor = self.transform(img)
            img_tensor = img_tensor*2 - 1
            image_seq.append(img_tensor)
            cnt += 1
        
        image_seq = torch.stack(image_seq)

        return image_seq,frame_idx,interval_len,frame_namelist, True


    def get_information(self,index,frame_idx,interval_len,frame_namelist):

        #* 讀取選定video 的 information txt
        video_idx = index
        video_dir = self.video_dirs[video_idx]
        inform_path = '{}/{}.txt'.format(self.inform_root,video_dir)

        frame_num = -1
        frame_list = []

        with open(inform_path, 'r') as file:
            for line in file:
                frame_num+=1
                if frame_num==0:
                    continue
                frame_informlist = line.split()
                frame_list.append(frame_informlist)

        #* 同一個video 的intrinsic 都一樣
        fx,fy,cx,cy = np.array(frame_list[0][1:5], dtype=float)


        intrinsics = np.array([ [fx,0,cx,0],
                                [0,fy,cy,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        
        #* unnormalize
        # intrinsics[0] = intrinsics[0]*self.W
        # intrinsics[1] = intrinsics[1]*self.H

        #* 調整 crop 後的 cx cy
        # if self.square_crop:
        #     intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
        #     intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

        c2w_seq = []
        frame_idxs = [frame_idx, frame_idx+interval_len]
        
        if self.mode == "test":      #* 做 inference 取 infer_len 張圖片，用來做比較
            frame_idxs = np.arange(self.infer_len)

        # for idx in frame_idxs:
        #     c2w = np.array(frame_list[idx][7:], dtype=float).reshape(3,4)
        #     c2w_4x4 = np.eye(4)
        #     c2w_4x4[:3,:] = c2w
        #     c2w_seq.append(torch.tensor(c2w_4x4))
        
        f_idx = 0
        for idx in range(len(frame_list)):
            #* 根據timestamp 與圖片檔名配對，找到對應的extrinsic
            if int(frame_list[idx][0])//1000 != int(frame_namelist[frame_idxs[f_idx]].split('.')[0])//1000: 
                continue

            c2w = np.array(frame_list[idx][7:], dtype=float).reshape(3,4)
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3,:] = c2w
            c2w_seq.append(torch.tensor(c2w_4x4))

            f_idx+=1
            if f_idx == len(frame_idxs):
                break

        c2w_seq = torch.stack(c2w_seq)

        intrinsics = torch.from_numpy(intrinsics)

        return intrinsics, c2w_seq
    
    def crop_image(self,img):
        original_width, original_height = img.size

        # center crop 的大小
        new_width = min(original_height,original_width)
        new_height = new_width

        # 保留中心的crop 的部分
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # 使用PIL的crop方法来截取图像
        cropped_img = img.crop((left, top, right, bottom))

        return cropped_img

    def __getitem__(self,index):

        img, frame_idx,interval_len, frame_namelist, good_video = self.get_image(index)

        #* video frame 數量 < max interval 
        #* 或 < inference 的長度
        if good_video == False:
            # print(f"false")
            return self.__getitem__(index+1)

        intrinsics,c2w = self.get_information(index,frame_idx,interval_len,frame_namelist)

        infer_result = {
            'img':img,
            'intrinsics':intrinsics,
            'c2w': c2w
        }

        #* 用來對標controlnet 的condition image, (-1,1) , (h,w,c)
        src_img = img[0]
        src_img = rearrange(src_img, 'c h w -> h w c')

        #* 用來對標controlnet 的target image, (-1,1)
        tgt_img = img[1]
        tgt_img = rearrange(tgt_img, 'c h w -> h w c')

        result = {
            'src_img':src_img,
            'tgt_img':tgt_img,
            'intrinsics':intrinsics,
            'c2w': c2w,
            'txt': " ",
        }

        if self.mode == "train":
            return result
        elif self.mode == "test":
            return infer_result


if __name__ == '__main__':
    test = Re10k_dataset("../../dataset","test")
    print(test.video_dirs[:10])
    # data = test[0]
    # print(data['img'].shape)
    # print(data['intrinsics'])
    # print(data['w2c'][0])
    # print(test.__len__())
    # print(test.interval_len)

    # for i in range(data['img'].shape[0]):
    #     image = data['img'][i].numpy()
    #     image = (image+1)/2
    #     image *= 255
    #     image = image.astype(np.uint8)
    #     image = rearrange(image,"C H W -> H W C")
    #     image = Image.fromarray(image)
    #     image.save(f"../test_folder/test_{i}.png")
