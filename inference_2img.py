from share import *
import os

from cldm.model import create_model, load_state_dict
import random
import numpy as np
from PIL import Image
import cv2
import torch

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(512,512))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img)
    img = img.unsqueeze(0)

    return img

def save_image(img,path):
    # img [-1,1]
    img = img.detach().cpu()
    img = torch.clamp(img, -1., 1.)
    img = img.squeeze()
    img = img.transpose(0,1).transpose(1,2)
    img = img.numpy()
    img = (img+1)/2
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

#* ----- 參數設置 -------------

resume_path = "save_log/se3_range50_v2/logs/default/version_0/checkpoints/epoch=2-step=1013.ckpt"
t_range = 10                #* 影片長度
guidance_scale = 100.0        #* condition guidance 強度
speed = 0.15                #* 相機飛行速度，每次往 z 軸前進大小

#* ---------------------------

model = create_model('./models/cldm_v21_se3.yaml').to("cuda")
model.load_state_dict(load_state_dict(resume_path, location='cuda'))

model.eval()

img_root = "test_guidance"
begin_path = f"{img_root}/begin.png"
end_path = f"{img_root}/end.png"
begin_img = Image.open(begin_path)
end_img = Image.open(end_path)
begin_img.save(f"{img_root}/1.png")
end_img.save(f"{img_root}/{t_range}.png")

begin_img = read_image(begin_path)
end_img = read_image(end_path)

predict_seq = []
begin_seq = []
end_seq = []

def generate_seq(begin,end):
    if end-begin<2:
        return
    mid = (begin+end)//2
    predict_seq.append(mid)
    begin_seq.append(begin)
    end_seq.append(end)
    generate_seq(begin,mid)
    generate_seq(mid,end)

generate_seq(1,t_range)

se3_list = []
se3_cur = np.eye(4)
def generate_se3(t_range):
    for i in range(t_range):
        se3_list.append(se3_cur)
        se3_cur[2][3] -= speed

generate_se3(t_range)
    

prompt="nature scene,  a professional, detailed, high-quality image"

with torch.no_grad():
    for i in range(len(predict_seq)):
        begin_idx = begin_seq[i]
        end_idx = end_seq[i]
        predict_idx = predict_seq[i]
        begin_path = f"{img_root}/{begin_idx}.png"
        end_path = f"{img_root}/{end_idx}.png"
        begin_img = read_image(begin_path)
        end_img = read_image(end_path)

        t_range = end_idx-begin_idx+1

        begin_w2c = se3_list[begin_idx-1].astype(np.float32)
        mid_w2c = se3_list[predict_idx-1].astype(np.float32)
        end_w2c = se3_list[end_idx-1].astype(np.float32)
        se3_all = []
        se3_all.append(torch.tensor(begin_w2c).unsqueeze(0).unsqueeze(0).to("cuda"))
        se3_all.append(torch.tensor(mid_w2c).unsqueeze(0).unsqueeze(0).to("cuda"))
        se3_all.append(torch.tensor(end_w2c).unsqueeze(0).unsqueeze(0).to("cuda"))

        input = dict(jpg=begin_img, txt=[prompt], hint=begin_img,\
                    begin = begin_img,end = end_img, range = torch.tensor([t_range]).to("cuda"),
                    se3 = se3_all,
                    )   
        
        output = model.inference(input,unconditional_guidance_scale=guidance_scale)
        output_path = f"{img_root}/{predict_idx}-g{guidance_scale}.png"
        save_image(output,output_path)
        break






