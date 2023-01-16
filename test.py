import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms, models
import os
from utils.utils import read_cfg, get_optimizer, get_device, build_network
import face_recognition
from models.CDCNs import CDCN, CDCNpp
from utils.eval import predict, calc_accuracy
cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")

device = get_device(cfg)
val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

state = torch.load("experiments/output/CDCNpp_nuaa_all_img.pth")
network = CDCNpp()
network = network.to(device)
network.load_state_dict(state['state_dict'])
network.eval()
path_test = "data/test/real"
with torch.no_grad():
    
    for fname in os.listdir(path_test) : 
        img = Image.open(os.path.join(path_test, fname))
        im_pil = val_transform(img).unsqueeze(0)
        im_pil= im_pil.to(device)

        net_depth_map, _, _, _, _, _ = network(im_pil)
        preds, score = predict(net_depth_map)

        print("fname: {}, pred: {}, score:{}".format(fname, preds, score))
        # img = cv2.imread(fname)
        # face_locations = face_recognition.face_locations(img)
        # if face_locations is not None :
        #     for (top, right, bottom, left) in (face_locations):
        #         face_frame = img[top : bottom, left : right, :]
        #         im_pil = Image.fromarray(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
        #         im_pil = val_transform(im_pil).unsqueeze(0)
        #         im_pil= im_pil.to(device)

        #         net_depth_map, _, _, _, _, _ = network(im_pil)
        #         preds, score = predict(net_depth_map)

        #         print(preds, score)
        # else : print(fname + "not dectect face")