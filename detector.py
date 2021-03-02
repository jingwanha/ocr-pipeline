import os
import sys
sys.path.append(os.path.abspath('./detector/PAN'))
sys.path.append(os.path.abspath('./detector/CRAFT/'))

import numpy as np
import cv2
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms

# PAN modules 
from models import get_model
from post_processing import decode

# CRAFT modules
from craft import CRAFT as craft_model
from utility import getDetBoxes, adjustResultCoordinates, resize_aspect_ratio, cvt2HeatmapImg,normalizeMeanVariance

from collections import OrderedDict


class Detector:
    def __init__(self, config, gpu_id=None):
        
        self.detector = None
        
        if config['name'] not in ['pan','craft'] :
            raise Exception('Detection Name should be in [pan | craft]')
            
        if config['name'] == 'pan': self.detector = PAN(config['model_path'], gpu_id)
        elif config['name'] == 'craft' : self.detector=CRAFT(config,gpu_id)
        print(config['name'].upper())
        

# PAN Model Definition
class PAN:
    def __init__(self, model_path, gpu_id=None):
        # GPU  설정
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        
        # 모델로드
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['args']['pretrained'] = False
        self.net = get_model(config)

        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()
        
    # prediction
    def predict(self, img: np.array, short_size: int = 736):

        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
                
            # 시간 측정    
            start = time.time()
            
            preds = self.net(tensor)[0]
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
                
            preds, boxes_list = decode(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            
            if len(boxes_list):
                boxes_list = boxes_list / scale
                
            t = time.time() - start
            
        return preds, boxes_list, t # preds : 
    
    
    
# CRAFT  모델 정의
class CRAFT:
    def __init__(self, config, gpu_id=None):
                
        self.gpu_id = gpu_id
        self.net = None
        
        # CRAfT PARAMETERS
        self.canvas_size = config['craft_options']['canvas_size']
        self.mag_ratio = config['craft_options']['mag_ratio']
        self.text_threshold = config['craft_options']['text_threshold']
        self.link_threshold = config['craft_options']['link_threshold']
        self.low_text = config['craft_options']['low_text']
        
        # 모델 로드
        self.net = craft_model()
        
        if gpu_id is not None: 
            self.device = torch.device('cuda')
            torch.cuda.set_device(gpu_id)
            
            self.net.load_state_dict(self._copyStateDict(torch.load(config["model_path"])))
            
        else:
            self.net.load_state_dict(self._copyStateDict(torch.load(config["model_path"], map_location='cpu')))
            

        if gpu_id is not None:
            self.net = self.net.cuda()

        self.net.eval()
        
    def _copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"): start_idx = 1
        else: start_idx = 0
            
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
            
        return new_state_dict
    
    
    def prediction(self, image:np.array):
        
        # 입력 이미지 정규화
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 
                                                                      self.canvas_size, 
                                                                      interpolation=cv2.INTER_LINEAR, 
                                                                      mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        
        if self.gpu_id is not None: x = x.cuda()

        start = time.time()
        
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        
        
        boxes, polys = getDetBoxes(score_text,score_link,
                                   self.text_threshold,
                                   self.link_threshold,
                                   self.low_text,
                                   False)
    
        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
            

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img) # text 영영의 heatmap image
        
        t = time.time() - start
        
        return polys, ret_score_text, t