import torch

import numpy as np
import cv2 

from typing import List,Tuple
import os
import sys
sys.path.append(os.path.abspath('./recognizer'))

from converter import CTCLabelConverter, AttnLabelConverter
from utils import modeling_dataset
from model import Model

class Recognizer:
    def __init__(self, config, gpu_id=None):
        self.net = None
        self.converter = None
        self.device = None
        self.prediction_type = config['model']["prediction"]
                    
        self.input_channel = config['model']['input_channel']
        self.imgH = config['dataset']['imgH']
        self.imgW = config['dataset']['imgW']
        
        
        self.batch_max_length = config['training']['batch_max_length'] 
        
        if gpu_id is not None: 
            self.device = torch.device('cuda')
            torch.cuda.set_device(gpu_id)
            
        else: 
            self.device = 'cpu'
            torch.cuda.set_device(-1)
        

        characters = config['dataset']['characters']
        self.converter = CTCLabelConverter(characters) if 'CTC' in config["model"]["prediction"] else AttnLabelConverter(characters)
    
        
        self.net = Model(config)

        self.net = torch.nn.DataParallel(self.net,device_ids=[gpu_id]).to(self.device)
        self.net.load_state_dict(torch.load(config['model']['saved_model'], map_location=self.device))

        self.net.eval()
        
    def _normalized_input(self, images: List[np.array] ,batch_size : int) -> torch.Tensor:
        # 0~255의 픽셀값을 가지는 이미지를 -1~1 사이의 픽셀값을 가지는 이미지로 정규화 하는 함숫
        input_images = np.zeros((batch_size,
                                 self.input_channel,
                                 self.imgH,
                                 self.imgW),dtype=np.float64)

        for idx, img in enumerate(images):
            img = cv2.resize(img,dsize=(self.imgW,self.imgH))\
            # 값 정규화
            input_images[idx,0,:,:] = (img-127.5)/127.5

        input_images = torch.FloatTensor(input_images)
        return input_images
    
    def prediction(self, text_images : List[np.array]) -> List[Tuple]:
        # text_images는 이미지(np.array)가 저장되어 있는 리스트
        # 각 입력 이미지의 픽셀값은 0~255 사이의 값을 가짐
        
        batch_size = len(text_images)
        length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
        
        # 모델의 입력에 맞게 이미지 픽셀 값을 -1~1 사이로 정규화
        input_images = self._normalized_input(text_images,batch_size)
        
        # prediction
        preds =  self.net(input_images, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)
        
        result_str = []
        
        if self.prediction_type == 'CTC' : result_str = preds_str
        else : 
            for pred in preds_str: 
                eos = pred.find('[s]') #end of sentence 
                result_str.append(pred[:eos])
            
        return (text_images,result_str)