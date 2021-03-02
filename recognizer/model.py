"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

from torch.nn import Transformer

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.stages = {'Trans': cfg['model']['transform'], 'Feat': cfg['model']['extraction'],
                       'Seq': cfg['model']['sequence'], 'Pred': cfg['model']['prediction']}
        
        

        """ Transformation """
        if cfg['model']['transform'] == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=cfg['transform']['num_fiducial'],
                I_size=(cfg['dataset']['imgH'],cfg['dataset']['imgW']),
                I_r_size=(cfg['dataset']['imgH'], cfg['dataset']['imgW']),
                I_channel_num=cfg['model']['input_channel'])
            print ("Transformation moduls : {}".format(cfg['model']['transform']))
            
        else:
            print('No Transformation module specified')
            
            
            
        """ FeatureExtraction """
        if cfg['model']['extraction'] == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(cfg['model']['input_channel'], cfg['model']['output_channel'])
            
        elif cfg['model']['extraction'] == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(cfg['model']['input_channel'], cfg['model']['output_channel'])
            
        elif cfg['model']['extraction'] == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(cfg['model']['input_channel'], cfg['model']['output_channel'])
            
        else:
            raise Exception('No FeatureExtraction module specified')
            
        self.FeatureExtraction_output = cfg['model']['output_channel']  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        print ('Feature extractor : {}'.format(cfg['model']['extraction']))

        
        """ Sequence modeling"""
        if cfg['model']['sequence'] == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, cfg['model']['hidden_size'], cfg['model']['hidden_size']),
                BidirectionalLSTM(cfg['model']['hidden_size'], cfg['model']['hidden_size'], cfg['model']['hidden_size']))
            self.SequenceModeling_output = cfg['model']['hidden_size']
        
        # SequenceModeling : Transformer
        elif cfg['model']['sequence'] == 'Transformer':
            self.SequenceModeling = Transformer(
                d_model=self.FeatureExtraction_output, 
                nhead=2, 
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=cfg['model']['hidden_size'],
                dropout=0.1,
                activation='relu')
            print('SequenceModeling: Transformer initialized.')
            self.SequenceModeling_output = self.FeatureExtraction_output # 입력의 차원과 같은 차원으로 출력 됨
        
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output
        print('Sequence modeling : {}'.format(cfg['model']['sequence']))

        
        
        """ Prediction """
        if cfg['model']['prediction'] == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, cfg['training']['num_class'])
            
        elif cfg['model']['prediction'] == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, cfg['model']['hidden_size'], cfg['training']['num_class'])
            
        elif cfg['model']['prediction'] == 'Transformer':
            self.Prediction = nn.Linear(self.SequenceModeling_output, cfg['training']['num_class'])
            
        else:
            raise Exception('Prediction should be in [CTC | Attn | Transformer]')
        
        print ("Prediction : {}".format(cfg['model']['prediction']))
            
            
            
    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        elif self.stages['Seq'] == 'Transformer':
            contextual_feature = visual_feature.contiguous().permute(1, 0, 2)
            contextual_feature = self.SequenceModeling(contextual_feature, contextual_feature)
            contextual_feature = contextual_feature.contiguous().permute(1, 0, 2)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        # Prediction이라는 이름의 내부 인스턴스로 접근하나, 호출 시 매개변수가 다 다름.
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
            
        elif self.stages['Pred'] == 'Transformer':
            batch_size = contextual_feature.shape[0]
            seq_length = contextual_feature.shape[1]
            # print(contextual_feature.shape)
            # print(contextual_feature.contiguous().view(-1, self.SequenceModeling_output).shape)
            prediction = self.Prediction(contextual_feature.contiguous().view(-1, self.SequenceModeling_output))
            # print(prediction.shape)
            prediction = prediction.reshape(-1, seq_length, self.Prediction.out_features)
            # print(prediction.shape)
            
        else: # Attn.
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.cfg['training']['batch_max_length'])

        return prediction
