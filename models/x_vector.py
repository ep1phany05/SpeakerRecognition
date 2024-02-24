# -*- coding:utf-8 -*-
# file name: x_vector.py
import torch.nn as nn
from models.tdnn import TDNN
import torch


class Xvector(nn.Module):
    def __init__(self, input_dim=40, emb_dim=512, num_classes=8):
        super(Xvector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=emb_dim, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=emb_dim, output_dim=emb_dim, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=emb_dim, output_dim=emb_dim, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=emb_dim, output_dim=emb_dim, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=emb_dim, output_dim=emb_dim, context_size=1, dilation=3, dropout_p=0.5)
        # Frame levelPooling
        self.segment6 = nn.Linear(emb_dim * 2, emb_dim)
        self.segment7 = nn.Linear(emb_dim, emb_dim)
        self.output = nn.Linear(emb_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def forward(self, inputs):
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        # Stat Pool

        mean = torch.mean(tdnn5_out, 1)
        std = torch.var(tdnn5_out, 1)
        stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions, x_vec

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
