import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math



class PureKernalMetricLogits(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(PureKernalMetricLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weights = nn.Parameter( torch.FloatTensor(class_num, feature_dim))
        self.scale = 2.0 * math.log(self.class_num - 1)
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, feat, label):
        # Calculating metric
        diff = torch.unsqueeze(self.weights, dim=0) - torch.unsqueeze(feat, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)
        kernal_metric = torch.exp(-1.0 * metric / 1.8)
        # Corresponding kernal metric calculating
        cor_metrics = []
        for i in range(kernal_metric.size(0)):
            label_i = int(label[i])
            distance = kernal_metric[i, label_i].item()
            cor_metrics.append(distance)
        avg_distance = get_average(cor_metrics)
        if avg_distance < 0.5:
            avg_distance = 0.5
        self.scale = (1.0/avg_distance) * math.log(self.class_num-1.0) #(get_average(Bs))
        # Return data
        train_logits = self.scale * kernal_metric
        # return train_logits, kernal_metric
        return train_logits


