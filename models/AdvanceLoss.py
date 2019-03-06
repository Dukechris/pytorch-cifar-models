import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math


# def get_average(list):
def get_average(num):
    sum = 0
    for i in range(len(num)):
        sum += num[i]
    return sum/len(num)
 
 

def get_range(num):
    return max(num) - min(num)
 
 

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2)-1
        return listnum[i]
    else:
        i = int(lnum / 2)-1
        return (listnum[i] + listnum[i + 1]) / 2
 

def get_variance(num):
    sum = 0
    average = get_average(num)
    for i in range(len(num)):
        sum += (num[i] - average)**2
    return sum/len(num)
 

def get_stddev(num):
    average = get_average(num)
    sdsq = sum( [(num[i] - average) ** 2 for i in range(len(num))] )
    stdev = (sdsq / (len(num) - 1)) ** .5
    return stdev
 

def get_n_moment(num,n):
    sum = 0
    for i in ange(len(num)):
        sum += num[i]**n
    return sum/len(num)






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
        kernal_metric = torch.exp(-1.0 * metric / 1.2)
        # Corresponding kernal metric calculating
        cor_metrics = []
        # Corresponding Euchlidean metric 
        cor_eu_metrics = []
        for i in range(kernal_metric.size(0)):
            label_i = int(label[i])
            distance = kernal_metric[i, label_i].item()
            cor_metrics.append(distance)
            distance = metric[i, label_i].item()
            cor_eu_metrics.append(distance)
        avg_k_distance = get_average(cor_metrics)
        var_k_distance = get_variance(cor_metrics)
        avg_e_distance = get_average(cor_eu_metrics)
        var_e_distance = get_variance(cor_eu_metrics)
        print('The average corresponding metric is {:.4f}'.format(avg_distance))
        print('The variance metric is {:.4f}'.format(var_distance))
        print('The average corresponding eu metric is {:.4f}'.format(avg_e_distance))
        print('The variance metric eu is {:.4f}'.format(var_e_distance))
        if avg_distance < 0.5:
            avg_distance = 0.5
        self.scale = (1.0/avg_distance) * math.log(self.class_num-1.0) #(get_average(Bs))
        # Return data
        train_logits = 3.0 * self.scale * kernal_metric

        # return train_logits, kernal_metric
        return train_logits


