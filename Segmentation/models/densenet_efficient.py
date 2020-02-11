import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

def bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient = False):
        super(Dense_Layer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace = True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size = 1, stride = 1, bias = False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace = True)),
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate, Kernel_size = 3, stride =1, padding = 1, bias = False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p = self.drop_rate, training= self.training)
        return new_features

class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace = True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, Kernel_size =1, stride = 1, bias = False))
        self.add_module('pool', nn.AvgPool2d(kernel_size = 2, stride = 2))

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient = False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features = num_input_features + i* growth_rate, growth_rate = growth_rate,
             bn_size = bn_size, drop_rate = drop_rate, efficient = efficient)
            self.add_module('denselayer%d' % ( i+1), layer)
    
    def forward(self, init_features):
        features = [init_features]
        

