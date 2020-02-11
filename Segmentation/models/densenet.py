# import os 
# import numpy as np
# import pandas as pd 
# import re
# import torch.nn as nn
# import torch.nn.functional as F 
# import torch.utils.ceckpoint as cp 
# from collections import OrderedDict

# class Bottleneck(nn.Module):
#     def __init__(self, n_channels, growth_rate):
#         super(Bottleneck, self).__init__()
#         inter_channels = 4 * growth_rate
#         self.bn1 = nn.BatchNorm2d(n_channels)
#         self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size = 1, bias = False )
#         self.bn2 = nn.BatchNorm2d(inter_channels)
#         self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size = 3, padding = 1, bias = False)

#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = torch.cat((x, out), 1)
#         return out

# class single_layer(nn.Module):
#     def __init__(self, n_channels, n_out_channels):
#         super(single_layer, self).__init__()
#         self.bn1 = nn.BatchNorm2d(n_channels)
#         self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size = 3, padding = 1,  bias = False)
#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = torch.cat((x, out), 1)
#         return out


# class transition(nn.Module):
#     def __init__(self, n_channels, n_out_channels):
#         super(transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(n_channels)
#         self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size =1 , bias = False)
#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = F.avg_pool2d( out, 2)
#         return out

# class DenseNet(nn.Module):
#     def __init__(self, growth_rate, depth, reduction, n_classes, bottleneck):
#         super(DenseNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 1, bias = False) #can be 7
#         sel.pool1 = nn.MaxPool2d(kernel = 3, stride = 2)
#         self.dense1 = self.make_dense(n_channels = 64, growth_rate = 32, n_dense_blocks = 6, bottleneck)
#         self.trans1 = transition(n_channels = 256, n_out_channels = 128)
#         self.dense2 = self.make_dense(n_channels = 128 , growth_rate = 32,n_dense_blocks = 12, bottleneck)
#         self.trans2 = transition(n_channels = 512,n_out_channels = 256)
#         self.dense3 = self.make_dense(n_channels = 256, growth_rate = 32, n_dense_blocks = 24, bottleneck)
#         self.trans3 = transition(n_channels = 1024, n_out_channels = 512)
#         self.dense4 = self.make_dense(n_channels = 512 , growth_rate = 32, n_dense_blocks = 16, bottleneck)
#         self.trans4 = transition(n_channels = 1024 ,n_out_channels = 1)

#     def make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck):
#         layers = []
#         for i in range(int(n_dense_blocks)):
#             if bottleneck:
#                 layers.append(Bottleneck(n_channels, growth_rate))
#             else:
#                 layers.append(single_layer(n_channels, growth_rate))
#             n_channels += growth_rate
#         return nn.Sequential(*layers)


#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.dense1(out)
#         out = self.trans1(out)
#         out = self.dense2(out) 
#         out = self.trans2(out)
#         out = self.dense3(out)
#         out = self.trans3(out)
#         out = self.dense4(out)
#         out = self.trans4(out)













# class Dense_Block(nn.Module):
#     def __init__(self, in_channels):
#         super(Dense_Block, self).__init__()
#         self.relu = nn.ReLU(inplace = True)
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
#         self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size =3, stride = 1, padding = 1)
#         self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size  =3, stride = 1, padding = 1)
#         self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
#         self.conv5 = nn.Conv2d(in_channels =128, out_channels = 32, kernel_size = 3, stride =1, padding = 1)

#         def forward(self, x):
#             bn = self.bn(x)
#             conv1 = self.relu(self.conv1(bn))
#             conv2 = self.relu(self.conv2(conv1))
#             c2_dense = self.relu(torch.cat([conv1, conv2], 1))
#             conv3 = self.relu(self.conv3(c2_dense))
#             c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
#             conv4 = self.relu(self.conv4(c3_dense))
#             c4_dense =  self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
#             conv5 = self.relu(self.conv5(c4_dense))
#             c5_dense = self.relu(torch.cat([conv1, conv2, conv3,conv4, conv5], 1))

#             return c5_dense

# class Transition_Layer(nn.Module):
#     def __init__(self, in_channels, out_channels)):
#         super(Transition_Layer, self).__init__()
#         self.relu = nn.ReLU(inplace = True)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
#         self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

#         def forward(self, x):
#             conv = self.relu(self.conv(x))
#             bn = self.bn(conv)
#             out = self.avg_pool(bn)

#             return out

# class DenseNet(nn.Module):
#     def __init__(self, n_classes):
#         super(DenseNet, self ).__init__()
#         self.conv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 3, bias =False)
#         self.relu  = nn.ReLU()

#         def make_dense_block(self, block, in_channels):
#             layers =[]
#             layers.append(block(in_channels))
#             return nn.Sequential(*layers)
        
#         def make_transition_layer(self,layer, in_channels, out_channels):
#             modules = []
#             modules.append(layer(in_channels, out_channels))
#             return nn.Sequential(*modules)

#         self.denseblock1 = self.make_dense_block(Dense_Block, 64)
#         self.denseblock2 = self.make_dense_block(Dense_Block, 128)
#         self.denseblock3 = self.make_dense_block(Dense_Block, 128)

#         self.transition_layer1 = self.make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)

#         def forward(self, x):
#             out = self.relu(self.conv(x))
#             out = self.denseblock1(out)
#             out = self.transition_layer1(out)
#             out = self.denseblock2(out)
#             out = self.transition_layer2(out)
#             out = self.denseblock3(out)
#             out = self.transition_layer3(out)



        