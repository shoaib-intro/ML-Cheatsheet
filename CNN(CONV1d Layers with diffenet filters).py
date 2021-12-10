
###############################################################
#################   Conv1D layer with different fillters ######
#################                                        ######
###############################################################

import torch.nn as nn
import numpy as np
import torch


# How does convolution work? (Kernel size = 1)

class TestConv1d(nn.Module):
    def __init__(self):
        super(TestConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        self.init_weights()

    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)


in_x = torch.tensor([[[1,2,3,4,5,6]]]).float()
print("in_x.shape", in_x.shape)
print(in_x)

net = TestConv1d()
out_y = net(in_x)

print("out_y.shape", out_y.shape)
print(out_y)


# Effect of kernel size (Kernel size = 2)
class TestConv1D(nn.Module):
    def __init__(self):
        super(TestConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
        self.init_weights()
    
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):

        new_weights = torch.ones(self.conv.weight.shape) * 2.0
        print(new_weights)
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)

        
in_x = torch.tensor([[[1,2,3,4,5,6]]]).float()
net = TestConv1D()
y_out = net(in_x)

print(in_x)
print(y_out, len(y_out))


# Effect of kernel size (Kernel size = 3)

class testConv1d(nn.Module):
    def __init__(self):
        super(testConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        self.init_weights()
    
    def forward(self, x):
        return self.conv(x)
    
    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)

in_x = torch.tensor([[[1,2,3,4,5,6]]]).float()
net = testConv1d()
y_out = net(in_x)

print(in_x, in_x.shape)
print(y_out, y_out.shape)



# How to produce an output vector of the same size? (Padding)

class testConv1d(nn.Module):
    def __init__(self):
        super(testConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.init_weight()
    
    def forward(self, x):
        return self.conv(x)
    
    def init_weight(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)

in_x = torch.tensor([[[1,2,3,4,5,6]]]).float()
net = testConv1d()
y_out = net(in_x)

print(in_x, in_x.shape)
print(y_out.int(), y_out.shape)



# Dilated convolutions “inflate” the kernel by inserting spaces between the kernel elements, and a parameter controls the dilation rate.
# A dilation rate of 2 means there is a space between the kernel elements. 
# Essentially, a convolution kernel with dilation = 1 corresponds to a regular convolution.


class testConv1d(nn.Module):
    def __init__(self):
        super(testConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,  dilation=2)
        self.init_weight()
    
    def forward(self, x):
        return self.conv(x)
    
    def init_weight(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)

in_x = torch.tensor([[[1,2,3,4,5,6]]]).float()
net = testConv1d()
y_out = net(in_x)

print(in_x, in_x.shape)
print(y_out, y_out.shape)



# Separate the weights (Groups)
# By default, the “groups” parameter is set to 1, where all the inputs channels are convolved to all outputs.
# this will force the training to split the input vector’s channels into different groupings of features.

class testConv1d(nn.Module):
    def __init__(self):
        super(testConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1, groups=2)
        self.init_weights()
    
    def forward(self,x):
        return self.conv(x)
    
    def init_weights(self):
        print(self.conv.weight.shape)
        self.conv.weight[0,0,0] = 2.
        self.conv.weight[1,0,0] = 4.

in_x = torch.tensor([[[1,2,3,4,5,6],[10,20,30,40,50,60]]]).float()
net = testConv1d()
y_out = net(in_x)

print(in_x, in_x.shape)
print(y_out, y_out.shape)




# Refference
# https://jinglescode.github.io/2020/11/01/how-convolutional-layers-work-deep-learning-neural-networks/

