import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.autograd import Function
from roi_pool_cuda import *


class RoIPoolFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.output = None
        self.argmax = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)
        argmax = torch.IntTensor(num_rois, num_channels, self.pooled_height, self.pooled_width).zero_()

        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pool_forward(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                 _features, rois, output)

            # output = output.cuda()
        else:
            output = output.cuda()
            argmax = argmax.cuda()
            roi_pool_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                      features, rois, output, argmax)
            self.output = output
            self.argmax = argmax
            self.rois = rois
            self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()
        roi_pool_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                   grad_output, self.rois, grad_input, self.argmax)

        # print grad_input

        return grad_input, None

