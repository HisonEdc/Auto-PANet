"""
    author: He Jiaxin
    date: 14/8/2019
    version: 1.0
    function: generate a PANet according to the sequence generated from RNN controller
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
from torch.nn import functional as F


# 实现给定一组actions，具现化为模型，并可以将图片进行加工
class PathAggregation(nn.Module):
    def __init__(self, resnet, actions, scales):
        super(PathAggregation, self).__init__()
        self.backbone = resnet
        self.actions = actions
        self.scales = scales
        self.dim_out = 256
        self.num_channels = [256, 512, 1024, 2048]
        self.layer1_1, self.layer1_2 = self.build_first_pyramid(actions[0], scales[0], scales[1])
        self.layer2_1, self.layer2_2 = self.build_first_pyramid(actions[2], scales[2], scales[3])
        self.layer3_1, self.layer3_2 = self.build_first_pyramid(actions[4], scales[4], 3)
        self.layer4_1, self.layer4_2 = self.build_first_pyramid(actions[5], scales[5], 2)
        self.layer5_1, self.layer5_2 = self.build_first_pyramid(actions[6], scales[6], 1)
        self.layer6_1, self.layer6_2 = self.build_first_pyramid(actions[7], scales[7], 0)
        self.layer7_1, self.layer7_2 = self.build_second_pyramid(scales[8], scales[9])
        self.layer8_1, self.layer8_2 = self.build_second_pyramid(scales[10], scales[11])
        self.layer9_1, self.layer9_2 = self.build_second_pyramid(scales[12], 0)
        self.layer10_1, self.layer10_2 = self.build_second_pyramid(scales[13], 1)
        self.layer11_1, self.layer11_2 = self.build_second_pyramid(scales[14], 2)
        self.layer12_1, self.layer12_2 = self.build_second_pyramid(scales[15], 3)

        self.mlp = nn.Linear(256*4, 10, bias=False)

    def build_first_pyramid(self, action, scale, output):
        strides = [output - scale[0], output - scale[1]]
        if strides[0] >= 0 and action[0] in [0, 1, 2, 3]:
            component1 = nn.Conv2d(self.num_channels[scale[0]], self.dim_out, 3, pow(2, strides[0]), 1, bias=False)
        elif strides[0] >= 0 and action[0] not in [0, 1, 2, 3]:
            component1 = nn.Conv2d(self.dim_out, self.dim_out, 3, pow(2, strides[0]), 1, bias=False)
        elif strides[0] < 0 and action[0] in [0, 1, 2, 3]:
            component1 = nn.Sequential(
                nn.Upsample(scale_factor=pow(2, -strides[0]), mode='bilinear', align_corners=False),
                nn.Conv2d(self.num_channels[scale[0]], self.dim_out, 1, 1, 0, bias=False)
            )
        else:
            component1 = nn.Sequential(
                nn.Upsample(scale_factor=pow(2, -strides[0]), mode='bilinear', align_corners=False),
                nn.Conv2d(self.dim_out, self.dim_out, 1, 1, 0, bias=False)
            )
        if strides[1] >= 0 and action[1] in [0, 1, 2, 3]:
            component2 = nn.Conv2d(self.num_channels[scale[1]], self.dim_out, 3, pow(2, strides[1]), 1, bias=False)
        elif strides[1] >= 0 and action[1] not in [0, 1, 2, 3]:
            component2 = nn.Conv2d(self.dim_out, self.dim_out, 3, pow(2, strides[1]), 1, bias=False)
        elif strides[1] < 0 and action[1] in [0, 1, 2, 3]:
            component2 = nn.Sequential(
                nn.Upsample(scale_factor=pow(2, -strides[1]), mode='bilinear', align_corners=False),
                nn.Conv2d(self.num_channels[scale[1]], self.dim_out, 1, 1, 0, bias=False)
            )
        else:
            component2 = nn.Sequential(
                nn.Upsample(scale_factor=pow(2, -strides[1]), mode='bilinear', align_corners=False),
                nn.Conv2d(self.dim_out, self.dim_out, 1, 1, 0, bias=False)
            )
        return component1, component2

    def build_second_pyramid(self, scale, output):
        strides = [output - scale[0], output - scale[1]]
        if strides[0] >= 0:
            component1 = nn.Conv2d(self.dim_out, self.dim_out, 3, pow(2, strides[0]), 1, bias=False)
        else:
            component1 = nn.Sequential(
                nn.Upsample(scale_factor=pow(2, -strides[0]), mode='bilinear', align_corners=False),
                nn.Conv2d(self.dim_out, self.dim_out, 1, 1, 0, bias=False)
            )
        if strides[1] >= 0:
            component2 = nn.Conv2d(self.dim_out, self.dim_out, 3, pow(2, strides[1]), 1, bias=False)
        else:
            component2 = nn.Sequential(
                nn.Upsample(scale_factor=pow(2, -strides[1]), mode='bilinear', align_corners=False),
                nn.Conv2d(self.dim_out, self.dim_out, 1, 1, 0, bias=False)
            )
        return component1, component2

    def forward(self, x):
        actions = self.actions
        scales = self.scales
        image_0, image_1, image_2, image_3 = self.backbone(x)
        image_4 = self.layer1_1(locals()['image_%d' % self.actions[0][0]]) + self.layer1_2(locals()['image_%d' % self.actions[0][1]])
        image_5 = self.layer2_1(locals()['image_%d' % self.actions[2][0]]) + self.layer2_2(locals()['image_%d' % self.actions[2][1]])
        image_6 = self.layer3_1(locals()['image_%d' % self.actions[4][0]]) + self.layer3_2(locals()['image_%d' % self.actions[4][1]])
        image_7 = self.layer4_1(locals()['image_%d' % self.actions[5][0]]) + self.layer4_2(locals()['image_%d' % self.actions[5][1]])
        image_8 = self.layer5_1(locals()['image_%d' % self.actions[6][0]]) + self.layer5_2(locals()['image_%d' % self.actions[6][1]])
        image_9 = self.layer6_1(locals()['image_%d' % self.actions[7][0]]) + self.layer6_2(locals()['image_%d' % self.actions[7][1]])
        image_10 = self.layer7_1(locals()['image_%d' % self.actions[8][0]]) + self.layer7_2(locals()['image_%d' % self.actions[8][1]])
        image_11 = self.layer8_1(locals()['image_%d' % self.actions[10][0]]) + self.layer8_2(locals()['image_%d' % self.actions[10][1]])
        image_12 = self.layer9_1(locals()['image_%d' % self.actions[12][0]]) + self.layer9_2(locals()['image_%d' % self.actions[12][1]])
        image_13 = self.layer10_1(locals()['image_%d' % self.actions[13][0]]) + self.layer10_2(locals()['image_%d' % self.actions[13][1]])
        image_14 = self.layer11_1(locals()['image_%d' % self.actions[14][0]]) + self.layer11_2(locals()['image_%d' % self.actions[14][1]])
        image_15 = self.layer12_1(locals()['image_%d' % self.actions[15][0]]) + self.layer12_2(locals()['image_%d' % self.actions[15][1]])

        linear_1 = F.adaptive_avg_pool2d(image_12, (1, 1))
        linear_2 = F.adaptive_avg_pool2d(image_13, (1, 1))
        linear_3 = F.adaptive_avg_pool2d(image_14, (1, 1))
        linear_4 = F.adaptive_avg_pool2d(image_15, (1, 1))
        linear = torch.cat([linear_1, linear_2, linear_3, linear_4], dim=1)
        out = linear.view(linear.size(0), -1)
        out = self.mlp(out)

        return out



