# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import cv2
import numpy as np
BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        head_conv = 256

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers_1 = self._make_deconv_layer(256, 4)
        self.deconv_layers_2 = self._make_deconv_layer(128, 4)
        self.deconv_layers_3 = self._make_deconv_layer(64, 4)

        self.heads = heads
        self.heads["regression_hp"] = 20
        self.heads["regression_2dbox"] = 4
        self.heads["regression_3dbox"] = 12
        self.hm = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 3, kernel_size=1, stride=1, padding=0)
        )
        self.hm_hp = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 9, kernel_size=1, stride=1, padding=0)
        )
        self.regression_hp = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 20, kernel_size=1, stride=1, padding=0)
        )
        self.regression_2dbox = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 4, kernel_size=1, stride=1, padding=0)
        )
        self.regression_3dbox = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 12, kernel_size=1, stride=1, padding=0)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):
            layers = []
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels)
            planes = num_filters
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        up_level16 = self.deconv_layers_1(x)
        up_level8 = self.deconv_layers_2(up_level16)
        up_level4 = self.deconv_layers_3(up_level8)

        ret = {}
        ret["hm"] = self.hm(up_level4)
        ret["hm_hp"] = self.hm_hp(up_level4)
        regression_hp = self.regression_hp(up_level4)
        regression_2dbox = self.regression_2dbox(up_level4)
        regression_3dbox = self.regression_3dbox(up_level4)
        # {'hm': 3, 'wh': 2, 'hps': 18, 'rot': 8, 'dim': 3, 'prob': 1, 'reg': 2, 'hm_hp': 9, 'hp_offset': 2}
        ret["reg"] = regression_2dbox[:, :2, ...]
        ret["wh"] = regression_2dbox[:, 2:, ...]
        ret["hp_offset"] = regression_hp[:, :2, ...]
        ret["hps"] = regression_hp[:, 2:, ...]
        ret["rot"] = regression_3dbox[:, :8, ...]
        ret["dim"] = regression_3dbox[:, 8:11, ...]
        ret["prob"] = regression_3dbox[:, 11, ...].unsqueeze(1)
        return [ret]

    def draw_features(self,width, height, x, savename):

        fig = plt.figure(figsize=(40, 16))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.00001, hspace=0.00003)
        for i in range(width * height):
            plt.subplot(height, width, i + 1)
            plt.axis('off')
            img = x[0, i, :, :]
            pmin = np.min(img)
            pmax = np.max(img)
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
            img = img.astype(np.uint8)  # 转成unit8
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
            img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
            plt.imshow(img)
            print("{}/{}".format(i, width * height))
        fig.savefig(savename, dpi=200)
        fig.clf()
        plt.close()

    def init_deconv(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_conv_weight(self, layers, head):
        for i, m in enumerate(layers.modules()):
            if isinstance(m, nn.Conv2d):
                if 'hm' in head or 'hm_hp' in head:
                    if m.weight.shape[0] == self.heads[head]:
                        nn.init.constant_(m.bias, -2.19)
                else:
                    if m.weight.shape[0] == self.heads[head]:
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)
    
    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            self.init_deconv(self.deconv_layers_1)
            self.init_deconv(self.deconv_layers_2)
            self.init_deconv(self.deconv_layers_3)

            self._init_conv_weight(self.hm, 'hm')
            self._init_conv_weight(self.hm_hp, 'hm_hp')
            self._init_conv_weight(self.regression_hp, 'regression_hp')
            self._init_conv_weight(self.regression_2dbox, 'regression_2dbox')
            self._init_conv_weight(self.regression_3dbox, 'regression_3dbox')
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers, pretrained=True)
  return model
