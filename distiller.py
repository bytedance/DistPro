# Copyright 2021 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractclassmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np

import math

def hcl(fs, ft):
    n,c,h,w = fs.shape
    loss = F.mse_loss(fs, ft, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
        tmpft = F.adaptive_avg_pool2d(ft, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    return loss

def distillation_loss(source, target):
    loss = hcl(source, target)
    return loss

def build_feature_connector(t_channel, s_channel, out_shape):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel),
         Interpolate(out_shape)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

class Interpolate(nn.Module):
    def __init__(self, out_shape, mode="nearest"):
        super(Interpolate, self).__init__()
        self.out_shape = out_shape
        self.mode = mode
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, self.out_shape, mode=self.mode)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_channel):
        super(SelfAttention, self).__init__()
        self.conv = nn.Conv2d(input_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        att = self.conv(x)
        x = x * att
        return x

def build_feature_connector_2conv(t_channel, s_channel, out_shape):
    mid_channel = np.min([512, t_channel, s_channel])
    C = [nn.Conv2d(s_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(mid_channel),
         SelfAttention(mid_channel),
         Interpolate(out_shape),
         nn.Conv2d(mid_channel, t_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def build_feature_connector_3conv(t_channel, s_channel, out_shape):
    mid_channel = np.min([512, t_channel, s_channel])
    C = [nn.Conv2d(s_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(mid_channel),
         SelfAttention(mid_channel),
         Interpolate(out_shape),
         nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(mid_channel),
         nn.Conv2d(mid_channel, t_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def build_feature_connector_complex(t_channel, s_channel, out_shape):
    mid_channel = np.min([512, t_channel, s_channel])
    C = [nn.Conv2d(s_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(mid_channel),
         nn.ReLU(),
         SelfAttention(mid_channel),
         nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(mid_channel),
         nn.ReLU(),
         Interpolate(out_shape),
         nn.Conv2d(mid_channel, t_channel, kernel_size=3, stride=1, padding=1, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class Distiller(nn.Module):
    def __init__(self, t_net, s_net, alpha_init, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.connections = []
        self.Connectors = []
        for idx_t in range(len(t_channels)):
            for idx_s in range(len(s_channels)):
                if idx_s < idx_t:
                    continue
                self.connections.append([idx_t, idx_s])
                self.Connectors.append(build_feature_connector_complex(t_channels[idx_t], s_channels[idx_s], [32, 32, 16, 8][idx_t]))

        self.Connectors = nn.ModuleList(self.Connectors)
        if alpha_init is not None:
            assert len(alpha_init) == len(self.Connectors), 'wrong alpha_init length'
            self.alpha = np.array(alpha_init)
        else:
            self.alpha = np.ones(len(self.Connectors))
        self.alpha = torch.nn.Parameter(torch.from_numpy(self.alpha).float())
        self.alpha.requires_grad = True

        self.t_net = t_net
        self.s_net = s_net

        self.criterion_CE = nn.CrossEntropyLoss()

        self.args = args

    def forward(self, x):
        with torch.no_grad():
            t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)
        feat_num = len(t_feats)
        if self.args.alpha_normalization_style == 0:
            weights = torch.ones(self.alpha.shape).type(self.alpha.type()) * self.args.kd_weight
        elif self.args.alpha_normalization_style == 1:
            weights = torch.nn.functional.softmax(self.alpha.reshape(-1), dim=0)
        elif self.args.alpha_normalization_style == 2:
            weights = torch.nn.functional.softmax(torch.cat([self.alpha, torch.ones(1).type(self.alpha.type())]).reshape(-1), dim=0)
        elif self.args.alpha_normalization_style == 3:
            weights = torch.sigmoid(self.alpha)
        elif self.args.alpha_normalization_style == 4:
            weights = torch.abs(self.alpha)
        elif self.args.alpha_normalization_style == 5:
            weights = torch.abs(self.alpha)
            weights = weights / (1e-12+weights.sum()) * 3.
        elif self.args.alpha_normalization_style == 6:
            weights = torch.abs(self.alpha)
            weights = weights / (1.+weights.sum()) * 10.
        elif self.args.alpha_normalization_style == 333:
            weights = torch.nn.functional.softmax(torch.cat([self.alpha, torch.ones(1).type(self.alpha.type())]).reshape(-1), dim=0) * self.alpha.reshape(-1).shape[0] * self.args.kd_weight
        else:
            raise ValueError('wrong alpha_normalization_style')
        loss_distill = []
        for i, connection in enumerate(self.connections):
            t_feat = t_feats[connection[0]]
            s_feat = s_feats[connection[1]]
            s_feat = self.Connectors[i](s_feat)
            loss = distillation_loss(s_feat, t_feat) * weights[i]
            loss_distill.append(loss)
        batch_size = x.shape[0]
        loss_distill = sum(loss_distill)
        return s_out, loss_distill

    def _loss(self, input, target, val=False):
        s_out, loss_distill = self.forward(input)
        loss = self.criterion_CE(s_out, target)
        distill_loss_weight = 1. if not val else 0.
        loss += loss_distill * distill_loss_weight
        return loss

    def parameters(self):
        p = []
        for k, v in self.named_parameters():
            p.append(v)
        return p

    def named_parameters(self):
        dic = dict()
        for k, v in super().named_parameters():
            if 't_net' not in k and 'alpha' not in k:
                dic[k] = v
        return dic.items()

    def arch_parameters(self):
        return [self.alpha,]
