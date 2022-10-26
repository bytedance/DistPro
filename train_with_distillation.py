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

from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import sys

import distiller
import load_settings

from distpro import DistPro

def get_parser():
    parser = argparse.ArgumentParser(description='CIFAR-100 training')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--paper_setting', default='a', type=str)
    parser.add_argument('--epochs', default=240, type=int, help='number of total epochs to run')
    parser.add_argument('--seed', default=1024, type=int, help='random seed')
    parser.add_argument('--val_ratio', default=0.2, type=float, help='ratio of samples used for validation')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_steps', default=[150, 180, 210], nargs='+', type=int, help='steps where to decrease the lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--arch_lr', default=0.1, type=float, help='initial learning rate for searching')
    parser.add_argument('--arch_weight_decay', default=0., type=float, help='weight decay for searching')
    parser.add_argument('--alpha', nargs='+', type=float, help='alpha init')
    parser.add_argument('--noDist', dest='noDist', action='store_false')
    parser.set_defaults(Dist=True)
    parser.add_argument('--kd_weight', default=1., type=float, help='kd weight')
    parser.add_argument('--alpha_normalization_style', default=1, type=int, help='normalization method for alpha')
    parser.add_argument('--redirect', dest='redirect', action='store_true')
    parser.set_defaults(redirect=False)
    return parser.parse_args()

 def train_with_distill(d_net, epoch):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.Dist:
            (inputs_val, targets_val) = next(iter(valloader))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            if args.Dist:
                inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()

        if args.Dist:
            dtp.step(inputs, targets, inputs_val, targets_val, optimizer.param_groups[0]['lr'], optimizer, unrolled=True)

        optimizer.zero_grad()

        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)
        loss_CE = d_net.criterion_CE(outputs, targets)

        loss = loss_CE + loss_distill.sum()

        loss.backward()
        optimizer.step()

        train_loss += loss_CE.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total



def main(args):

    if args.paper_setting in ['i', 'k', 'l']:
        #'shuffle' in args.model or 'mobile' in args.model:
        args.lr = 0.02

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    if not os.path.exists(args.log):
        os.mkdir(args.log)
    os.system('cp models/ -r %s'%args.log)
    os.system('cp *.py %s'%args.log)
    with open(args.log+'/cmd.txt', 'a') as f:
        f.write(' '.join(sys.argv))

    if args.redirect:
        sys.stdout = open(args.log+'/stdout.txt', 'w')
        sys.stderr = open(args.log+'/stderr.txt', 'w')

    gpu_num = 0
    use_cuda = torch.cuda.is_available()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                            np.array([63.0, 62.1, 66.7]) / 255.0)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    indices = np.random.permutation(len(trainset))
    split = int(np.floor((1-args.val_ratio) * len(trainset)))
    if not args.Dist:
        split = len(trainset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))
    valloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]))
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    alpha_init = args.alpha

    # Model
    t_net, s_net, args = load_settings.load_paper_settings(args)

    # Module for distillation
    d_net = distiller.Distiller(t_net, s_net, alpha_init, args)
    criterion_CE = d_net.criterion_CE

    dtp = DistPro(d_net, args)

    print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
    print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

    if use_cuda:
        torch.cuda.set_device(0)
        d_net.cuda()
        s_net.cuda()
        t_net.cuda()
        cudnn.benchmark = True

    print('Performance of teacher network')
    test(t_net)

    best_accuracy = 0.
    lr = args.lr
    print('Base learning rate: ', args.lr)
    for epoch in range(args.epochs):
        if epoch is 0:
            optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        elif epoch in args.lr_steps:
            lr = lr / 10
            optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                                lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

        train_loss = train_with_distill(d_net, epoch)
        test_loss, accuracy = test(s_net)
        torch.save(d_net.state_dict()['alpha'].detach().cpu(), args.log+'/alpha_epoch_%d.pth.tar'%epoch)
        if accuracy > best_accuracy:
            torch.save(s_net.state_dict(), args.log+'/best_model.pth.tar')
            torch.save(d_net.state_dict(), args.log+'/d_net_best_model.pth.tar')
            best_accuracy = accuracy
        with open(args.log+'/log.txt', 'a') as f:
            f.write('%lf\n'%accuracy)

    if args.redirect:
        sys.stdout.close()
        sys.stderr.close()


if __name__ == "__main__":
    args = get_parser()
    main(args)



