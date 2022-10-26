import torch
import os

import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN
from models.ResNet_KR import build_resnet_backbone, build_resnetx4_backbone
from models.shufflenetv1_KR import ShuffleV1
from models.shufflenetv2_KR import ShuffleV2

def load_paper_settings(args):

    WRN_path = os.path.join(args.data_path, 'WRN28-4_21.09.pt')
    Pyramid_path = os.path.join(args.data_path, 'pyramid200_mixup_15.6.tar')

    if args.paper_setting == 'a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=28, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'c':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'd':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=56, num_classes=100)

    elif args.paper_setting == 'e':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'f':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = PYN.PyramidNet(depth=110, alpha=84, num_classes=100, bottleneck=False)

    elif args.paper_setting == 'g':
        teacher = WRN.WideResNet(depth=40, widen_factor=2, num_classes=100)
        state = torch.load('checkpoints/cifar100_wrn-40-2__baseline1_best.pt', map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'h':
        teacher = WRN.WideResNet(depth=40, widen_factor=2, num_classes=100)
        state = torch.load('checkpoints/cifar100_wrn-40-2__baseline1_best.pt', map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=40, widen_factor=1, num_classes=100)

    elif args.paper_setting == 'i':
        teacher = WRN.WideResNet(depth=40, widen_factor=2, num_classes=100)
        state = torch.load('checkpoints/cifar100_wrn-40-2__baseline1_best.pt', map_location={'cuda:0': 'cpu'})
        teacher.load_state_dict(state)
        student = ShuffleV1(num_classes=100)

    elif args.paper_setting == 'j':
        teacher = build_resnetx4_backbone(depth = 32, num_classes=100)
        state = torch.load('checkpoints/cifar100_resnet32x4__baseline5_best.pt', map_location={'cuda:0': 'cpu'})        
        teacher.load_state_dict(state)
        student = build_resnetx4_backbone(depth = 8, num_classes=100)

    elif args.paper_setting == 'k':
        teacher = build_resnetx4_backbone(depth = 32, num_classes=100)
        state = torch.load('checkpoints/cifar100_resnet32x4__baseline5_best.pt', map_location={'cuda:0': 'cpu'})        
        teacher.load_state_dict(state)
        student = ShuffleV1(num_classes=100)

    elif args.paper_setting == 'l':
        teacher = build_resnetx4_backbone(depth = 32, num_classes=100)
        state = torch.load('checkpoints/cifar100_resnet32x4__baseline5_best.pt', map_location={'cuda:0': 'cpu'})        
        teacher.load_state_dict(state)
        student = ShuffleV2(num_classes=100)

    elif args.paper_setting == 'm':
        teacher = build_resnet_backbone(depth = 56, num_classes=100)
        state = torch.load('checkpoints/cifar100_resnet56__baseline5_best.pt', map_location={'cuda:0': 'cpu'})        
        teacher.load_state_dict(state)
        student = build_resnet_backbone(depth = 20, num_classes=100)

    elif args.paper_setting == 'n':
        teacher = build_resnet_backbone(depth = 110, num_classes=100)
        state = torch.load('checkpoints/cifar100_resnet110__baseline3_best.pt', map_location={'cuda:0': 'cpu'})        
        teacher.load_state_dict(state)
        student = build_resnet_backbone(depth = 32, num_classes=100)

    else:
        print('Undefined setting name !!!')

    return teacher, student, args
