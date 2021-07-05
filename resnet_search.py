import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from auto_augment import AutoAugment
import numpy as np
# CHANNELS = [64, 128, 256, 512]
# CHANNELS = [128, 256, 512, 768]
# CHANNELS = [32, 64, 128, 192]

# CHANNELS = [16, 32, 64, 128]
# CHANNELS = [8, 16, 32, 64]
# CHANNELS = [16, 32, 48, 64]
CHANNELS = [16,24,32,40,44,48,56,64]
CHANNELS = [40,40,48,48,56,56,64,64]
CHANNELS = [32,36,40,44,48,56,60,64]
CHANNELS = [40,56]

CHANNELS = [[40,44],
            [44,48],
            [48,52],
            [40,44]]


# CHANNELS = [[40,44],
#             [44,48],
#             [48,52],
#             [52,56]]




# CHANNELS = [44,48,52,56]
# CHANNELS = [56,56]


# CHANNELS = [56,56,56,56,56,56,56,56]

# CHANNELS = [8, 16, 24, 32, 48, 56, 64, 72]
# CHANNELS = [48, 52, 56, 64]

# RESOLUTION = [1, 2, 3, 4]
RESOLUTION = [1,1,1,1,1.5,1.5,2,2]
RESOLUTION = [1,2]
# RESOLUTION = [1,1.5,2,2.5]

AUG = [0,0,0,0,1,1,1,1]
AUG = [0,1]
# AUG = [0,0,1,1]

# LAYERS = [[1,2,3,4],
#           [2,4,6,8],
#           [2,4,6,8],
#           [1,2,3,4]]

LAYERS = [[1,2,3,4,5,6,7,8],
          [1,2,3,4,5,6,7,8],
          [1,2,3,4,5,6,7,8],
          [1,2,3,4,5,6,7,8]]


LAYERS = [[4,6],
          [4,6],
          [4,6],
          [4,6]]

LAYERS = [[2,4],
          [4,6],
          [4,6],
          [2,4]]

# LAYERS = [[2,4],
#           [4,6],
#           [4,6],
#           [2,4]]

# LAYERS = [[2,4,6,8],
#           [2,4,6,8],
#           [2,4,6,8],
#           [2,4,6,8]]



# LAYERS = [[3,3],
#           [4,4],
#           [7,7],
#           [2,2]]

# LAYERS = [[3,3,3,3,3,3,3,3],
#           [4,4,4,4,4,4,4,4],
#           [7,7,7,7,7,7,7,7],
#           [2,2,2,2,2,2,2,2]]

# LAYERS = [[2,4,6,8,10,12,14,16],
#           [2,4,6,8,10,12,14,16],
#           [2,4,6,8,10,12,14,16],
#           [2,4,6,8,10,12,14,16]]
#
# LAYERS = [[2,4,6,8,2,4,6,8],
#           [2,4,6,8,2,4,6,8],
#           [2,4,6,8,2,4,6,8],
#           [2,4,6,8,2,4,6,8]]



# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def normalization(data, dim):
    _range = torch.max(data, dim)[0] - torch.min(data, dim)[0]
    _range = _range.unsqueeze(-1).expand(-1, -1, data.size(-1))
    return (data - torch.min(data, dim)[0].unsqueeze(-1).expand(-1, -1, data.size(-1))) / _range

class conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.stride = stride
        self.padding = dilation

    def forward(self, x):
        in_planes = x.size(1)
        filter1 = self.conv.weight[:, :in_planes, :, :]
        out = F.conv2d(x, filter1, None, stride=self.stride, padding=self.padding)
        return out


class conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.stride = stride

    def forward(self, x):
        in_planes = x.size(1)
        filter1 = self.conv.weight[:, :in_planes, :, :]
        out = F.conv2d(x, filter1, None, stride=self.stride)
        return out

class MixedOp_conv(nn.Module):

    def __init__(self, in_planes):
        super(MixedOp_conv, self).__init__()
        self._ops = nn.ModuleList()
        for stride in range(1,3):
            for C in CHANNELS[0]:
                op = nn.Sequential(nn.Conv2d(in_planes, C, kernel_size=7, stride=stride, padding=3, bias=False),
                                   nn.BatchNorm2d(C),
                                   nn.ReLU())
                self._ops.append(op)

    def forward(self, x, stride, path):

        return self._ops[(stride-1)*len(CHANNELS[0])+path](x)


class MixedOp_fc(nn.Module):

    def __init__(self, num_classes):
        super(MixedOp_fc, self).__init__()
        self._ops = nn.ModuleList()
        for C in CHANNELS[-1]:
            op = nn.Sequential(nn.Linear(C, num_classes))
            self._ops.append(op)

    def forward(self, x, path):

        return self._ops[path](x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.swish = Swish()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # self.se = SELayer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.swish(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # print(out.shape)
        # print(identity.shape)
        out += identity
        out = self.relu(out)
        # out = self.swish(out)
        # out = self.se(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_search(nn.Module):

    def __init__(self, block=BasicBlock, in_planes = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_search, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.inplanes = CHANNELS[0][-1]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.ModuleList()
        # self.conv1.append(nn.Conv2d(in_planes, self.inplanes, kernel_size=7, stride=1, padding=3,
        #                        bias=False))
        # self.conv1.append(nn.Conv2d(in_planes, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                             bias=False))

        self.conv1 = nn.ModuleList()
        self.conv1.append(MixedOp_conv(in_planes))
        self.conv1.append(MixedOp_conv(in_planes))
        self.conv1.append(MixedOp_conv(in_planes))
        self.conv1.append(MixedOp_conv(in_planes))


        # self.conv1 = MixedOp_conv(in_planes)
        self.maxpool = nn.ModuleList()
        self.maxpool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.maxpool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.maxpool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.maxpool.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # self.layer1 = self._make_layer(block, 0)
        self.layer1 = nn.ModuleList()
        self.layer1.append(self._make_layer(block, 0))
        self.layer1.append(self._make_layer(block, 0))
        self.layer1.append(self._make_layer(block, 0))
        self.layer1.append(self._make_layer(block, 0))

        self.layer2 = nn.ModuleList()
        self.layer2.append(self._make_layer(block, 1, stride=2, dilate=replace_stride_with_dilation[0]))
        self.layer2.append(self._make_layer(block, 1, stride=2, dilate=replace_stride_with_dilation[0]))
        self.layer2.append(self._make_layer(block, 1, stride=2, dilate=replace_stride_with_dilation[0]))
        self.layer2.append(self._make_layer(block, 1, stride=2, dilate=replace_stride_with_dilation[0]))

        self.layer3 = nn.ModuleList()
        self.layer3.append(self._make_layer(block, 2, stride=2, dilate=replace_stride_with_dilation[1]))
        self.layer3.append(self._make_layer(block, 2, stride=2, dilate=replace_stride_with_dilation[1]))
        self.layer3.append(self._make_layer(block, 2, stride=2, dilate=replace_stride_with_dilation[1]))
        self.layer3.append(self._make_layer(block, 2, stride=2, dilate=replace_stride_with_dilation[1]))

        self.layer4 = nn.ModuleList()
        self.layer4.append(self._make_layer(block, 3, stride=2, dilate=replace_stride_with_dilation[2]))
        self.layer4.append(self._make_layer(block, 3, stride=2, dilate=replace_stride_with_dilation[2]))
        self.layer4.append(self._make_layer(block, 3, stride=2, dilate=replace_stride_with_dilation[2]))
        self.layer4.append(self._make_layer(block, 3, stride=2, dilate=replace_stride_with_dilation[2]))

        self.avgpool = nn.ModuleList()
        self.avgpool.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.avgpool.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.avgpool.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.avgpool.append(nn.AdaptiveAvgPool2d((1, 1)))



        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.ModuleList()
        self.fc.append(MixedOp_fc(num_classes))
        self.fc.append(MixedOp_fc(num_classes))
        self.fc.append(MixedOp_fc(num_classes))
        self.fc.append(MixedOp_fc(num_classes))



        # self.conv1 = MixedOp_conv(in_planes)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        # self.layer1 = self._make_layer(block, 0)
        # self.layer2 = self._make_layer(block, 1, stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 2, stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 3, stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = MixedOp_fc(num_classes)

        self.train_transform = transforms.Compose([

            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            # transforms.Normalize(MEAN, STD),
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, layer_id, stride=1, dilate=False):
        _ops = nn.ModuleList()
        for i in range(len(LAYERS[0])):
            for j in range(len(CHANNELS[0])):
                planes = CHANNELS[layer_id][j]
                blocks = LAYERS[layer_id][i]
                # if layer_id == 0:
                #     self.inplanes = 64
                norm_layer = self._norm_layer
                downsample = None
                previous_dilation = self.dilation
                if dilate:
                    self.dilation *= stride
                    stride = 1

                if layer_id > 0:
                    self.inplanes = CHANNELS[layer_id-1][-1]
                # if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))

                self.inplanes = CHANNELS[layer_id][-1]
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))
                _ops.append(nn.Sequential(*layers))

        return _ops

    def forward(self, x, paths=[1,0,0,1,0,1,2,3,0]):
        resolution = int(x.shape[2]*RESOLUTION[paths[8]])
        # if paths[-1] > 2:
        #     stride = 1
        # else:
        #     stride = 0
        stride = 1


        # print(resolution)
        # x = F.interpolate(x, size=[resolution, resolution], mode='bilinear', align_corners=True)

        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        # print(x.shape)
        x1 = normalization(x.view(x.size(0), x.size(1), -1), 2)
        x2 = x.clone()
        # x2 = torch.zeros([x.size(0), 3, x.size(2), x.size(3)]).cuda()

        # print(x1.shape)
        if self.training:
            x1 = x1.view(x.shape).permute(0, 2, 3, 1)
            x1 = x1.cpu().numpy()
            x1 = (x1 * 255).astype(np.uint8)
            for i in range(x1.shape[0]):
                # print(x1[i,:,:,:].shape)
                # if x.shape[1] == 1:
                #     im = Image.fromarray(x1[i,:,:,0], mode='L').convert('RGB')
                # else:
                im = Image.fromarray(x1[i, :, :, :])
                x2[i] = self.train_transform(im)
            # print(x.shape)
            x = x2.cuda()
        else:
            x = x1.view(x.shape)

        # index = RESOLUTION[paths[8]] - 1 + (AUG[paths[9]])*2
        index = 0


        x = self.conv1[index](x, stride, paths[4])
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool[index](x)
        # print(paths)
        x = self.layer1[index][paths[0]*len(CHANNELS[0])+paths[4]](x)
        x = self.layer2[index][paths[1]*len(CHANNELS[0])+paths[5]](x)
        x = self.layer3[index][paths[2]*len(CHANNELS[0])+paths[6]](x)
        x = self.layer4[index][paths[3]*len(CHANNELS[0])+paths[7]](x)

        x = self.avgpool[index](x)
        x = torch.flatten(x, 1)
        x = self.fc[index](x, paths[7])

        return x




