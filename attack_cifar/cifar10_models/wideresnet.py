import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x, LinBP = 0):
        if(LinBP == 1):
            flag2 = [1,1]
        elif(LinBP == 2):
            flag2 = [0,1]
        elif(LinBP == 3):
            flag2 = [1,0]
        else:
            flag2 = [0,0]
        if not self.equalInOut:
            xx = self.bn1(x)
            if(flag2[0]):
                xx_p = F.relu(-xx)
                x = xx + xx_p.data
                #x = xx - xx.data + F.relu(xx).data
            else:
                x = self.relu1(xx)
        else:
            xx = self.bn1(x)
            if(flag2[0]):
                xx_p = F.relu(-xx)
                out = xx + xx_p.data
                #out = xx - xx.data + F.relu(xx).data
            else:
                out = self.relu1(xx)
        
        out_ = self.bn2(self.conv1(out if self.equalInOut else x))
        if(flag2[1]):
            out__p = F.relu(-out_)
            out = out_ + out__p.data
        else:
            out = self.relu2(out_)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]
        self.name = 'WRN'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x, LinBP = False):
        cnt = 0
        out = self.conv1(x)

        if(LinBP):
            flag = [0,0,0,0,0,0,0,0,0,0,0,0,1]
        else:
            flag = [0,0,0,0,0,0,0,0,0,0,0,0,0]

        for i,basic_blk in enumerate(self.block1.layer):
            out = basic_blk(out, flag[cnt])
            cnt = cnt + 1
            
        
        for i,basic_blk in enumerate(self.block2.layer):
            out = basic_blk(out, flag[cnt])
            cnt = cnt + 1

        for i,basic_blk in enumerate(self.block3.layer):
            out = basic_blk(out, flag[cnt])
            cnt = cnt + 1
        
    
        #out = self.block1(out)
        #out = self.block2(out)
        #out = self.block3(out)
        out = self.bn1(out)
        if(flag[cnt] == 1):
            #out = out - out.data + F.relu(out).data
            out_p = F.relu(-out)
            out = out + out_p.data
        else:
            out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def WRN(pretrained=False, progress=True, device='cpu', **kwargs):
    model = WideResNet(depth=28, widen_factor=10, sub_block1=True)


    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir+'/state_dicts/Zhang2020Geometry.pt', map_location=device)
        #state_dict = state_dict['model_state_dict']
        state_dict = state_dict['state_dict']
        #model.load_state_dict(state_dict)
        
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        

    return model


