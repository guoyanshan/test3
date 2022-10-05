from backbone import ResBlk
from mutual_information_loss import Mutual_Information_Loss
import torch.nn as nn
import torch
from torch.nn import functional as F


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1_1 = ResBlk(64, 64, stride=1)
        self.blk2_1 = ResBlk(64, 128, stride=1)
        self.blk3_1 = ResBlk(128, 256, stride=1)
        self.blk4_1 = ResBlk(256, 512, stride=1)

        self.blk1_2 = ResBlk(64, 64, stride=1)
        self.blk2_2 = ResBlk(64, 128, stride=1)
        self.blk3_2 = ResBlk(128, 256, stride=1)
        self.blk4_2 = ResBlk(256, 512, stride=1)

        self.blk1_3 = ResBlk(64, 64, stride=1)
        self.blk2_3 = ResBlk(64, 128, stride=1)
        self.blk3_3 = ResBlk(128, 256, stride=1)
        self.blk4_3 = ResBlk(256, 512, stride=1)

        self.blk1_4 = ResBlk(64, 64, stride=1)
        self.blk2_4 = ResBlk(64, 128, stride=1)
        self.blk3_4 = ResBlk(128, 256, stride=1)
        self.blk4_4 = ResBlk(256, 512, stride=1)

        self.mutual_information_loss = Mutual_Information_Loss()
        self.outlayer = nn.Linear(1024, 7)

    def forward(self, ms, ms_up, pan_down_up, pan, phase='train'):
        f_ms_1 = F.relu(self.conv1(ms))
        f_ms_up_1 = F.relu(self.conv2(ms_up))
        f_pan_down_up_1 = F.relu(self.conv3(pan_down_up))
        f_pan_1 = F.relu((self.conv4(pan)))

        f_ms_2 = self.blk1_1(f_ms_1)
        f_ms_up_2 = self.blk1_2(f_ms_up_1)
        f_pan_down_up_2 = self.blk1_3(f_pan_down_up_1)
        f_pan_2 = self.blk1_4(f_pan_1)

        f_ms_3 = self.blk2_1(f_ms_2)
        f_ms_up_3 = self.blk2_2(f_ms_up_2)
        f_pan_down_up_3 = self.blk2_3(f_pan_down_up_2)
        f_pan_3 = self.blk2_4(f_pan_2)

        f_ms_4 = self.blk3_1(f_ms_3)
        f_ms_up_4 = self.blk3_2(f_ms_up_3)
        f_pan_down_up_4 = self.blk3_3(f_pan_down_up_3)
        f_pan_4 = self.blk3_4(f_pan_3)

        f_ms_5 = self.blk4_1(f_ms_4)
        f_ms_up_5 = self.blk4_2(f_ms_up_4)
        f_pan_down_up_5 = self.blk4_3(f_pan_down_up_4)
        f_pan_5 = self.blk4_4(f_pan_4)

        Q = f_ms_up_5
        K = f_pan_down_up_5
        Vp = f_pan_5
        # print(Vp.shape)
        _, _, h1, w1 = Vp.size()
        Vm = F.upsample(f_ms_5, size=(h1, w1), mode='bilinear')
        # print(Vm.shape)


        b, c, h, w = Q.size(0), Q.size(1), Q.size(2), Q.size(3)
        # print(b, c, h, w)

        Q = Q.view(b, c, h * w)
        K = K.view(b, c, h * w)
        Vm = Vm.view(b, c, h * w)
        Vp = Vp.view(b, c, h * w)

        # Compute attention
        attn = torch.matmul(Q, K.transpose(-2, -1))

        # Normalization(Softmax)
        attn = F.softmax(attn, dim=-1)

        # Attention output
        spectral_output = torch.matmul(attn, Vm).view(b, c, h, -1)
        spacial_output = torch.matmul(attn, Vp).view(b, c, h, -1)
        # print(spectral_output.shape)

        out = []
        if phase == 'train':
            # loss
            spectral_loss = self.mutual_information_loss(spectral_output, f_pan_5)
            spacial_loss = self.mutual_information_loss(spacial_output, f_ms_5)
            out.append(spectral_loss+spacial_loss)

        f_fusion = torch.cat([spectral_output, spacial_output], 1)
        s = f_fusion.view(f_fusion.size()[0], -1)
        rel = self.outlayer(s)
        out.append(rel)
        return out
