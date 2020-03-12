""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import numpy as np

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input,mask=None):
        x = self.conv2d(input)
        if mask is None:
            mask = self.mask_conv2d(input)
        else:
            upsample = nn.UpsamplingBilinear2d(size = (input.size()[-2],input.size()[-1]))
            mask = upsample(mask)
            mask = self.mask_conv2d(mask)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x
class noise_filter(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(noise_filter, self).__init__()    
        c=np.zeros((3,3,3))
        c[0]=[[0,0,0],[0,-1,0],[0,1,0]]#,[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        #c[0]=c[0]/12


        c[1][1][1]=-1
        c[1][1][2]=1
        #c[1]=c[1]/4

        c[2][1][1]=-1
        #c[2][2][2]=-2
        #c[2][2][3]=1

        #c[2]=c[2]/2
        c[2][2][2]=1
        Wcnn=np.zeros((out_channels,in_channels,3,3))
        for i in range(in_channels):
          #k=i%10+1
          #Wcnn[i]=[c[3*k-3],c[3*k-2],c[3*k-1]]
          Wcnn[i,0,:,:]=c[i]
          Wcnn[i+3,1,:,:]=c[i]
          Wcnn[i+6,2,:,:]=c[i]
        
          #kernel = tf.get_variable('weights',
                                #shape=[5, 5, 3, 3],
                                #initializer=tf.constant_initializer(c))
        if torch.cuda.is_available():
            self.weight = Variable(torch.from_numpy(Wcnn).cuda().float(), requires_grad=True)
        else:
            self.weight = Variable(torch.from_numpy(Wcnn).float(),  requires_grad=True)
    def forward(self, noise_im):
        conv = F.conv2d(noise_im, self.weight,padding=1)
        return conv

def to_tridiagonal_multidim(w):
    N,W,C,D = w.size()
    tmp_w = w.unsqueeze(2).expand([N,W,W,C,D])

    eye_a = Variable(torch.diag(torch.ones(W-1).cuda(),diagonal=-1))
    eye_b = Variable(torch.diag(torch.ones(W).cuda(),diagonal=0))
    eye_c = Variable(torch.diag(torch.ones(W-1).cuda(),diagonal=1))

    
    tmp_eye_a = eye_a.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
    a = tmp_w[:,:,:,:,0] * tmp_eye_a
    tmp_eye_b = eye_b.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
    b = tmp_w[:,:,:,:,1] * tmp_eye_b
    tmp_eye_c = eye_c.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
    c = tmp_w[:,:,:,:,2] * tmp_eye_c

    return a+b+c
def recursive_filter(feature,tmp, direction):
    N,C,H,W = feature.size()
    rnn_h1 = Variable(torch.zeros((N,C,H,W)).cuda())
    tmp = torch.nn.Sigmoid()(tmp)

    if direction==1:
        #pdb.set_trace()
        #left to right
        for i in range(W):
            
            w_h_prev_1 = tmp[:,:,:,i] * rnn_h1[:,:,:,i-1].clone()
            w_x_curr_1 = (1 - tmp[:,:,:,i]) * feature[:,:,:,i]
            rnn_h1[:,:,:,i] = w_x_curr_1 + w_h_prev_1
    elif direction==2:

        for i in range(W-1,-1,-1):
        #right to left
            if i==W-1:
                w_h_prev_2 = tmp[:,:,:,i] * rnn_h1[:,:,:,0].clone()
                w_x_curr_2 = (1 - tmp[:,:,:,i]) * feature[:,:,:,W -1]
            else:
                w_h_prev_2 = tmp[:,:,:,i] * rnn_h1[:,:,:,i+1].clone()
                w_x_curr_2 = (1 - tmp[:,:,:,i]) * feature[:,:,:, i]
            rnn_h1[:,:,:,i] = w_x_curr_2 + w_h_prev_2  
    elif direction==3:
        # up to down

        for i in range(H):
            w_h_prev_3 = tmp[:,:,i,:] * rnn_h1[:,:,i-1,:].clone()
            w_x_curr_3 = (1 - tmp[:,:,i,:]) * feature[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_3 + w_h_prev_3
    else:
        #down to up
        for i in range(H-1,-1,-1):
            if i == H-1:
                w_h_prev_4 = tmp[:,:,i,:] * rnn_h1[:,:,0,:].clone()
                w_x_curr_4 = (1 - tmp[:,:,i,:]) * feature[:,:,H-1,:]
            else:

                w_h_prev_4 = tmp[:,:,i,:] * rnn_h1[:,:,i+1,:].clone()
                w_x_curr_4 = (1 - tmp[:,:,i,:]) * feature[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_4 + w_h_prev_4
    return rnn_h1

def recursive_filter1(feature,tmp, direction):
    N,C,H,W = feature.size()
    rnn_h1 = Variable(torch.zeros((N,C,H,W)).cuda())
    tmp = torch.nn.Sigmoid()(tmp)

    if direction==1:
        #pdb.set_trace()
        #left to right
        for i in range(W):
            
            w_h_prev_1 = tmp[:,:,:,i-1] * rnn_h1[:,:,:,i-1].clone()
            w_x_curr_1 = (1 - tmp[:,:,:,i-1]) * feature[:,:,:,i]
            rnn_h1[:,:,:,i] = w_x_curr_1 + w_h_prev_1
    elif direction==2:

        for i in range(W-1,-1,-1):
        #right to left
            if i==W-1:
                w_h_prev_2 = tmp[:,:,:,0] * rnn_h1[:,:,:,0].clone()
                w_x_curr_2 = (1 - tmp[:,:,:,0]) * feature[:,:,:,W -1]
            else:
                w_h_prev_2 = tmp[:,:,:,i+1] * rnn_h1[:,:,:,i+1].clone()
                w_x_curr_2 = (1 - tmp[:,:,:,i+1]) * feature[:,:,:, i]
            rnn_h1[:,:,:,i] = w_x_curr_2 + w_h_prev_2  
    elif direction==3:
        # up to down

        for i in range(H):
            w_h_prev_3 = tmp[:,:,i-1,:] * rnn_h1[:,:,i-1,:].clone()
            w_x_curr_3 = (1 - tmp[:,:,i-1,:]) * feature[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_3 + w_h_prev_3
    else:
        #down to up
        for i in range(H-1,-1,-1):
            if i == H-1:
                w_h_prev_4 = tmp[:,:,0,:] * rnn_h1[:,:,0,:].clone()
                w_x_curr_4 = (1 - tmp[:,:,0,:]) * feature[:,:,H-1,:]
            else:

                w_h_prev_4 = tmp[:,:,i+1,:] * rnn_h1[:,:,i+1,:].clone()
                w_x_curr_4 = (1 - tmp[:,:,i+1,:]) * feature[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_4 + w_h_prev_4
    return rnn_h1

def recursive_filter_edge(feature,edge, direction):
    N,C,H,W = feature.size()
    #pdb.set_trace()
    #rnn_h1 = Variable(torch.zeros((N,C,H,W)).cuda())
    rnn_h1 = feature
    #tmp = torch.nn.Sigmoid()(tmp)
    tmp = torch.exp(-np.sqrt(2)*(1+edge*100/1)/100)
    if direction==1:
        #pdb.set_trace()
        #left to right
        for i in range(W):
            
            w_h_prev_1 = tmp[:,:,:,i] * rnn_h1[:,:,:,i-1].clone()
            w_x_curr_1 = (1 - tmp[:,:,:,i]) * edge[:,:,:,i]
            rnn_h1[:,:,:,i] = w_x_curr_1 + w_h_prev_1
    elif direction==2:

        for i in range(W-1,-1,-1):
        #right to left
            if i==W-1:
                w_h_prev_2 = tmp[:,:,:,i] * rnn_h1[:,:,:,0].clone()
                w_x_curr_2 = (1 - tmp[:,:,:,i]) * edge[:,:,:,W -1]
            else:
                w_h_prev_2 = tmp[:,:,:,i] * rnn_h1[:,:,:,i+1].clone()
                w_x_curr_2 = (1 - tmp[:,:,:,i]) * edge[:,:,:, i]
            rnn_h1[:,:,:,i] = w_x_curr_2 + w_h_prev_2  
    elif direction==3:
        # up to down

        for i in range(H):
            w_h_prev_3 = tmp[:,:,i,:] * rnn_h1[:,:,i-1,:].clone()
            w_x_curr_3 = (1 - tmp[:,:,i,:]) * edge[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_3 + w_h_prev_3
    else:
        #down to up
        for i in range(H-1,-1,-1):
            if i == H-1:
                w_h_prev_4 = tmp[:,:,i,:] * rnn_h1[:,:,0,:].clone()
                w_x_curr_4 = (1 - tmp[:,:,i,:]) * edge[:,:,H-1,:]
            else:

                w_h_prev_4 = tmp[:,:,i,:] * rnn_h1[:,:,i+1,:].clone()
                w_x_curr_4 = (1 - tmp[:,:,i,:]) * edge[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_4 + w_h_prev_4
    return rnn_h1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            #nn.MaxPool2d(1),
            #DoubleConv(in_channels, out_channels)
            #GatedConv2dWithActivation(in_channels, out_channels),
            #GatedConv2dWithActivation(out_channels, out_channels),
        #)

        self.maxpool = nn.MaxPool2d(2)
        self.gated1 = GatedConv2dWithActivation(in_channels, out_channels)
        self.gated2 = GatedConv2dWithActivation(out_channels, out_channels)

    def forward(self, x, mask=None):
        
        x = self.maxpool(x)
        x= self.gated1(x,mask)
        x = self.gated2(x,mask)
        return x
        #return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
