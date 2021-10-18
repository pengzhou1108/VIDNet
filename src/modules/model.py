import torch
import torch.nn as nn
from .clstm import ConvLSTMCell, ConvLSTMCellMask
import argparse
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn as nn
import math
from .vision import VGG16, ResNet34, ResNet50, ResNet101,VGG16_BN
import sys
from .aspp import build_aspp, _ASPPModule
import pdb
sys.path.append("..")
from utils.utils import get_skip_dims
from .unet_parts import *
import torch.nn.functional as F
def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)
    return grid
class WarpingLayer(nn.Module):
    
    def __init__(self):
        super(WarpingLayer, self).__init__()
    
    def forward(self, x, flow):
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow).cuda()
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)

        grid = (get_grid(x).cuda() + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp
class FeatureExtractor_segment(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self,args):
        super(FeatureExtractor_segment,self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)

        if args.base_model == 'resnet34':
            self.base = ResNet34()
            self.base.load_state_dict(models.resnet34(pretrained=True).state_dict())
        elif args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
        elif args.base_model == 'resnet101':
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        elif args.base_model == 'vgg16':
            self.base = VGG16()
            self.base.load_state_dict(models.vgg16(pretrained=True).state_dict())
        else:
            raise Exception("The base model you chose is not supported !")

        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.padding = 0 if self.kernel_size == 1 else 1


        self.sk5 = nn.Conv2d(skip_dims_in[0],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2],int(self.hidden_size/2),self.kernel_size,padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3],int(self.hidden_size/4),self.kernel_size,padding=self.padding)

        self.bn5 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn4 = nn.BatchNorm2d(int(self.hidden_size))
        self.bn3 = nn.BatchNorm2d(int(self.hidden_size/2))
        self.bn2 = nn.BatchNorm2d(int(self.hidden_size/4))

    def forward(self,x,semseg=False, raw = False):
        x5,x4,x3,x2,x1 = self.base(x)
        #pdb.set_trace()

        if semseg:
            return x5
        elif raw:
            return x5, x4, x3, x2, x1
        else:
            #return total_feats
            x5_skip = self.bn5(self.sk5(x5))
            x4_skip = self.bn4(self.sk4(x4))
            x3_skip = self.bn3(self.sk3(x3))
            x2_skip = self.bn2(self.sk2(x2))
            del x5, x4, x3, x2, x1, x
            return x5_skip, x4_skip, x3_skip, x2_skip
class FeatureExtractor(nn.Module):
    '''
    Returns base network to extract visual features from image
    '''
    def __init__(self,args):
        super(FeatureExtractor,self).__init__()
        skip_dims_in = get_skip_dims(args.base_model)

        if args.base_model == 'resnet34':
            self.base = ResNet34()
            aa = models.resnet34(pretrained=True).state_dict()
            if args.input_dim>3:
                
                self.base.conv1.weight.data = aa['conv1.weight'].repeat(1,args.input_dim//3,1,1)
                
                del aa['conv1.weight']
                
            self.base.load_state_dict(aa,strict=False)

            self.noise_filter = noise_filter()
        elif args.base_model == 'resnet50':
            self.base = ResNet50()
            self.base.load_state_dict(models.resnet50(pretrained=True).state_dict())
            
        elif args.base_model == 'resnet101':
            self.base = ResNet101()
            self.base.load_state_dict(models.resnet101(pretrained=True).state_dict())
        elif args.base_model == 'vgg16':
            self.base = VGG16()
            self.base1 = VGG16()
            aa = models.vgg16(pretrained=True).state_dict()
            
            if args.input_dim>3:
                self.base.features[0].weight.data = aa['features.0.weight'].repeat(1,2,1,1)
                self.base.features[0].bias.data = aa['features.0.bias']
                del aa['features.0.weight']
                del aa['features.0.bias']
            self.base.load_state_dict(aa,strict=False)
            self.base1.load_state_dict(aa,strict=False)
        elif args.base_model == 'vgg16_bn':
            self.base = VGG16_BN()
            self.base1 = VGG16_BN()
            aa = models.vgg16_bn(pretrained=True).state_dict()
            
            if args.input_dim>3:
                self.base.features[0].weight.data = aa['features.0.weight'].repeat(1,2,1,1)
                self.base.features[0].bias.data = aa['features.0.bias']
                del aa['features.0.weight']
                del aa['features.0.bias']
            self.base.load_state_dict(aa,strict=False)
            self.base1.load_state_dict(aa, strict=False)
        else:
            raise Exception("The base model you chose is not supported !")

        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.padding = 0 if self.kernel_size == 1 else 1

        

        self.sk5 = nn.Conv2d(skip_dims_in[0],int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk4 = nn.Conv2d(skip_dims_in[1]*2,int(self.hidden_size),self.kernel_size,padding=self.padding)
        self.sk3 = nn.Conv2d(skip_dims_in[2]*2,int(self.hidden_size/2),self.kernel_size,padding=self.padding)
        self.sk2 = nn.Conv2d(skip_dims_in[3]*2,int(self.hidden_size/4),self.kernel_size,padding=self.padding)
        self.sk1 = nn.Conv2d(skip_dims_in[4]*2,int(self.hidden_size/8),self.kernel_size,padding=self.padding)

        # BN version
        #self.bn5 = nn.BatchNorm2d(int(self.hidden_size))
        #self.bn4 = nn.BatchNorm2d(int(self.hidden_size))
        #self.bn3 = nn.BatchNorm2d(int(self.hidden_size/2))
        #self.bn2 = nn.BatchNorm2d(int(self.hidden_size/4))
        #self.bn1 = nn.BatchNorm2d(int(self.hidden_size/8))

        #IN version
        self.bn5 = nn.InstanceNorm2d(int(self.hidden_size))
        self.bn4 = nn.InstanceNorm2d(int(self.hidden_size))
        self.bn3 = nn.InstanceNorm2d(int(self.hidden_size/2))
        self.bn2 = nn.InstanceNorm2d(int(self.hidden_size/4))
        self.bn1 = nn.InstanceNorm2d(int(self.hidden_size/8))




    def forward(self,x,semseg=False, raw = False, x_ela=None):
        # HPF loss pass filter
        #x5,x4,x3,x2,x1 = self.base(x[:,:3,:,:],mask=x[:,3:,:,:])
        #x_noise = self.noise_filter(x)

        x5,x4,x3,x2,x1 = self.base(x)
        if x_ela is not None:

            x51,x41,x31,x21,x11 = self.base1(x_ela)

                
            x4 = F.normalize(x4, p=2, dim=1)
            x3 = F.normalize(x3, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)

            x51 = F.normalize(x51, p=2, dim=1)
            x41 = F.normalize(x41, p=2, dim=1)
            x31 = F.normalize(x31, p=2, dim=1)
            x21 = F.normalize(x21, p=2, dim=1)
            x11 = F.normalize(x11, p=2, dim=1)

            x4_skip = self.bn4(self.sk4(torch.cat([x4,x41],1)))
            x3_skip = self.bn3(self.sk3(torch.cat([x3,x31],1)))
            x2_skip = self.bn2(self.sk2(torch.cat([x2,x21],1)))
            x1_skip = self.bn1(self.sk1(torch.cat([x1,x11],1)))
            return x5, x4_skip, x3_skip, x2_skip, x1_skip




        if semseg:
            return x5
        elif raw:
            
            return x5, x4, x3, x2, x1
        else:
            
            x5_skip = self.bn5(self.sk5(x5))
            x4_skip = self.bn4(self.sk4(x4))
            x3_skip = self.bn3(self.sk3(x3))
            x2_skip = self.bn2(self.sk2(x2))
            x1_skip = self.bn1(self.sk1(x1))
            del x5, x4, x3, x2,x1, x
            return x5_skip, x4_skip, x3_skip, x2_skip, x1_skip

class RSIS_segment(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):

        super(RSIS_segment,self).__init__()
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1
        self.args = args
        self.dropout = args.dropout
        self.skip_mode = args.skip_mode

        # convlstms have decreasing dimension as width and height increase
        skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         int(self.hidden_size/4),int(self.hidden_size/8)]                     

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                clstm_in_dim = self.hidden_size
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2

            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(16, 1,self.kernel_size, padding = padding)

        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk


   
    def forward(self, skip_feats, prev_state_spatial, prev_hidden_temporal, T=0):     
        
        
        clstm_in = skip_feats[0]
        skip_feats = list(skip_feats[1:])


        hidden_list = []

        for i in range(len(skip_feats)+1):
            # hidden states will be initialized the first time forward is called
            if prev_state_spatial is None:
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in,None, None)
                else:
                    state = self.clstm_list[i](clstm_in,None, prev_hidden_temporal[i])
            else:
                # else we take the ones from the previous step for the forward pass
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], None)
                    
                else:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], prev_hidden_temporal[i])

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden

        out_mask = self.conv_out(clstm_in)
        

        # classification branch

        return out_mask, hidden_list

class RSIS(nn.Module):
    """
    The recurrent decoder
    """

    def __init__(self, args):

        super(RSIS,self).__init__()
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        padding = 0 if self.kernel_size == 1 else 1
        self.args = args
        self.dropout = args.dropout
        self.skip_mode = args.skip_mode

        # convlstms have decreasing dimension as width and height increase
        #skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         #int(self.hidden_size/4),int(self.hidden_size/8)]
        skip_dims_out = [self.hidden_size, int(self.hidden_size/2),
                         int(self.hidden_size/4),int(self.hidden_size/8),int(self.hidden_size/16)]
                         

        dilations = [1, 6, 12, 18]
        self.aspp_f = nn.ModuleList()
    

       
        for i in range(len(skip_dims_out)+1):
            if i==0: 
                self.aspp_f.append(nn.Conv2d(skip_dims_out[0],skip_dims_out[0],self.kernel_size, padding = padding))

            else:

                self.aspp_f.append(nn.Conv2d(skip_dims_out[i-1],skip_dims_out[i-1],self.kernel_size, padding = padding))

        # initialize layers for each deconv stage
        self.clstm_list = nn.ModuleList()
        # 4 is the number of deconv steps that we need to reach image size in the output
        for i in range(len(skip_dims_out)):
            if i == 0:
                #clstm_in_dim = skip_dims_out[0]//2
                clstm_in_dim = skip_dims_out[0]
            else:
                clstm_in_dim = skip_dims_out[i-1]
                if self.skip_mode == 'concat':
                    clstm_in_dim*=2
            clstm_i = ConvLSTMCell(args, clstm_in_dim, skip_dims_out[i],self.kernel_size, padding = padding)
            self.clstm_list.append(clstm_i)

        self.conv_out = nn.Conv2d(args.maxseqlen, 1,self.kernel_size, padding = padding)

        self.conv_out_seg = nn.Conv2d(skip_dims_out[-1], 1,self.kernel_size, padding = padding)
        # calculate the dimensionality of classification vector
        # side class activations are taken from the output of the convlstm
        # therefore we need to compute the sum of the dimensionality of outputs
        # from all convlstm layers
        fc_dim = 0
        for sk in skip_dims_out:
            fc_dim+=sk


   
    def forward(self, skip_feats, prev_state_spatial, prev_hidden_temporal, T=0):      
        
        clstm_in = skip_feats[0]
        skip_feats = list(skip_feats[1:])

        if T>0:

            clstm_in = recursive_filter(clstm_in,self.aspp_f[0](clstm_in),T)

        hidden_list = []


        for i in range(len(skip_feats)+1):

            # hidden states will be initialized the first time forward is called
            if prev_state_spatial is None:
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in,None, None)
                else:
                    state = self.clstm_list[i](clstm_in,None, prev_hidden_temporal[i])
            else:
                # else we take the ones from the previous step for the forward pass
                if prev_hidden_temporal is None:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], None)
                    
                else:
                    state = self.clstm_list[i](clstm_in, prev_state_spatial[i], prev_hidden_temporal[i])

            hidden_list.append(state)
            hidden = state[0]

            if self.dropout > 0:
                hidden = nn.Dropout2d(self.dropout)(hidden)

            # apply skip connection
            if i < len(skip_feats):

                skip_vec = skip_feats[i]
                upsample = nn.UpsamplingBilinear2d(size = (skip_vec.size()[-2],skip_vec.size()[-1]))
                hidden = upsample(hidden)
                # skip connection
                if self.skip_mode == 'concat':
                    clstm_in = torch.cat([hidden,skip_vec],1)
                elif self.skip_mode == 'sum':
                    clstm_in = hidden + skip_vec
                elif self.skip_mode == 'mul':
                    clstm_in = hidden*skip_vec
                elif self.skip_mode == 'none':
                    clstm_in = hidden
                else:
                    raise Exception('Skip connection mode not supported !')
            else:
                self.upsample = nn.UpsamplingBilinear2d(size = (hidden.size()[-2]*2,hidden.size()[-1]*2))
                hidden = self.upsample(hidden)
                clstm_in = hidden
        
        seg_mask = self.conv_out_seg(clstm_in)

    
        out_mask = seg_mask
        # classification branch

        return None, out_mask, hidden_list
        
