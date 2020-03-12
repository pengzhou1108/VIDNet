import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from modules.model import RSIS, FeatureExtractor,WarpingLayer,FeatureExtractor_segment,RSIS_segment
import torchvision.models as models
from utils.hungarian import match, softIoU, reorder_mask
from utils.utils import get_optimizer, batch_to_var_vi, make_dir, check_parallel,make_boundaries
from utils.utils import outs_perms_to_cpu, save_checkpoint, load_checkpoint, get_base_params,get_skip_params,merge_params
from dataloader.dataset_utils import get_dataset
from dataloader.dataset_utils import sequence_palette
from scipy.ndimage.measurements import center_of_mass
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as data
from utils.objectives import softIoULoss,DiceLoss
import time
import math
import os
import warnings
import sys
from PIL import Image
import pickle
import random
import pdb
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def init_dataloaders(args):
    loaders = {}

    # init dataloaders for training and validation
    for split in ['train', 'val']:
        batch_size = args.batch_size
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_transforms = transforms.Compose([to_tensor, normalize])

                              
        if args.dataset == 'davis2016' or args.dataset == 'davis2016_vi':
            dataset = get_dataset(args,
                                split=split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and split == 'train',
                                inputRes = (240,427),
                                video_mode = True,
                                use_prev_mask = False)
        else: #args.dataset == 'youtube'
            dataset = get_dataset(args,
                                split=split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and split == 'train',
                                inputRes = (256,448),
                                video_mode = True,
                                use_prev_mask = False)

        loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True)
    return loaders
def runIter_segment(args, encoder, decoder,encoder1, decoder1, x, x_org,
            crits, optims, mode='train', permutation = None, loss = None, prev_hidden_temporal_list=None,
            prev_hidden_temporal_list_org=None, last_frame=False,cmp_mask=None,flow=None,warp_layer=None, retain_graph=False):
    """
    Runs forward a batch
    """
    mask_siou = crits
    enc_opt, dec_opt = optims
    T = 3
    hidden_spatial = None
    hidden_spatial_org = None
    out_masks = []
    out_masks_org= []
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
    else:
        encoder.train(False)
        decoder.train(False)
   
    feats = encoder1(x)
    feats_org = encoder1(x_org)

    hidden_temporal_list = []
    hidden_temporal_list_org = []
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            hidden_temporal_org = prev_hidden_temporal_list_org[t]
        else:
            hidden_temporal = None
            hidden_temporal_org = None
            
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        out_mask, hidden = decoder1(feats, hidden_spatial, hidden_temporal)
        out_mask_org, hidden_org = decoder1(feats_org, hidden_spatial_org, hidden_temporal_org)
        hidden_tmp = []
        hidden_tmp_org = []
        for ss in range(len(hidden)):

            hidden_tmp.append(hidden[ss][0].data)
            hidden_tmp_org.append(hidden_org[ss][0].data)

        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)
        hidden_spatial_org = hidden_org
        hidden_temporal_list_org.append(hidden_tmp_org)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)

        
        out_mask_org = upsample_match(out_mask_org)

        out_mask = out_mask.view(out_mask.size(0), -1)
        out_mask_org = out_mask_org.view(out_mask.size(0), -1)
        

        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_masks_org.append(out_mask_org)
    #pdb.set_trace()
    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_mask.size(0),len(out_masks), -1)
    out_masks_org = torch.cat(out_masks_org,1).view(out_mask_org.size(0),len(out_masks_org), -1)
    out_masks_org = torch.sigmoid(out_masks_org)
    y_mask_perm = out_masks_org.data
    #y_mask_perm_re = y_mask_refine
    # loss is masked with sw_mask
    loss_mask_iou = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks.view(-1,out_masks.size()[-1]))
    loss_mask_iou = torch.mean(loss_mask_iou)
    loss_mask_recall = torch.mean(mask_siou(cmp_mask,out_masks,recall=True))
    
    #loss_mask_iou_re = torch.mean(mask_siou(y_mask_perm_re.view(-1,y_mask_perm_re.size()[-1]),out_masks_refine.view(-1,out_masks_refine.size()[-1])))
    #pdb.set_trace()

    # total loss is the weighted sum of all terms
    if loss is None:
        loss = args.iou_weight * loss_mask_iou + args.iou_weight * loss_mask_recall
    else:
        loss += args.iou_weight * loss_mask_iou + args.iou_weight * loss_mask_recall
  
    if last_frame:
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        decoder.zero_grad()
        encoder.zero_grad()
    
        if mode == 'train':
            loss.backward(retain_graph=retain_graph)
            dec_opt.step()
            if args.update_encoder:
                enc_opt.step()

    #pytorch 0.4
    #losses = [loss.data[0], loss_mask_iou.data[0]]
    #pytorch 1.0
    losses = [loss.data.item(), loss_mask_iou.data.item()]
    
    out_masks = torch.sigmoid(out_masks)
    #out_masks_refine = torch.sigmoid(out_masks_refine)
    outs = out_masks.data
    #outs_refine = out_masks_refine.data


    del loss_mask_iou, loss_mask_recall, feats, x, x_org,feats_org
    if last_frame:
        del loss
        loss = None

    return loss, losses, outs.max(1,True)[0], y_mask_perm.max(1,True)[0], hidden_temporal_list, hidden_temporal_list_org

def runIter_dg(args, encoder, decoder, x, x1, y_mask, sw_mask,
            crits, optims, mode='train', permutation = None, loss = None, prev_hidden_temporal_list=None,prev_hidden_temporal_list1=None, last_frame=False):
    """
    Runs forward a batch
    """
    mask_siou = crits
    enc_opt, dec_opt = optims
    T = args.maxseqlen
    hidden_spatial = None
    hidden_spatial1 = None
    out_masks = []
    out_masks1 = []
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
    else:
        encoder.train(False)
        decoder.train(False)
    #feats = encoder(x)
    #feats1 = encoder(x1)
    feats = encoder(x, raw=True)
    feats1 = encoder(x1, raw=True)
    scores = torch.ones(y_mask.size(0),args.gt_maxseqlen,args.maxseqlen)

    hidden_temporal_list = []
    hidden_temporal_list1 = []
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            hidden_temporal1 = prev_hidden_temporal_list1[t]
            if args.only_temporal:
                hidden_spatial = None
                hidden_spatial1 = None
        else:
            hidden_temporal = None
            hidden_temporal1 = None

        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        _, out_mask, hidden = decoder(feats, hidden_spatial, hidden_temporal)
        _, out_mask1, hidden1 = decoder(feats1, hidden_spatial1, hidden_temporal1)
        hidden_tmp = []
        hidden_tmp1 = []
        for ss in range(len(hidden)):
            if mode == 'train':
                hidden_tmp.append(hidden[ss][0])
                hidden_tmp1.append(hidden1[ss][0])
            else:
                hidden_tmp.append(hidden[ss][0].data)
                hidden_tmp1.append(hidden1[ss][0].data)
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)
        hidden_spatial1 = hidden1
        hidden_temporal_list1.append(hidden_tmp1)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)
        out_mask1 = upsample_match(out_mask1)
        out_mask1 = out_mask1.view(out_mask1.size(0), -1)

        # repeat predicted mask as many times as elements in ground truth.
        # to compute iou against all ground truth elements at once
        y_pred_i = out_mask.unsqueeze(0)
        y_pred_i = y_pred_i.permute(1,0,2)
        y_pred_i = y_pred_i.repeat(1,y_mask.size(1),1)
        y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))
        y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))

        c = args.iou_weight * softIoU(y_true_p, y_pred_i)
        c = c.view(sw_mask.size(0),-1)
        scores[:,:,t] = c.cpu().data

        # get predictions in list to concat later
        out_masks.append(out_mask)
        out_masks1.append(out_mask1)

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_mask.size(0),len(out_masks), -1)
    out_masks1 = torch.cat(out_masks1,1).view(out_mask1.size(0),len(out_masks1), -1)

    #First frame prediction is compared with first frame ground truth to decide a matching between them by using the hungarian
    #algorithm based on the scores computed before. That matching is considered as right and is applied to the following frames in the sequences.
    if not args.single_object:
        if hidden_temporal is None:
            # get permutations of ground truth based on predictions (CPU computation)
            masks = [y_mask,out_masks]
            masks1 = [y_mask,out_masks1]

            sw_mask_mult = sw_mask.unsqueeze(-1).repeat(1,1,args.maxseqlen).byte()
            sw_mask_mult_T = sw_mask[:,0:args.maxseqlen].unsqueeze(-1).repeat(1,1,args.gt_maxseqlen).byte()
            sw_mask_mult_T = sw_mask_mult_T.permute(0,2,1).byte()
            sw_mask_mult = (sw_mask_mult.data.cpu() & sw_mask_mult_T.data.cpu()).float()
            scores = torch.mul(scores,sw_mask_mult) + (1-sw_mask_mult)*10
        
            scores = Variable(scores,requires_grad=False)
            if args.use_gpu:
                scores = scores.cuda()
        
            #match function also return the permutation indices from the matching that will be used in the following frames
            y_mask_perm, permutation = match(masks, scores)
        else:
            #The reorder_mask reorder the ground truth masks of the object instances for the following frames according to the permutation
            #indices obtained in the first frame. This way, the predicted masks can be compared directly one to one with the reordered ground truth masks.
            y_mask_perm = reorder_mask(y_mask, permutation)
        
        # move permuted ground truths back to GPU
        y_mask_perm = Variable(torch.from_numpy(y_mask_perm[:,0:t]), requires_grad=False)

        if args.use_gpu:
            y_mask_perm = y_mask_perm.cuda()

        sw_mask = Variable(torch.from_numpy(sw_mask.data.cpu().numpy()[:,0:t])).contiguous().float()

        if args.use_gpu:
            sw_mask = sw_mask.cuda()
        else:
            out_masks = out_masks.contiguous()
            y_mask_perm = y_mask_perm.contiguous()
    else:
        y_mask_perm = y_mask    
    # loss is masked with sw_mask
    loss_mask_iou = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks.view(-1,out_masks.size()[-1]))
    loss_mask_iou = torch.mean(loss_mask_iou)

    loss_mask_iou1 = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks1.view(-1,out_masks1.size()[-1]))
    loss_mask_iou1 = torch.mean(loss_mask_iou1)
    # total loss is the weighted sum of all terms
    if loss is None:
        loss = args.iou_weight * loss_mask_iou + args.iou_weight * loss_mask_iou1

    else:
        loss += args.iou_weight * loss_mask_iou + args.iou_weight * loss_mask_iou1
    #pdb.set_trace()
    #loss += torch.mean(torch.abs((feats[0]-feats1[0])*(1-y_mask_perm.clone().resize_(feats[0].shape[0],1,feats[0].shape[-2],feats[0].shape[-1]))))
    #loss += torch.mean(torch.abs((feats[1]-feats1[1])*(1-y_mask_perm.clone().resize_(feats[1].shape[0],1,feats[1].shape[-2],feats[1].shape[-1]))))
    #loss += torch.mean(torch.abs((feats[2]-feats1[2])*(1-y_mask_perm.clone().resize_(feats[2].shape[0],1,feats[2].shape[-2],feats[2].shape[-1]))))
    #loss += torch.mean(torch.abs((feats[3]-feats1[3])*(1-y_mask_perm.clone().resize_(feats[3].shape[0],1,feats[3].shape[-2],feats[3].shape[-1]))))
    if last_frame:
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        decoder.zero_grad()
        encoder.zero_grad()
    
        if mode == 'train':
            loss.backward()
            dec_opt.step()
            if args.update_encoder:
                enc_opt.step()

    #pytorch 0.4
    #losses = [loss.data[0], loss_mask_iou.data[0]]
    #pytorch 1.0
    losses = [loss.data.item(), loss_mask_iou.data.item()]
    #pdb.set_trace()
    out_masks = torch.sigmoid(out_masks)
    outs = out_masks.data
    out_masks1 = torch.sigmoid(out_masks1)
    outs1 = out_masks1.data


    del loss_mask_iou, loss_mask_iou1, feats,feats1, x, x1, y_mask, sw_mask, y_mask_perm
    if last_frame:
        del loss
        loss = None

    return loss, losses, outs,outs1, permutation, hidden_temporal_list, hidden_temporal_list1

def runIter(args, encoder, decoder, x, x_org, y_mask, sw_mask,
            crits, optims, mode='train', permutation = None, loss = None, prev_hidden_temporal_list=None, last_frame=False,cmp_mask=None,flow=None,warp_layer=None, retain_graph=False):
    """
    Runs forward a batch
    """
    mask_siou = crits
    enc_opt, dec_opt = optims
    T = args.maxseqlen
    hidden_spatial = None
    out_masks = []
    out_masks_refine = []
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
    else:
        encoder.train(False)
        decoder.train(False)
    feats = encoder(x)
    scores = torch.ones(y_mask.size(0),args.gt_maxseqlen,args.maxseqlen)

    hidden_temporal_list = []
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            if args.only_temporal:
                hidden_spatial = None
        else:
            hidden_temporal = None
            
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        out_mask, hidden = decoder(feats, hidden_spatial, hidden_temporal)
        hidden_tmp = []
        for ss in range(len(hidden)):
            if mode == 'train':
                hidden_tmp.append(hidden[ss][0])
            else:
                hidden_tmp.append(hidden[ss][0].data)
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)

        if False:
            out_m = torch.sigmoid(out_mask)
            x_refine = out_m*x_org + x*(1-out_m)
            feats_re = encoder(x_refine)
            out_mask_refine, _ = decoder(feats_re, hidden_spatial, hidden_temporal)
            out_mask_refine = upsample_match(out_mask_refine)
            out_mask_refine = out_mask_refine.view(out_mask_refine.size(0), -1)
            y_mask_refine = y_mask - y_mask*out_m.view(y_mask.shape)
            out_masks_refine.append(out_mask_refine)


        out_mask = out_mask.view(out_mask.size(0), -1)
        
        # repeat predicted mask as many times as elements in ground truth.
        # to compute iou against all ground truth elements at once
        y_pred_i = out_mask.unsqueeze(0)
        y_pred_i = y_pred_i.permute(1,0,2)
        y_pred_i = y_pred_i.repeat(1,y_mask.size(1),1)
        y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))
        y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))

        c = args.iou_weight * softIoU(y_true_p, y_pred_i)
        c = c.view(sw_mask.size(0),-1)
        scores[:,:,t] = c.cpu().data

        # get predictions in list to concat later
        out_masks.append(out_mask)
        
    #pdb.set_trace()
    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_mask.size(0),len(out_masks), -1)
    #out_masks_refine = torch.cat(out_masks_refine,1).view(out_mask_refine.size(0),len(out_masks_refine), -1)
    if not args.single_object:
        #First frame prediction is compared with first frame ground truth to decide a matching between them by using the hungarian
        #algorithm based on the scores computed before. That matching is considered as right and is applied to the following frames in the sequences.
        if hidden_temporal is None:
            # get permutations of ground truth based on predictions (CPU computation)
            masks = [y_mask,out_masks]

            sw_mask_mult = sw_mask.unsqueeze(-1).repeat(1,1,args.maxseqlen).byte()
            sw_mask_mult_T = sw_mask[:,0:args.maxseqlen].unsqueeze(-1).repeat(1,1,args.gt_maxseqlen).byte()
            sw_mask_mult_T = sw_mask_mult_T.permute(0,2,1).byte()
            sw_mask_mult = (sw_mask_mult.data.cpu() & sw_mask_mult_T.data.cpu()).float()
            scores = torch.mul(scores,sw_mask_mult) + (1-sw_mask_mult)*10
        
            scores = Variable(scores,requires_grad=False)
            if args.use_gpu:
                scores = scores.cuda()
        
            #match function also return the permutation indices from the matching that will be used in the following frames
            y_mask_perm, permutation = match(masks, scores)
            #y_mask_perm_re = reorder_mask(y_mask_refine, permutation)
        else:
            #The reorder_mask reorder the ground truth masks of the object instances for the following frames according to the permutation
            #indices obtained in the first frame. This way, the predicted masks can be compared directly one to one with the reordered ground truth masks.
            y_mask_perm = reorder_mask(y_mask, permutation)
            #y_mask_perm_re = reorder_mask(y_mask_refine, permutation)

        # move permuted ground truths back to GPU
        y_mask_perm = Variable(torch.from_numpy(y_mask_perm[:,0:t]), requires_grad=False)
        #y_mask_perm_re = Variable(torch.from_numpy(y_mask_perm_re[:,0:t]), requires_grad=False)
        if args.use_gpu:
            y_mask_perm = y_mask_perm.cuda()
            #y_mask_perm_re = y_mask_perm_re.cuda()

        sw_mask = Variable(torch.from_numpy(sw_mask.data.cpu().numpy()[:,0:t])).contiguous().float()

        if args.use_gpu:
            sw_mask = sw_mask.cuda()
        else:
            out_masks = out_masks.contiguous()
            y_mask_perm = y_mask_perm.contiguous()
            #y_mask_perm_re = y_mask_perm_re.contiguous()
    else:
        y_mask_perm = y_mask
        #y_mask_perm_re = y_mask_refine
    # loss is masked with sw_mask
    loss_mask_iou = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks.view(-1,out_masks.size()[-1]))
    loss_mask_iou = torch.mean(loss_mask_iou)
    
    #loss_mask_iou_re = torch.mean(mask_siou(y_mask_perm_re.view(-1,y_mask_perm_re.size()[-1]),out_masks_refine.view(-1,out_masks_refine.size()[-1])))
    #pdb.set_trace()
    if flow is not None:
        #pdb.set_trace()
        mask_warp = warp_layer(out_masks.view(out_masks.shape[0],out_masks.shape[1],x.shape[-2],x.shape[-1]), flow)
        mask_warp = torch.sigmoid(mask_warp)
    else:
        mask_warp = None
    if cmp_mask is not None:
        loss_mask_iou += 0.5*torch.mean(mask_siou(cmp_mask.data,out_masks.view(out_masks.shape[0],out_masks.shape[1],x.shape[-2],x.shape[-1])))

    # total loss is the weighted sum of all terms
    if loss is None:
        loss = args.iou_weight * loss_mask_iou #+ args.iou_weight * loss_mask_iou_re
    else:
        loss += args.iou_weight * loss_mask_iou #+ args.iou_weight * loss_mask_iou_re
  
    if last_frame:
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        decoder.zero_grad()
        encoder.zero_grad()
    
        if mode == 'train':
            loss.backward(retain_graph=retain_graph)
            dec_opt.step()
            if args.update_encoder:
                enc_opt.step()

    #pytorch 0.4
    #losses = [loss.data[0], loss_mask_iou.data[0]]
    #pytorch 1.0
    losses = [loss.data.item(), loss_mask_iou.data.item()]
    
    out_masks = torch.sigmoid(out_masks)
    #out_masks_refine = torch.sigmoid(out_masks_refine)
    outs = out_masks.data
    #outs_refine = out_masks_refine.data


    del loss_mask_iou, feats, x, y_mask, sw_mask, y_mask_perm,#feats_re, y_mask_perm_re
    if last_frame:
        del loss
        loss = None

    return loss, losses, outs, permutation, hidden_temporal_list,mask_warp
    

def trainIters(args):

    epoch_resume = 0
    model_dir = os.path.join('../models/', args.model_name)

    if args.resume:
        # will resume training the model with name args.model_name
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name,args.use_gpu)

        epoch_resume = load_args.epoch_resume
        encoder = FeatureExtractor(load_args)
        decoder = RSIS(load_args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

        args = load_args

    elif args.transfer:
        # load model from args and replace last fc layer
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.transfer_from,args.use_gpu)
        encoder = FeatureExtractor(load_args)
        decoder = RSIS(args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)


    else:
        encoder = FeatureExtractor(args)
        decoder = RSIS(args)
    if args.use_segment:
        encoder1_dict, decoder1_dict, _, _, load_args = load_checkpoint(args.segment_name,args.use_gpu)

        encoder1 = FeatureExtractor_segment(load_args)
        decoder1 = RSIS_segment(load_args)
        if args.ngpus > 1 and args.use_gpu:
            decoder1 = torch.nn.DataParallel(decoder1, device_ids=range(args.ngpus))
            encoder1 = torch.nn.DataParallel(encoder1, device_ids=range(args.ngpus))

        encoder1_dict, decoder1_dict = check_parallel(encoder1_dict,decoder1_dict)
        encoder1.load_state_dict(encoder1_dict)
        to_be_deleted_dec = []
        for k in decoder1_dict.keys():
            if 'fc_stop' in k:
                to_be_deleted_dec.append(k)
        for k in to_be_deleted_dec:
            del decoder1_dict[k]
        decoder1.load_state_dict(decoder1_dict) 

        if args.use_gpu:
            encoder1.cuda()
            decoder1.cuda()
        encoder1.eval()
        decoder1.eval() 

    #warp_layer = WarpingLayer()
    # model checkpoints will be saved here
    make_dir(model_dir)
    inv_normalize = transforms.Compose([transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                    std=[1/0.229, 1/0.224, 1/0.255]
                                            )])
    # save parameters for future use
    pickle.dump(args, open(os.path.join(model_dir,'args.pkl'),'wb'))

    encoder_params = get_base_params(args,encoder)
    skip_params = get_skip_params(encoder)
    decoder_params = list(decoder.parameters()) + list(skip_params)
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params, args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params, args.weight_decay_cnn)

    if args.resume:
        enc_opt.load_state_dict(enc_opt_dict)
        dec_opt.load_state_dict(dec_opt_dict)
        from collections import defaultdict
        dec_opt.state = defaultdict(dict, dec_opt.state)

    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(model_dir, 'train.log'))
        #sys.stdout = open(os.path.join(model_dir, 'train.log'), 'w')
        #sys.stderr = open(os.path.join(model_dir, 'train.err'), 'w')

    print (args)

    # objective function for mask output
    mask_siou = softIoULoss()
    #mask_siou = DiceLoss()
    if args.ngpus > 1 and args.use_gpu:
        decoder = torch.nn.DataParallel(decoder, device_ids=range(args.ngpus))
        encoder = torch.nn.DataParallel(encoder, device_ids=range(args.ngpus))
        mask_siou = torch.nn.DataParallel(mask_siou, device_ids=range(args.ngpus))
    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()
        mask_siou.cuda()



    crits = mask_siou
    optims = [enc_opt, dec_opt]
    if args.use_gpu:
        torch.cuda.synchronize()
    start = time.time()

    # vars for early stopping
    best_val_loss = args.best_val_loss
    acc_patience = 0
    mt_val = -1
    kernel = np.array([ [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1] ], dtype=np.float32)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda()
    # keep track of the number of batches in each epoch for continuity when plotting curves
    loaders = init_dataloaders(args)
    num_batches = {'train': 0, 'val': 0}
    writer = SummaryWriter(model_dir)
    tensorboard_step = 0
    for e in range(args.max_epoch):
        print ("Epoch", e + epoch_resume)
        # store losses in lists to display average since beginning
        epoch_losses = {'train': {'total': [], 'iou': []},
                            'val': {'total': [], 'iou': []}}
            # total mean for epoch will be saved here to display at the end
        total_losses = {'total': [], 'iou': []}

        # check if it's time to do some changes here
        if e + epoch_resume >= args.finetune_after and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            args.update_encoder = True
            acc_patience = 0
            mt_val = -1

        # we validate after each epoch
        for split in ['train', 'val']:
            if args.dataset == 'davis2016' or args.dataset == 'davis2016_vi' or args.dataset == 'youtube':
                #for batch_idx, (inputs, inputs1, inputs_org, targets,seq_name,starting_frame, imgs_flow) in enumerate(loaders[split]):
                for batch_idx, (inputs,inputs1, inputs_org, targets,seq_name,starting_frame, imgs_flow,imgs_ela,imgs1_ela) in enumerate(loaders[split]):
                    # send batch to GPU
                    #pdb.set_trace()
                    prev_hidden_temporal_list = None
                    segment_prev_hidden_temporal_list = None
                    prev_hidden_temporal_list1 = None
                    segment_prev_hidden_temporal_list1 = None
                    segment_prev_hidden_temporal_list_org = None
                    permutation = None
                    loss = None
                    cmp_mask = None
                    last_frame = False
                    max_ii = min(len(inputs),args.length_clip)                      
                                        
                    for ii in range(max_ii):
                        #If are on the last frame from a clip, we will have to backpropagate the loss back to the beginning of the clip.
                        if ii == max_ii-1:
                            last_frame = True
                        #                x: input images (N consecutive frames from M different sequences)
                        #                y_mask: ground truth annotations (some of them are zeros to have a fixed length in number of object instances)
                        #                sw_mask: this mask indicates which masks from y_mask are valid
                        x,x1, x_org, y_mask, sw_mask = batch_to_var_vi(args, inputs[ii],inputs1[ii], targets[ii], input_org=inputs_org[ii])
                        #flow = Variable(imgs_flow[ii],requires_grad=False)
                        flow = None
                        #From one frame to the following frame the prev_hidden_temporal_list is updated.
                        #loss, losses, outs, permutation, hidden_temporal_list, outs_warp = runIter(args, encoder, decoder, x, x_org, y_mask, sw_mask,
                                                                                            #crits, optims, split, permutation,
                                                                                            #loss, prev_hidden_temporal_list, last_frame=False, cmp_mask= cmp_mask, flow=flow, warp_layer=warp_layer,retain_graph=False)

                        loss, losses, outs, outs1, permutation, hidden_temporal_list, hidden_temporal_list1 = runIter_dg(args, encoder, decoder, x, x1, y_mask, sw_mask,
                                                                                            crits, optims, split, permutation,
                                                                                            loss, prev_hidden_temporal_list,prev_hidden_temporal_list1, last_frame=False)
                        if True:                    
                            I1_outs = outs1.view(outs1.shape[0],outs1.shape[1],x.shape[-2],x.shape[-1])
                            #boundary = torch.clamp(F.conv2d(I_outs, kernel_tensor, padding=(1, 1)), 0, 1) * torch.clamp(F.conv2d((1-I_outs).float(), kernel_tensor, padding=(1, 1)), 0, 1)

                            x1_refine = I1_outs*x_org \
                                        + x1*(1-I1_outs)
                            #y_mask_re = y_mask - y_mask*I_outs.view(outs.shape)
                            #loss, losses, outs_refine,_, hidden_temporal_list, _ = runIter(args, encoder, decoder, x_refine, x_org, y_mask_re, sw_mask,
                                                    #crits, optims, split, permutation,
                                                    #loss, prev_hidden_temporal_list, last_frame)
                            loss, losses, outs1_refine, y1_mask_re, segment_hidden_temporal_list1,segment_hidden_temporal_list_org = runIter_segment(args, encoder, decoder, encoder1, decoder1, x1_refine, x_org,
                                                    crits, optims, split, permutation,
                                                    loss, segment_prev_hidden_temporal_list1,segment_prev_hidden_temporal_list_org, last_frame=False, cmp_mask=outs1)

                            I_outs = outs.view(outs.shape[0],outs.shape[1],x.shape[-2],x.shape[-1])
                            x_refine = I_outs*x_org \
                                        + x*(1-I_outs)
                            #y_mask_re = y_mask - y_mask*I_outs.view(outs.shape)
                            #loss, losses, outs_refine,_, hidden_temporal_list, _ = runIter(args, encoder, decoder, x_refine, x_org, y_mask_re, sw_mask,
                                                    #crits, optims, split, permutation,
                                                    #loss, prev_hidden_temporal_list, last_frame)
                            loss, losses, outs_refine, y_mask_re, segment_hidden_temporal_list,_ = runIter_segment(args, encoder, decoder, encoder1, decoder1, x_refine, x_org,
                                                    crits, optims, split, permutation,
                                                    loss, segment_prev_hidden_temporal_list,segment_prev_hidden_temporal_list_org, last_frame=last_frame, cmp_mask=outs)
                        writer.add_scalar('training_loss', losses[0], tensorboard_step)                 
                        
                        if tensorboard_step % 200 == 0:
                            x_im = vutils.make_grid([inv_normalize(k) for k in x], normalize=True, scale_each=True)
                            
                            x_o = vutils.make_grid(outs.view(outs.shape[0],outs.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                            
                            x_m = vutils.make_grid(y_mask.view(outs.shape[0],outs.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                            
                            writer.add_image('images', x_im, tensorboard_step)
                            
                            writer.add_image('prediction', x_o, tensorboard_step)
                            
                            writer.add_image('masks', x_m, tensorboard_step)
                            if True:
                                x_im_re = vutils.make_grid([inv_normalize(k) for k in x1_refine], normalize=True, scale_each=True)
                                x_re = vutils.make_grid(outs1_refine.view(outs1_refine.shape[0],outs1_refine.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                                x_m_re = vutils.make_grid(y_mask_re.view(outs.shape[0],outs.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                                writer.add_image('refine_images', x_im_re, tensorboard_step)
                                writer.add_image('pred_segment', x_re, tensorboard_step)
                                writer.add_image('refine_masks', x_m_re, tensorboard_step)
                            if cmp_mask is not None:
                                x_warp = vutils.make_grid(outs_warp, normalize=True, scale_each=True)
                                writer.add_image('warp_masks', x_warp, tensorboard_step)

                        #Hidden temporal state from time instant ii is saved to be used when processing next time instant ii+1
                        prev_hidden_temporal_list = hidden_temporal_list
                        segment_prev_hidden_temporal_list = segment_hidden_temporal_list
                        segment_prev_hidden_temporal_list1 = segment_hidden_temporal_list1
                        segment_prev_hidden_temporal_list_org = segment_hidden_temporal_list_org
                        prev_hidden_temporal_list1 = hidden_temporal_list1
                        cmp_mask = None
                        tensorboard_step += 1
        
                    # store loss values in dictionary separately
                    epoch_losses[split]['total'].append(losses[0])
                    epoch_losses[split]['iou'].append(losses[1])

    
                    # print after some iterations
                    if (batch_idx + 1)% args.print_every == 0:
    
                        mt = np.mean(epoch_losses[split]['total'])
                        mi = np.mean(epoch_losses[split]['iou'])
    
                        te = time.time() - start
                        print ("iter %d:\ttotal:%.4f\tiou:%.4f\ttime:%.4f" % (batch_idx, mt, mi, te))
                        if args.use_gpu:
                            torch.cuda.synchronize()
                        start = time.time()

            num_batches[split] = batch_idx + 1
            # compute mean val losses within epoch

            if split == 'val' and args.smooth_curves:
                if mt_val == -1:
                    mt = np.mean(epoch_losses[split]['total'])
                else:
                    mt = 0.9*mt_val + 0.1*np.mean(epoch_losses[split]['total'])
                mt_val = mt

            else:
                mt = np.mean(epoch_losses[split]['total'])

            mi = np.mean(epoch_losses[split]['iou'])

            # save train and val losses for the epoch
            total_losses['iou'].append(mi)

            args.epoch_resume = e + epoch_resume

            print ("Epoch %d:\ttotal:%.4f\tiou:%.4f\t(%s)" % (e, mt, mi, split))


        if mt < (best_val_loss - args.min_delta):
            print ("Saving checkpoint.")
            best_val_loss = mt
            args.best_val_loss = best_val_loss
            # saves model, params, and optimizers
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt)
            acc_patience = 0
        else:
            acc_patience += 1


        if acc_patience > args.patience and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            acc_patience = 0
            args.update_encoder = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            mt_val = -1
            encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, _ = load_checkpoint(args.model_name,args.use_gpu)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            enc_opt.load_state_dict(enc_opt_dict)
            dec_opt.load_state_dict(dec_opt_dict)
            

        # early stopping after N epochs without improvement
        if acc_patience > args.patience_stop:
            break


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    trainIters(args)
