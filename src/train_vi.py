import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from modules.model import RSIS, FeatureExtractor
import torchvision.models as models
from utils.hungarian import match, softIoU, reorder_mask
from utils.utils import get_optimizer, batch_to_var_vi, make_dir, check_parallel
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
from utils.objectives import softIoULoss,MaskedBCELoss
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

        #image_transforms = transforms.Compose([to_tensor])
                              
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


def runIter(args, encoder, decoder, x,x_ela, x1,x1_ela, y_mask,y_edge, sw_mask,
            crits,edge_crits, optims, mode='train', permutation = None, loss = None, prev_hidden_temporal_list=None,prev_hidden_temporal_list1=None, last_frame=False):
    """
    Runs forward a batch
    """
    mask_siou = crits
    edge_bce = edge_crits
    enc_opt, dec_opt = optims
    T = args.maxseqlen
    hidden_spatial = None
    hidden_spatial1 = None
    out_masks = []
    out_masks1 = []

    edge_masks = []
    edge_masks1 = []
    y_masks = [y_mask]
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
    else:
        encoder.train(False)
        decoder.train(False)
    feats = encoder(x,x_ela=x_ela)
    feats1 = encoder(x1,x_ela=x1_ela)
    #feats = encoder(x, raw=True)
    #feats1 = encoder(x1,raw=True)
    #feats = encoder(x)
    #feats1 = encoder(x1)
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
        #pdb.set_trace()
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        edge_mask,out_mask, hidden = decoder(feats, hidden_spatial, hidden_temporal,T=t)
        edge_mask1, out_mask1, hidden1 = decoder(feats1, hidden_spatial1, hidden_temporal1,T=t)
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
        #
        out_mask1 = upsample_match(out_mask1)
        #
        #if t>0:
            #out_mask_data = (torch.sigmoid(out_mask)>0.5).type(torch.cuda.FloatTensor).data.view(out_mask.size(0),1, -1)
            #pdb.set_trace()
            #y_pred_i = torch.max(y_masks[-1]-out_mask_data,torch.zeros(out_mask_data.size()).cuda())
            #y_masks.append(y_pred_i)
        if False:

            

            out_mask1 = out_mask1.view(out_mask1.size(0), -1)
            edge_mask = upsample_match(edge_mask)
            edge_mask = edge_mask.view(edge_mask.size(0), -1)
            edge_mask1 = upsample_match(edge_mask1)
            edge_mask1 = edge_mask1.view(edge_mask1.size(0), -1)
            edge_masks.append(edge_mask)
            edge_masks1.append(edge_mask1)

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
    #pdb.set_trace()
    #out_masks = torch.cat(out_masks,1).view(out_mask.size(0),t, -1)
    #out_masks1 = torch.cat(out_masks1,1).view(out_mask1.size(0),t, -1) 
    if args.maxseqlen<2:
        out_masks = torch.cat(out_masks,1).view(out_mask.size(0),t, -1)
        out_masks1 = torch.cat(out_masks1,1).view(out_mask1.size(0),t, -1) 
    elif args.ngpus<2:     
        out_masks = decoder.conv_out(torch.cat(out_masks,1)).view(out_mask.size(0),1, -1)
        out_masks1 = decoder.conv_out(torch.cat(out_masks1,1)).view(out_mask1.size(0),1, -1)
    else:
        out_masks = decoder.module.conv_out(torch.cat(out_masks,1)).view(out_mask.size(0),1, -1)
        out_masks1 = decoder.module.conv_out(torch.cat(out_masks1,1)).view(out_mask1.size(0),1, -1)
        #out_masks = torch.cat(out_masks,1).view(out_mask.size(0),t, -1)
        #out_masks1 = torch.cat(out_masks1,1).view(out_mask1.size(0),t, -1)
    #edge_masks = torch.cat(edge_masks,1).view(edge_mask.size(0),len(edge_masks), -1)
    #edge_masks1 = torch.cat(edge_masks1,1).view(edge_mask1.size(0),len(edge_masks1), -1)

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
        #y_mask_perm = torch.cat(y_masks,1).view(y_mask.size(0),t, -1)
        y_edge_perm = y_edge


       
    # loss is masked with sw_mask
    loss_mask_iou = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks.view(-1,out_masks.size()[-1]))
    loss_mask_iou = torch.mean(loss_mask_iou)

    loss_mask_iou1 = mask_siou(y_mask_perm.view(-1,y_mask_perm.size()[-1]),out_masks1.view(-1,out_masks1.size()[-1]))
    loss_mask_iou1 = torch.mean(loss_mask_iou1)

    #loss_mask_iou += torch.mean(mask_siou(y_mask.view(-1,y_mask.size()[-1]),out_masks_f.view(-1,out_masks_f.size()[-1])))
    #loss_mask_iou1 += torch.mean(mask_siou(y_mask.view(-1,y_mask.size()[-1]),out_masks1_f.view(-1,out_masks1_f.size()[-1])))


    if False:
        loss_edge_iou = edge_bce(y_edge_perm.view(-1,y_edge_perm.size()[-1]),edge_masks.view(-1,edge_masks.size()[-1]))
        loss_edge_iou = torch.mean(loss_edge_iou)

        loss_edge_iou1 = edge_bce(y_edge_perm.view(-1,y_edge_perm.size()[-1]),edge_masks1.view(-1,edge_masks1.size()[-1]))
        loss_edge_iou1 = torch.mean(loss_edge_iou1)    
    # total loss is the weighted sum of all terms
    if loss is None:
        loss = args.iou_weight * loss_mask_iou + args.iou_weight * loss_mask_iou1 #+ loss_edge_iou +loss_edge_iou1

    else:
        loss += args.iou_weight * loss_mask_iou + args.iou_weight * loss_mask_iou1 #+ loss_edge_iou +loss_edge_iou1
    #pdb.set_trace()
    #feat_mask1 = y_mask_perm.clone().resize_(feats[0].shape[0],1,feats[0].shape[-2],feats[0].shape[-1])
    #loss += (torch.abs(feats[0]-feats1[0])*(1-feat_mask1)).sum()/((1-feat_mask1).sum()*feats[0].shape[1])
    #feat_mask2 = y_mask_perm.clone().resize_(feats[1].shape[0],1,feats[1].shape[-2],feats[1].shape[-1])
    #loss += (torch.abs(feats[1]-feats1[1])*(1-feat_mask2)).sum()/((1-feat_mask2).sum()*feats[1].shape[1])
    #feat_mask3 = y_mask_perm.clone().resize_(feats[2].shape[0],1,feats[2].shape[-2],feats[2].shape[-1])
    #loss += (torch.abs(feats[2]-feats1[2])*(1-feat_mask3)).sum()/((1-feat_mask3).sum()*feats[2].shape[1])
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

    #edge_masks = torch.sigmoid(edge_masks)
    #outs_edge = edge_masks.data
    outs_edge = None

    del loss_mask_iou, loss_mask_iou1, feats,feats1, x, x1, y_mask, sw_mask, y_mask_perm
    if last_frame:
        del loss
        loss = None

    return loss, losses, outs,outs1, permutation, hidden_temporal_list, hidden_temporal_list1,outs_edge
    

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
    #pdb.set_trace()
    if args.resume:
        enc_opt.load_state_dict(enc_opt_dict)
        dec_opt.load_state_dict(dec_opt_dict)
        from collections import defaultdict
        dec_opt.state = defaultdict(dict, dec_opt.state)
    
    de_scheduler = torch.optim.lr_scheduler.StepLR(dec_opt, step_size=30, gamma=0.1)
    
    scheduler = torch.optim.lr_scheduler.StepLR(enc_opt, step_size=30, gamma=0.1)
    if not args.log_term:
        print ("Training logs will be saved to:", os.path.join(model_dir, 'train.log'))
        #sys.stdout = open(os.path.join(model_dir, 'train.log'), 'w')
        #sys.stderr = open(os.path.join(model_dir, 'train.err'), 'w')

    print (args)

    # objective function for mask output
    mask_siou = softIoULoss()

    edge_bce = MaskedBCELoss()
    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()
        mask_siou.cuda()
        #edge_bce.cuda()
    if args.ngpus > 1 and args.use_gpu:
        decoder = torch.nn.DataParallel(decoder, device_ids=range(args.ngpus))
        encoder = torch.nn.DataParallel(encoder, device_ids=range(args.ngpus))
        mask_siou = torch.nn.DataParallel(mask_siou, device_ids=range(args.ngpus))
        #edge_bce = torch.nn.DataParallel(edge_bce, device_ids=range(args.ngpus))

        #torch.distributed.init_process_group(backend="nccl", init_method='env://')
        #decoder = torch.nn.parallel.DistributedDataParallel(decoder,device_ids=[args.local_rank],output_device=args.local_rank)
        #encoder = torch.nn.parallel.DistributedDataParallel(encoder,device_ids=[args.local_rank],output_device=args.local_rank)
        #mask_siou = torch.nn.parallel.DistributedDataParallel(mask_siou,device_ids=[args.local_rank],output_device=args.local_rank)
        #edge_bce = torch.nn.parallel.DistributedDataParallel(edge_bce,device_ids=range(args.ngpus))


    crits = mask_siou
    edge_crits = edge_bce
    optims = [enc_opt, dec_opt]
    if args.use_gpu:
        torch.cuda.synchronize()
    start = time.time()

    # vars for early stopping
    best_val_loss = args.best_val_loss
    acc_patience = 0
    mt_val = -1
    kernel = np.ones((5,5))
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda()
    # keep track of the number of batches in each epoch for continuity when plotting curves
    loaders = init_dataloaders(args)
    num_batches = {'train': 0, 'val': 0}
    writer = SummaryWriter(model_dir)
    tensorboard_step = 0
    for e in range(args.max_epoch):
        #scheduler.step(e)
        #de_scheduler.step(e)
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
            if args.dataset == 'davis2016' or args.dataset == 'youtube' or args.dataset == 'davis2016_vi':
                for batch_idx, (inputs,inputs1, inputs_org, targets,seq_name,starting_frame, imgs_flow,imgs_ela,imgs1_ela) in enumerate(loaders[split]):
                #for batch_idx, (inputs,inputs1, inputs_org, targets,seq_name,starting_frame, imgs_flow) in enumerate(loaders[split]):
                    # send batch to GPU
                    #pdb.set_trace()
                    prev_hidden_temporal_list = None
                    prev_hidden_temporal_list1 = None
                    permutation = None
                    loss = None
                    last_frame = False
                    max_ii = min(len(inputs),args.length_clip)                      
                                        
                    for ii in range(max_ii):
                        #If are on the last frame from a clip, we will have to backpropagate the loss back to the beginning of the clip.
                        if ii == max_ii-1:
                            last_frame = True
                        #                x: input images (N consecutive frames from M different sequences)
                        #                y_mask: ground truth annotations (some of them are zeros to have a fixed length in number of object instances)
                        #                sw_mask: this mask indicates which masks from y_mask are valid
                        x, x1, y_mask, sw_mask = batch_to_var_vi(args, inputs[ii],inputs1[ii], targets[ii], input_org=None)
                        #y_edge = Variable(targets_edge[ii],requires_grad=False).cuda()
                        #pdb.set_trace()
                        target_edge = y_mask.view(y_mask.shape[0],y_mask.shape[1],x.shape[-2],x.shape[-1])
                        target_edge = torch.clamp(F.conv2d(target_edge, kernel_tensor, padding=(2, 2)), 0, 1) * torch.clamp(F.conv2d((1-target_edge).float(), kernel_tensor, padding=(2, 2)), 0, 1)
                        y_edge = Variable(target_edge.view(y_mask.shape),requires_grad=False).cuda()
                        x_ela = Variable(imgs_ela[ii],requires_grad=False).cuda()
                        x1_ela = Variable(imgs1_ela[ii],requires_grad=False).cuda()
                        #x_cat = torch.cat([x,x_ela], 1)
                        #x1_cat = torch.cat([x1,x1_ela], 1)

                        #From one frame to the following frame the prev_hidden_temporal_list is updated.
                        loss, losses, outs, outs1, permutation, hidden_temporal_list, hidden_temporal_list1, outs_edge= runIter(args, encoder, decoder, x, x_ela, x1, x1_ela, y_mask,y_edge, sw_mask,
                                                                                            crits,edge_crits, optims, split, permutation,
                                                                                            loss, prev_hidden_temporal_list,prev_hidden_temporal_list1, last_frame)
                        writer.add_scalar('training_loss', losses[0], tensorboard_step)                 
                        
                        if tensorboard_step % 200 == 0:
                            x_im = vutils.make_grid([inv_normalize(k) for k in x], normalize=True, scale_each=True)
                            x1_im = vutils.make_grid([inv_normalize(k) for k in x1], normalize=True, scale_each=True)

                            x_ela_im = vutils.make_grid([inv_normalize(k) for k in x_ela], normalize=True, scale_each=True)
                            #x_im = vutils.make_grid(x, normalize=True, scale_each=True)
                            #x1_im = vutils.make_grid( x1, normalize=True, scale_each=True)                            

                            x_o = vutils.make_grid(outs.view(outs.shape[0],1,x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                            #x_r = vutils.make_grid(outs[:,1,:].view(outs.shape[0],1,x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                            x_m = vutils.make_grid(y_mask.view(y_mask.shape[0],y_mask.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                            
                            writer.add_image('images_1', x1_im, tensorboard_step)
                            writer.add_image('images', x_im, tensorboard_step)
                            writer.add_image('images_ela', x_ela_im, tensorboard_step)

                            writer.add_image('prediction', x_o, tensorboard_step)
                            #writer.add_image('refine', x_r, tensorboard_step)
                            writer.add_image('masks', x_m, tensorboard_step)    

                            #x_edge = vutils.make_grid(outs_edge.view(outs_edge.shape[0],outs_edge.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)                                        
                            #writer.add_image('edge_pred', x_edge, tensorboard_step) 
                            #x_edge_m = vutils.make_grid(y_edge.view(outs.shape[0],outs.shape[1],x.shape[-2],x.shape[-1]), normalize=True, scale_each=True)
                            #writer.add_image('edge_masks', x_edge_m, tensorboard_step)
                        #Hidden temporal state from time instant ii is saved to be used when processing next time instant ii+1
                        prev_hidden_temporal_list = hidden_temporal_list
                        prev_hidden_temporal_list1 = hidden_temporal_list1
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
            #save_checkpoint(args, encoder, decoder, enc_opt, dec_opt)
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt,epoch=args.epoch_resume)
            acc_patience = 0
        elif args.epoch_resume==args.max_epoch-1:
            save_checkpoint(args, encoder, decoder, enc_opt, dec_opt,epoch=args.epoch_resume)
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
        #torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
    trainIters(args)
