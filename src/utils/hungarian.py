import torch
from torch.autograd import Variable
import torch.nn as nn
from munkres import Munkres
import numpy as np
import time
import pdb
import torch.nn.functional as F

torch.manual_seed(0)

def MaskedNLL(target, probs, balance_weights=None):
    # adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, ) which contains the index of the true
            class for each corresponding step.
        probs: A Variable containing a FloatTensor of size
            (batch, num_classes) which contains the
            softmax probability for each class.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    log_probs = torch.log(probs)

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)
    return losses.squeeze()

def StableBalancedMaskedBCE(target, out, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total
    out = torch.sigmoid(out)
    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses.squeeze()

def bce2d(target,input):
    #n, c, h, w = input.size()

    # assert(max(target) == 1)
    #log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    #target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    input = torch.sigmoid(input)
    log_p = input.contiguous().view(1, -1)
    target_t = target.contiguous().view(1, -1)    
    target_trans = target_t.clone()
    pos_index = (target_t >0)
    neg_index = (target_t ==0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num
    #weight[pos_index] = 0.7
    #weight[neg_index] = 0.3
    #pdb.set_trace()
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss.squeeze()

def dice_loss(target,pred, recall=False):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    #pdb.set_trace()
    smooth = 1e-6
    pred = torch.sigmoid(pred)
    # have to use contiguous since they may from a torch.view op
    iflat = pred
    tflat = target
    if recall:
      #intersection = (iflat * tflat*(10*c0_w+1)).sum()
      intersection = (iflat * tflat).sum()
      A_sum = torch.sum(tflat * tflat)
      #B_sum = torch.sum(tflat * tflat*(10*c0_w+1)*(10*c0_w+1))
      B_sum = torch.sum(tflat * tflat)
    else:
      intersection = (iflat * tflat).sum()
      A_sum = torch.sum(iflat * iflat)
      #B_sum = torch.sum(tflat * tflat*(10*c0_w+1)*(10*c0_w+1))
      B_sum = torch.sum(tflat * tflat)
    

    cost = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    return cost.squeeze()

def softIoU(target, out, e=1e-6, recall=False):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """


    out = torch.sigmoid(out)
    if not recall:
        num = (out*target).sum(1,True).sum(-1,True)
        den = (out+target-out*target).sum(1,True).sum(-1,True) + e
        iou = num / den
    else:
        num = (out*target).sum(1,True).sum(-1,True)
        den = target.sum(1,True).sum(-1,True) + e
        iou = num / den        

    cost = (1 - iou)

    return cost.squeeze()

def match(masks, overlaps):
    """
    Args:
        masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
        overlaps - [batch_size,T,T] - matrix of costs between all pairs
    Returns:
        t_mask_cpu - [batch_size,T,N] permuted ground truth masks
        permute_indices - permutation indices used to sort the above
    """

    overlaps = (overlaps.data).cpu().numpy().tolist()
    m = Munkres()

    t_mask, p_mask = masks

    # get true mask values to cpu as well
    t_mask_cpu = (t_mask.data).cpu().numpy()
    # init matrix of permutations
    permute_indices = np.zeros((t_mask.size(0),t_mask.size(1)),dtype=int)
    # we will loop over all samples in batch (must apply munkres independently)
    for sample in range(p_mask.size(0)):
        # get the indexes of minimum cost
        indexes = m.compute(overlaps[sample])
        for row, column in indexes:
            # put them in the permutation matrix
            permute_indices[sample,column] = row

        # sort ground according to permutation
        t_mask_cpu[sample] = t_mask_cpu[sample,permute_indices[sample],:]
        
    return t_mask_cpu, permute_indices
    
def reorder_mask(y_mask, permutation):

     t_mask_cpu = (y_mask.data).cpu().numpy()
     size = y_mask.size(0)
     for sample in range(size):
         t_mask_cpu[sample] = t_mask_cpu[sample,permutation[sample],:]
         
     return t_mask_cpu
