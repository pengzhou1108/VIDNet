import torch
from utils.hungarian import softIoU,dice_loss, MaskedNLL, StableBalancedMaskedBCE, bce2d
import torch.nn as nn

import pdb

class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
    def forward(self, y_true, y_pred, sw):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1)
        costs = torch.masked_select(costs,sw.byte())
        return costs

class MaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred,sw=None):
        #costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight).view(-1,1)
        costs = bce2d(y_true,y_pred)
        #pdb.set_trace()
        #costs = torch.masked_select(costs,sw.byte())
        costs = torch.mean(costs)
        return costs

class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_true, y_pred, sw=None,recall=False):
        costs = softIoU(y_true,y_pred,recall).view(-1,1)
        #pdb.set_trace()
        if sw and (sw.data > 0).any():
            costs = torch.mean(torch.masked_select(costs,sw.byte()))
        else:
            costs = torch.mean(costs)
        return costs
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss,self).__init__()
    def forward(self, y_true, y_pred, sw=None,recall=False):
        costs = dice_loss(y_true,y_pred,recall).view(-1,1)
        if sw and (sw.data > 0).any():
            costs = torch.mean(torch.masked_select(costs,sw.byte()))
        else:
            costs = torch.mean(costs)
        return costs