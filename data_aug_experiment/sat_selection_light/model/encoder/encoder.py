import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import dgl
import dgl.nn.pytorch as dglnn
# from .gmt.gmt_pool import GMTPoolingLayer
# from .hgnn.hgnn import HGNN

class SATzillaEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
    
    def forward(self, sat_feats):
        return self.norm(sat_feats)

