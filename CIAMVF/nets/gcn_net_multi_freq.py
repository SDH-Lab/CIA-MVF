import torch
import torch.nn as nn
import dgl
import numpy as np
import matplotlib.pyplot as plt
import torchvision
                                 
from scipy.stats import ttest_ind

from scipy.fft import fft, fftfreq
import torch.nn.functional as F
import os  
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
              
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from scipy import stats          



from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout



     
class CascadeDecomposer(nn.Module):
    
    def __init__(self, in_dim, levels=2):                                  
        super().__init__()
        self.levels = levels
        
        self.low_pass_filter_1 = nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=4, dilation=2, groups=in_dim)
        self.high_pass_filter_1 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)

                                
        self.low_pass_filter_2 = nn.Conv1d(in_dim, in_dim, kernel_size=5, padding=8, dilation=4, groups=in_dim)
        self.high_pass_filter_2 = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        

    def forward(self, x):
        
                       
                   
        L1_raw = self.low_pass_filter_1(x)
        L1 = F.leaky_relu(L1_raw, 0.1)
        
                          
        residual_1 = x - L1
        H1_raw = self.high_pass_filter_1(residual_1)
        H1 = F.leaky_relu(H1_raw, 0.1)                    
        
                                            
                       
        return torch.stack([L1, H1], dim=1)                      

                                   
 