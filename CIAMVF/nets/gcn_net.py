import torch
import torch.nn as nn
from .correlations import pearson_correlation, partial_correlation, spearman_correlation, partial_spearman_correlation
                         
                                     
from torch.cuda.amp import autocast, custom_fwd 
import torch.utils.checkpoint as checkpoint               
from torch_geometric.utils import degree, add_self_loops
from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
import torch.autograd
torch.autograd.set_detect_anomaly(True)
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
import lightning as L
                                                        
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
from .gcn_net_multi_freq import CascadeDecomposer
        
import argparse
import datetime
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
import torch.optim as optim
from torchmetrics.classification import MulticlassF1Score
                             
import torch
import torch.nn as nn

from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import warnings
from torch_geometric.utils import to_dense_adj
from sklearn.preprocessing import StandardScaler
from scipy import stats
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.neighbors import NearestNeighbors
import warnings

from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import scipy.sparse as sp

warnings.filterwarnings('ignore')
from scipy.special import digamma

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
                                                                       
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):                          
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')                                                          
        pt = torch.exp(-ce_loss)                             
        focal_loss_term = (1 - pt)**self.gamma * ce_loss            
        if self.alpha is not None:
            if self.alpha.device != focal_loss_term.device:
                self.alpha = self.alpha.to(focal_loss_term.device)
                                                               
            alpha_t = self.alpha.gather(0, targets)            
            focal_loss_term = alpha_t * focal_loss_term

        if self.reduction == 'mean':
            return focal_loss_term.mean()
        elif self.reduction == 'sum':
            return focal_loss_term.sum()
        elif self.reduction == 'none':
            return focal_loss_term
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")


class CostSensitiveLoss(nn.Module):
    
    
    def __init__(self, cost_matrix=None, reduction='mean'):
        super(CostSensitiveLoss, self).__init__()
        self.cost_matrix = cost_matrix
        self.reduction = reduction
        
                                      
        if self.cost_matrix is None:
            self.use_default_cost_matrix = True
        else:
            self.use_default_cost_matrix = False
            if not isinstance(self.cost_matrix, torch.Tensor):
                self.cost_matrix = torch.tensor(self.cost_matrix, dtype=torch.float32)
    
    def _create_default_cost_matrix(self, num_classes, class_counts=None):
        
        cost_matrix = torch.zeros(num_classes, num_classes)
        
        if class_counts is not None:
                        
            total_samples = sum(class_counts)
            class_frequencies = [count / total_samples for count in class_counts]
            
            for i in range(num_classes):
                for j in range(num_classes):
                    if i == j:
                        cost_matrix[i, j] = 0.0            
                    else:
                                                 
                                                                                  
                        epsilon = 1e-6
                        penalty_factor = 2.0        
                        cost_matrix[i, j] = penalty_factor / (class_frequencies[i] + epsilon)
        else:
                                   
            for i in range(num_classes):
                for j in range(num_classes):
                    if i == j:
                        cost_matrix[i, j] = 0.0
                    else:
                                           
                        cost_matrix[i, j] = abs(i - j) + 1.0
        
        return cost_matrix
    
    def forward(self, inputs, targets, class_counts=None):
        
        batch_size, num_classes = inputs.shape
        device = inputs.device
        
                       
        if self.use_default_cost_matrix:
            cost_matrix = self._create_default_cost_matrix(num_classes, class_counts)
            cost_matrix = cost_matrix.to(device)
        else:
            cost_matrix = self.cost_matrix.to(device)
        
                     
        probs = F.softmax(inputs, dim=1)                             
        
                            
        log_probs = F.log_softmax(inputs, dim=1)                             
        
                        
        losses = []
        for i in range(batch_size):
            true_class = targets[i].item()
            
                             
            sample_probs = probs[i]                 
            
                                               
            sample_loss = 0.0
            for pred_class in range(num_classes):
                if pred_class == true_class:
                                  
                    sample_loss += -log_probs[i, pred_class]
                else:
                                 
                    cost = cost_matrix[true_class, pred_class]
                    sample_loss += cost * (-log_probs[i, pred_class])
            
            losses.append(sample_loss)
        
                   
        losses = torch.stack(losses)                
        
              
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:          
            return losses
    
    def update_cost_matrix(self, new_cost_matrix):
        
        if not isinstance(new_cost_matrix, torch.Tensor):
            new_cost_matrix = torch.tensor(new_cost_matrix, dtype=torch.float32)
        self.cost_matrix = new_cost_matrix
        self.use_default_cost_matrix = False
    
    def get_cost_matrix(self):
        
        if self.use_default_cost_matrix:
            return None            
        return self.cost_matrix


class NodeFeatureExtractor(nn.Module):
    
    def __init__(self, num_sub_bands, time_length, hidden_dim, output_dim, dropout=0.3):
        super(NodeFeatureExtractor, self).__init__()                    
        self.conv1 = nn.Conv1d(in_channels=num_sub_bands, out_channels=hidden_dim, kernel_size=9, padding='same')            
        self.ln1 = nn.LayerNorm(hidden_dim)                                
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)                            
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(x)                                                 
        h = self.ln1(h.transpose(1, 2)).transpose(1, 2)
        h = F.leaky_relu(h)                                                           
        h_pooled = self.global_avg_pool(h)                              
        h_flattened = h_pooled.squeeze(-1)
        h_dropped = self.dropout(h_flattened)
        output = self.fc(h_dropped)
        
        return output
                             
                                                        
class RCensNetConv(nn.Module):
    
    def __init__(self, nfeat_in, nfeat_out, nfeat_edge_in, nfeat_edge_out, 
                 num_relations, node_layer=True, bias=True):
        super(RCensNetConv, self).__init__()
        
                                             
        self.node_layer = True                
        self.num_relations = num_relations
        
                                        
        self.relation_weights = nn.ModuleList([
            nn.Linear(nfeat_in, nfeat_out, bias=False) for _ in range(num_relations)
        ])
        self.self_loop_weight = nn.Linear(nfeat_in, nfeat_out, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin in self.relation_weights:
            nn.init.kaiming_uniform_(lin.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.self_loop_weight.weight, a=np.sqrt(5))
        if self.self_loop_weight.bias is not None:
            nn.init.zeros_(self.self_loop_weight.bias)
    
   

                                                          
    def forward(self, node_features, edge_features, adj_e, adj_v, T, edge_index, edge_type):
        
                             
                                  
        out_node_features = torch.zeros(node_features.size(0), self.self_loop_weight.out_features, device=node_features.device)
        
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0: 
                continue
            
                                   
            r_edge_index = edge_index[:, mask]
            r_edge_features = edge_features[mask]                
            
                            
            transformed_nodes = self.relation_weights[r](node_features)
            
                           
                                                       
                                               
            
            if r_edge_index.shape[1] > 0:
                                                             
                if r_edge_features.dim() > 1:
                                                                
                    r_edge_weights_raw = r_edge_features.squeeze(-1) if r_edge_features.shape[1] == 1 else r_edge_features.mean(dim=-1)
                else:
                    r_edge_weights_raw = r_edge_features
                
                                              
                
                                                      
                                             
                target_nodes = r_edge_index[1]
                num_nodes = adj_v.size(0)
                
                                          
                r_edge_weights_abs = torch.abs(r_edge_weights_raw)
                weighted_in_degree = torch.zeros(num_nodes, device=node_features.device, dtype=torch.float)
                weighted_in_degree.scatter_add_(0, target_nodes, r_edge_weights_abs.float())
                
                                    
                epsilon = 1e-8
                weighted_in_degree = weighted_in_degree + epsilon
                
                                                               
                                                 
                r_edge_weights_normalized = r_edge_weights_raw / weighted_in_degree[target_nodes]
                                                                      
                
                              
                adj_v_r = torch.sparse_coo_tensor(
                    indices=r_edge_index,
                    values=r_edge_weights_normalized,
                    size=adj_v.size()
                ).coalesce()
                
                              
                out_node_features += torch.sparse.mm(adj_v_r, transformed_nodes)
        
                    
        out_node_features += self.self_loop_weight(node_features)
        
                                          
        return out_node_features, edge_features
  
class TSLANet(nn.Module):
                                                     
    def __init__(self, input_length=300, num_nodes=116, nhid=128, nclass=3, dropout=0.3, 
                 node_feature_dim=128, num_layers=3, leaky_slope=0.1, 
                 topk_strategy='local', global_topk_ratio=0.3):              
        super(TSLANet, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.leaky_slope = leaky_slope
        
                      
        self.num_orig_bands = 1                                                                                                
                                                           
            
                                                                                                    
        self.num_sub_bands_per_orig_band = 2                                      
                                                                      
        self.num_statistical_methods = 2                                                        
        self.total_num_relations = self.num_orig_bands * self.num_sub_bands_per_orig_band * self.num_statistical_methods

                                   
        self.node_feature_extractor = NodeFeatureExtractor(
            num_sub_bands=self.num_sub_bands_per_orig_band,
            time_length=input_length,
            hidden_dim=128,                 
            output_dim=node_feature_dim,
            dropout=dropout
        )
        
                             
        self.decomposers = nn.ModuleList([
            CascadeDecomposer(num_nodes) for _ in range(self.num_orig_bands)
        ])

                    
        self.topk_strategy = topk_strategy                      
        
                                                          
        self.k_ratio = 0.3
        
                                                           
        self.global_topk_ratio = global_topk_ratio


                                              
        self.layers = nn.ModuleList()
        current_node_dim = node_feature_dim
        current_edge_dim = 1                           
        
                     
        self.layers.append(RCensNetConv(current_node_dim, nhid, current_edge_dim, current_edge_dim, self.total_num_relations, node_layer=True))
        current_node_dim = nhid
        
                             
        for i in range(num_layers - 2):
            self.layers.append(RCensNetConv(current_node_dim, nhid, current_edge_dim, current_edge_dim, self.total_num_relations, node_layer=True))
            current_node_dim = nhid
        
                      
        self.layers.append(RCensNetConv(current_node_dim, nhid, current_edge_dim, current_edge_dim, self.total_num_relations, node_layer=True))


        self.bns_node = nn.ModuleList()
        for layer in self.layers:
                                     
            self.bns_node.append(nn.BatchNorm1d(nhid))

                                             
        self.classifier = nn.Sequential(
            nn.Linear(nhid, nhid // 2),
            nn.LeakyReLU(leaky_slope),
            nn.Dropout(dropout),
            nn.Linear(nhid // 2, nclass)
        )  
    
    def _compute_node_features(self, raw_ts_flat, num_timepoints):
        
        total_nodes = raw_ts_flat.shape[0]
                                                 
        x_reshaped = raw_ts_flat.view(total_nodes, 1, num_timepoints)
        x_reshaped_encoder = self.time_encoder(x_reshaped)
                                              
        return x_reshaped_encoder

    def forward(self, data: Batch,tau=None):
                         
        raw_ts_batched = data.raw_ts 
        batch_vec = data.batch 
        tau = tau
                                                                         
        num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        batch_size = data.num_graphs 
        time_length = raw_ts_batched.shape[-1]
        
               
        all_batch_edge_indices = []
        all_batch_edge_attrs = []
        all_batch_edge_types = []
                                               
        all_decomposed_bands = []
        
        node_offset_global = 0

                                                                   
        for i in range(batch_size):
            current_num_nodes = num_nodes_per_graph[i].item()
            
                         
            current_ts_unbatched = raw_ts_batched[node_offset_global : node_offset_global + current_num_nodes, :]
            current_ts_for_decomposer = current_ts_unbatched.unsqueeze(0)

                  
            all_sub_bands_for_current_graph = []
            for band_decomposer_idx in range(self.num_orig_bands):
                sub_bands_output = self.decomposers[band_decomposer_idx](current_ts_for_decomposer)
                all_sub_bands_for_current_graph.append(sub_bands_output)
            combined_sub_bands = torch.cat(all_sub_bands_for_current_graph, dim=1).squeeze(0)
            
                                                                        
            input_for_extractor = combined_sub_bands.permute(1, 0, 2)
            
                                  
            all_decomposed_bands.append(input_for_extractor)
            
            node_offset_global += current_num_nodes
        
                                         
        all_bands_stacked = torch.cat(all_decomposed_bands, dim=0)                                  
        X = self.node_feature_extractor(all_bands_stacked)             
        
                                                                    
        node_offset_global = 0         
        for i in range(batch_size):
            current_num_nodes = num_nodes_per_graph[i].item()
            
                          
            combined_sub_bands = all_decomposed_bands[i].permute(1, 0, 2)                            

            global_relation_counter = 0

            for sub_band_idx in range(self.num_sub_bands_per_orig_band):
                current_band_ts = combined_sub_bands[sub_band_idx, :, :]             
                fc_pearson = pearson_correlation(current_band_ts)                                                  
                fc_spearman = spearman_correlation(current_band_ts)
                                                                         
                current_band_fcns = [fc_spearman, fc_pearson]                                
                                                                                              
                for stat_method_idx, fcn_matrix_dense in enumerate(current_band_fcns):

                    num_nodes = fcn_matrix_dense.shape[0]
                                                       
                    if self.topk_strategy == 'local':
                                                               
                       
                        k = max(1, min(int(num_nodes * self.k_ratio), num_nodes - 1))

                        with torch.no_grad():             
                                                 
                            fcn_abs = torch.abs(fcn_matrix_dense)
                            fcn_abs.fill_diagonal_(0)          
                            
                                              
                            _, top_k_indices = torch.topk(fcn_abs, k=k, dim=1)
                            
                                               
                            row_indices = torch.arange(num_nodes, device=fcn_matrix_dense.device).view(-1, 1).repeat(1, k)
                            candidate_edge_index = torch.stack([row_indices.flatten(), top_k_indices.flatten()], dim=0).long()

                                                
                        candidate_edge_values = fcn_matrix_dense[candidate_edge_index[0], candidate_edge_index[1]].unsqueeze(-1)
                        
                    elif self.topk_strategy == 'global':
                                                              
                                                     
                        
                        with torch.no_grad():             
                                           
                            fcn_abs = torch.abs(fcn_matrix_dense)
                            fcn_abs.fill_diagonal_(0)          
                            
                                             
                                           
                            upper_tri_mask = torch.triu(torch.ones_like(fcn_abs, dtype=torch.bool), diagonal=1)
                            upper_tri_values = fcn_abs[upper_tri_mask]
                            upper_tri_indices = upper_tri_mask.nonzero(as_tuple=False)
                            
                                      
                            max_possible_edges = upper_tri_values.shape[0]
                            num_edges_to_keep = max(1, int(max_possible_edges * self.global_topk_ratio))
                            num_edges_to_keep = min(num_edges_to_keep, max_possible_edges)
                            
                                           
                            _, topk_edge_indices = torch.topk(upper_tri_values, k=num_edges_to_keep)
                            selected_indices = upper_tri_indices[topk_edge_indices]
                            
                                                    
                                       
                            forward_edges = selected_indices.t()                  
                            backward_edges = torch.stack([forward_edges[1], forward_edges[0]], dim=0)
                            candidate_edge_index = torch.cat([forward_edges, backward_edges], dim=1).long()

                                                
                        candidate_edge_values = fcn_matrix_dense[candidate_edge_index[0], candidate_edge_index[1]].unsqueeze(-1)
                    
                    else:
                        raise ValueError(f"Unsupported top-k strategy: {self.topk_strategy}. Supported: 'local', 'global'")
                    
                                        
                    scaled_edge_attr = candidate_edge_values

                                       
                    edge_type_r = torch.full((candidate_edge_index.shape[1],), global_relation_counter, 
                                            dtype=torch.long, device=raw_ts_batched.device)
                    
                    all_batch_edge_indices.append(candidate_edge_index + node_offset_global)
                    all_batch_edge_attrs.append(scaled_edge_attr)
                    all_batch_edge_types.append(edge_type_r)
                    
                                                                              
                    
                    global_relation_counter += 1

            node_offset_global += current_num_nodes 
        
                                                                
                   
        final_batch_edge_index = torch.cat(all_batch_edge_indices, dim=1)
        final_batch_edge_attr = torch.cat(all_batch_edge_attrs, dim=0)
        final_batch_edge_type = torch.cat(all_batch_edge_types, dim=0)

                                    
        total_nodes_in_batch = data.num_nodes 
        total_edges_in_batch = final_batch_edge_index.shape[1]

                                    
        adj_v_edge_index, adj_v_edge_weight = gcn_norm(
            edge_index=final_batch_edge_index, num_nodes=total_nodes_in_batch, add_self_loops=True
        )

                                
        Z = final_batch_edge_attr 
        
                                             
        adj_v_b = torch.sparse_coo_tensor(adj_v_edge_index, adj_v_edge_weight, 
                                        (total_nodes_in_batch, total_nodes_in_batch), 
                                        device=raw_ts_batched.device).coalesce()
        
                                                  
                                 
        adj_e_b = None                  
        T_b = None                 
                
                                                                                 
                                                                                        
                                                                                           
                                                                                 
                                                                                        
                                                                                           
                                                             
                                                                                    
                                                                                       

                                       
        current_X, current_Z = X, Z



            
        for layer_idx, layer in enumerate(self.layers):
                                                    
            current_X, current_Z = layer(current_X, current_Z, adj_e_b, adj_v_b, T_b, final_batch_edge_index, final_batch_edge_type)
            current_X = F.leaky_relu(current_X, negative_slope=self.leaky_slope)
            current_X = F.dropout(current_X, self.dropout, training=self.training)            
        graph_embedding = global_mean_pool(current_X, batch_vec)
        output = self.classifier(graph_embedding)
        
        return output
             

    def loss(self, pred, label, weights_fold=None, train=True, loss_type='focal', cost_matrix=None, class_counts=None):
        
        if loss_type == 'ce':
                     
            if weights_fold is None:
                criterion = nn.CrossEntropyLoss()
            else:
                weights_fold = torch.tensor(weights_fold, dtype=torch.float32).to(pred.device)
                criterion = nn.CrossEntropyLoss(weight=weights_fold)
            loss = criterion(pred, label)
            
        elif loss_type == 'focal':
                        
            criterion = FocalLoss()
            loss = criterion(pred, label)
            
        elif loss_type == 'cost_sensitive':
                         
            criterion = CostSensitiveLoss(cost_matrix=cost_matrix)
            loss = criterion(pred, label, class_counts=class_counts)
            
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported: 'ce', 'focal', 'cost_sensitive'")

        return loss