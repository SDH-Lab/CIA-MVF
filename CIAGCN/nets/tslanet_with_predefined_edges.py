import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from .gcn_net import TSLANet
from .correlations import pearson_correlation, spearman_correlation
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class TSLANetWithPredefinedEdges(TSLANet):
    def __init__(self, input_length=120, num_nodes=116, nhid=128, nclass=2, dropout=0.3, 
                 node_feature_dim=128, num_layers=3, leaky_slope=0.1,
                 topk_strategy='local', global_topk_ratio=0.3):
        super(TSLANetWithPredefinedEdges, self).__init__(
            input_length=input_length,
            num_nodes=num_nodes,
            nhid=nhid,
            nclass=nclass,
            dropout=dropout,
            node_feature_dim=node_feature_dim,
            num_layers=num_layers,
            leaky_slope=leaky_slope,
            topk_strategy=topk_strategy,
            global_topk_ratio=global_topk_ratio
        )
    
    def _build_edge_index_from_ts(self, data, tau=None):
        batch_size = data.num_graphs
        num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        raw_ts_batched = data.raw_ts
        
        all_batch_edge_indices = []
        all_batch_edge_attrs = []
        all_batch_edge_types = []
        
        node_offset_global = 0
        
        for i in range(batch_size):
            current_num_nodes = num_nodes_per_graph[i].item()
            current_ts = raw_ts_batched[node_offset_global:node_offset_global + current_num_nodes, :]
            

            fc_pearson = pearson_correlation(current_ts)

            k_ratio = 0.3
            k = max(1, min(int(current_num_nodes * k_ratio), current_num_nodes - 1))
            
            with torch.no_grad():
                fcn_abs = torch.abs(fc_pearson)
                fcn_abs.fill_diagonal_(0)  

                _, top_k_indices = torch.topk(fcn_abs, k=k, dim=1)
                

                row_indices = torch.arange(current_num_nodes, device=fc_pearson.device).view(-1, 1).repeat(1, k)
                candidate_edge_index = torch.stack([row_indices.flatten(), top_k_indices.flatten()], dim=0).long()

                candidate_edge_values = fc_pearson[candidate_edge_index[0], candidate_edge_index[1]].unsqueeze(-1)

                edge_type_r = torch.zeros(candidate_edge_index.shape[1], dtype=torch.long, device=raw_ts_batched.device)
                
                all_batch_edge_indices.append(candidate_edge_index + node_offset_global)
                all_batch_edge_attrs.append(candidate_edge_values)
                all_batch_edge_types.append(edge_type_r)
            
            node_offset_global += current_num_nodes

        final_batch_edge_index = torch.cat(all_batch_edge_indices, dim=1)
        final_batch_edge_attr = torch.cat(all_batch_edge_attrs, dim=0)
        final_batch_edge_type = torch.cat(all_batch_edge_types, dim=0)
        
        return final_batch_edge_index, final_batch_edge_attr, final_batch_edge_type
    
    def forward(self, data: Batch, tau=None, use_predefined_edges=False):

        if use_predefined_edges and hasattr(data, 'edge_index') and data.edge_index is not None:
            return self._forward_with_predefined_edges(data, tau)
        else:
            return super().forward(data, tau)
    
    def _forward_with_predefined_edges(self, data: Batch, tau=None):
       
        raw_ts_batched = data.raw_ts 
        batch_vec = data.batch 
        tau = tau
        
        num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        batch_size = data.num_graphs 
        time_length = raw_ts_batched.shape[-1]

        all_batch_node_features = []
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
            X_current_graph = self.node_feature_extractor(input_for_extractor)
            all_batch_node_features.append(X_current_graph)
            
            node_offset_global += current_num_nodes

        X_batched = torch.cat(all_batch_node_features, dim=0)

        final_batch_edge_index = data.edge_index
        final_batch_edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        final_batch_edge_type = data.edge_type if hasattr(data, 'edge_type') and data.edge_type is not None else None
   
        if final_batch_edge_attr is None:
            final_batch_edge_attr = torch.ones(final_batch_edge_index.shape[1], 1, device=raw_ts_batched.device)

        if final_batch_edge_type is None:
            final_batch_edge_type = torch.zeros(final_batch_edge_index.shape[1], dtype=torch.long, device=raw_ts_batched.device)

        Z_batched = final_batch_edge_attr

        total_nodes_in_batch = X_batched.shape[0]
        total_edges_in_batch = Z_batched.shape[0]
  
        row_T = torch.cat([final_batch_edge_index[0], final_batch_edge_index[1]])
        col_T = torch.arange(total_edges_in_batch, device=row_T.device).repeat(2)
        T_indices = torch.stack([row_T, col_T])
        T_values = torch.ones_like(row_T, dtype=torch.float32, device=raw_ts_batched.device)
        
        temp_T = torch.sparse_coo_tensor(T_indices, T_values, 
                                        (total_nodes_in_batch, total_edges_in_batch), 
                                        device=raw_ts_batched.device).coalesce()
        
        adj_e_raw = torch.sparse.mm(temp_T.t(), temp_T)
        adj_e_edge_index_raw = adj_e_raw.indices()
        
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        adj_e_edge_index, adj_e_edge_weight = gcn_norm(
            edge_index=adj_e_edge_index_raw, num_nodes=total_edges_in_batch, add_self_loops=True
        )
        
        adj_e_b = torch.sparse_coo_tensor(adj_e_edge_index, adj_e_edge_weight, 
                                        (total_edges_in_batch, total_edges_in_batch), 
                                        device=raw_ts_batched.device).coalesce()
        

        adj_v_b = torch.sparse_coo_tensor(final_batch_edge_index, 
                                        torch.ones(final_batch_edge_index.shape[1], dtype=torch.float32, device=raw_ts_batched.device),
                                        (total_nodes_in_batch, total_nodes_in_batch), 
                                        device=raw_ts_batched.device).coalesce()
        
 
        T_b = temp_T
        

        current_X = X_batched
        current_Z = Z_batched
        
        for layer_idx, layer in enumerate(self.layers):

            current_X, current_Z = layer(current_X, current_Z, adj_e_b, adj_v_b, T_b, 
                                       final_batch_edge_index, final_batch_edge_type)
            

            if layer.node_layer and layer_idx < len(self.bns_node):
                current_X = self.bns_node[layer_idx](current_X)
            elif not layer.node_layer and layer_idx < len(self.bns_edge):
                current_Z = self.bns_edge[layer_idx](current_Z)
            

            if layer.node_layer:
                current_X = F.leaky_relu(current_X, negative_slope=self.leaky_slope)
                current_X = F.dropout(current_X, p=self.dropout, training=self.training)
            else:
                current_Z = F.leaky_relu(current_Z, negative_slope=self.leaky_slope)
                current_Z = F.dropout(current_Z, p=self.dropout, training=self.training)
        

        graph_embeddings = global_mean_pool(current_X, batch_vec)
        

        output = self.classifier(graph_embeddings)
        
        return output
    
    def _build_adjacency_matrices(self, edge_index, edge_type, num_nodes, num_edges, layer_idx, tau):
  
        relation_idx = layer_idx % self.total_num_relations
        
        edge_type_mask = (edge_type == relation_idx)
        filtered_edge_index = edge_index[:, edge_type_mask]
        
        if filtered_edge_index.shape[1] == 0:
            adj_e = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros(0, dtype=torch.float32, device=edge_index.device),
                (num_edges, num_edges)
            )
        else:
            edge_indices = torch.arange(filtered_edge_index.shape[1], device=edge_index.device)
            edge_to_edge_indices = torch.stack([edge_indices, edge_indices])
            edge_values = torch.ones(filtered_edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
            adj_e = torch.sparse_coo_tensor(edge_to_edge_indices, edge_values, (num_edges, num_edges))
        
        if filtered_edge_index.shape[1] == 0:
            adj_v = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros(0, dtype=torch.float32, device=edge_index.device),
                (num_nodes, num_nodes)
            )
        else:
            adj_v = torch.sparse_coo_tensor(
                filtered_edge_index,
                torch.ones(filtered_edge_index.shape[1], dtype=torch.float32, device=edge_index.device),
                (num_nodes, num_nodes)
            )

        if filtered_edge_index.shape[1] == 0:
            T = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=edge_index.device),
                torch.zeros(0, dtype=torch.float32, device=edge_index.device),
                (num_edges, num_nodes)
            )
        else:
            edge_indices = torch.arange(filtered_edge_index.shape[1], device=edge_index.device)
            T_indices = torch.stack([edge_indices, filtered_edge_index[0]])
            T_values = torch.ones(filtered_edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
            T = torch.sparse_coo_tensor(T_indices, T_values, (num_edges, num_nodes))
        
        return adj_e, adj_v, T