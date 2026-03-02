import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class GradientImportanceCalculator:
    
    def __init__(self, model, device, target_class_idx: Optional[int] = None):
        self.model = model
        self.device = device
        self.target_class_idx = target_class_idx
        
    def compute_gradient_importance(self, data, tau=None, method='abs', 
                                  return_gradients=False, normalize=True):
        batch_size = data.num_graphs
        num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        
      
        original_training_mode = self.model.training
        self.model.train()
        
  
        base_model = self.model.module.base_model if hasattr(self.model, 'module') else self.model.base_model
        
        try:
            importance_scores, gradients = self._compute_batch_importance(
                base_model, data, tau, method, normalize, return_gradients
            )
            
            self.model.train(original_training_mode)
            
            if return_gradients:
                return importance_scores, gradients
            else:
                return importance_scores
                
        except Exception as e:
            print(f"Gradient importance calculation error: {e}")
            fallback_scores = []
            node_offset = 0
            for i in range(batch_size):
                current_num_nodes = num_nodes_per_graph[i].item()
                fallback_scores.append(
                    torch.randn(current_num_nodes, device=self.device).abs()
                )
                node_offset += current_num_nodes
            
            self.model.train(original_training_mode)
            return fallback_scores
    
    def _compute_batch_importance(self, base_model, data, tau, method, normalize, return_gradients):
        original_grad_state = {}
        for param in base_model.parameters():
            original_grad_state[param] = param.grad
        
        try:
            raw_ts = data.raw_ts.clone().detach().requires_grad_(True)
            
            data_with_grad = data.clone()
            data_with_grad.raw_ts = raw_ts
            
            output = base_model(data_with_grad, tau, use_predefined_edges=False)

            if self.target_class_idx is not None:
                target_logits = output[:, self.target_class_idx]
            else:
                target_logits = output.gather(1, data.y.view(-1, 1)).squeeze()
            
            target_logits.sum().backward(retain_graph=False)

            gradients = raw_ts.grad
            
            del raw_ts, data_with_grad, output, target_logits
            
        finally:
            for param in base_model.parameters():
                param.grad = original_grad_state.get(param, None)
        
        if gradients is None:
            batch_size = data.num_graphs
            num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
            fallback_scores = []
            node_offset = 0
            for i in range(batch_size):
                current_num_nodes = num_nodes_per_graph[i].item()
                fallback_scores.append(
                    torch.randn(current_num_nodes, device=self.device).abs()
                )
                node_offset += current_num_nodes
            return fallback_scores, None
        
        if method == 'abs':
            importance_all = torch.abs(gradients).mean(dim=1)  
        elif method == 'square':
            importance_all = (gradients ** 2).mean(dim=1)
        elif method == 'raw':
            importance_all = gradients.mean(dim=1)
        else:
            importance_all = torch.abs(gradients).mean(dim=1)
        
        batch_size = data.num_graphs
        num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        
        importance_scores = []
        gradients_list = [] if return_gradients else None
        
        node_offset = 0
        for i in range(batch_size):
            current_num_nodes = num_nodes_per_graph[i].item()
            current_importance = importance_all[node_offset:node_offset + current_num_nodes]
            
            if normalize and current_importance.sum() > 0:
                current_importance = current_importance / current_importance.sum()
            
            importance_scores.append(current_importance)
            
            if return_gradients:
                current_gradients = gradients[node_offset:node_offset + current_num_nodes].detach()
                gradients_list.append(current_gradients)
            
            node_offset += current_num_nodes
        
        return importance_scores, gradients_list
    
    def _extract_node_features(self, base_model, data, tau):
        try:
            raw_ts_batched = data.raw_ts
            batch_size = data.num_graphs
            num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
            
            all_batch_node_features = []
            node_offset_global = 0
            
            for i in range(batch_size):
                current_num_nodes = num_nodes_per_graph[i].item()

                current_ts_unbatched = raw_ts_batched[node_offset_global:node_offset_global + current_num_nodes, :]
                current_ts_for_decomposer = current_ts_unbatched.unsqueeze(0)

                all_sub_bands_for_current_graph = []
                for band_decomposer_idx in range(base_model.num_orig_bands):
                    sub_bands_output = base_model.decomposers[band_decomposer_idx](current_ts_for_decomposer)
                    all_sub_bands_for_current_graph.append(sub_bands_output)
                combined_sub_bands = torch.cat(all_sub_bands_for_current_graph, dim=1).squeeze(0)

                input_for_extractor = combined_sub_bands.permute(1, 0, 2)
                X_current_graph = base_model.node_feature_extractor(input_for_extractor)
                all_batch_node_features.append(X_current_graph)
                
                node_offset_global += current_num_nodes

            X_batched = torch.cat(all_batch_node_features, dim=0)
            return X_batched
            
        except Exception as e:
            print(f"Point feature extraction error: {e}")
            total_nodes = data.ptr[-1].item()
            feature_dim = 128  
            return torch.randn(total_nodes, feature_dim, device=self.device)


class GradientBasedCausalTSLANet(nn.Module):
    
    def __init__(self, input_length=120, num_nodes=116, nhid=128, nclass=2, dropout=0.3, 
                 node_feature_dim=128, num_layers=3, leaky_slope=0.1, 
                 alpha=0.5, beta=0.5, top_k_important=20, top_k_background=40, gradient_method='abs',
                 use_kl_consistency=True, kl_temperature=2.0, kl_gamma=0.1,
                 topk_strategy='local', global_topk_ratio=0.3):
        super(GradientBasedCausalTSLANet, self).__init__()

        from .tslanet_with_predefined_edges import TSLANetWithPredefinedEdges

        self.base_model = TSLANetWithPredefinedEdges(
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

        self.alpha = alpha  
        self.beta = beta   
        self.top_k_important = top_k_important  
        self.top_k_background = top_k_background  
        self.gradient_method = gradient_method  
        self.mixing_strategy = 'anatomical'  
        
        # KL散度一致性参数
        self.use_kl_consistency = use_kl_consistency  
        self.kl_temperature = kl_temperature  
        self.kl_gamma = kl_gamma  
        

        self.importance_calculator = None
        self.memory_bank = {}
        self.memory_bank_size = 3  
        self.memory_bank_update_freq = 5  
        self.memory_bank_counter = 0  
        self.memory_bank_enabled = True  
        

        self.memory_bank_stats = {
            'total_updates': 0,
            'total_uses': 0,
            'uses_per_class': defaultdict(int),
            'updates_per_class': defaultdict(int)
        }
        
        self.mixing_stats = {
            'causal_total': 0,  
            'causal_batches': 0,  
            'background_total': 0,
            'background_batches': 0
        }
        
    def _init_importance_calculator(self, device):
        if self.importance_calculator is None:
            self.importance_calculator = GradientImportanceCalculator(
                model=self, device=device, target_class_idx=None
            )
    
    def update_memory_bank(self, data, importance_scores=None):
        if not self.memory_bank_enabled:
            return
        
        batch_size = data.num_graphs
        labels = data.y.cpu()  
        
        for i in range(batch_size):
            label = labels[i].item()
            
            sample_start = data.ptr[i]
            sample_end = data.ptr[i + 1]

            sample_data = {
                'raw_ts': data.raw_ts[sample_start:sample_end].cpu().clone(), 
                'y': data.y[i].cpu().clone(),
                'num_nodes': (sample_end - sample_start).item()
            }
            
            if label not in self.memory_bank:
                self.memory_bank[label] = []
            
            self.memory_bank[label].append(sample_data)
            
            if len(self.memory_bank[label]) > self.memory_bank_size:
                self.memory_bank[label].pop(0)  

            self.memory_bank_stats['updates_per_class'][label] += 1
        
        self.memory_bank_stats['total_updates'] += 1
    
    def get_sample_from_memory_bank(self, target_label, device, tau=None):
        if not self.memory_bank_enabled or target_label not in self.memory_bank:
            return None, None, False
        
        if len(self.memory_bank[target_label]) == 0:
            return None, None, False

        idx = torch.randint(0, len(self.memory_bank[target_label]), (1,)).item()
        sample_data_cpu = self.memory_bank[target_label][idx]

        sample_data = {
            'raw_ts': sample_data_cpu['raw_ts'].to(device),
            'y': sample_data_cpu['y'].to(device),
            'num_nodes': sample_data_cpu['num_nodes']
        }

        try:
            from torch_geometric.data import Data, Batch

            single_sample = Data(
                raw_ts=sample_data['raw_ts'].clone(),
                y=sample_data['y'].clone(),
                num_nodes=sample_data['num_nodes']
            )
            single_data_batched = Batch.from_data_list([single_sample])

            sample_importance = self.compute_node_importance(single_data_batched, tau)[0]
            
        except Exception as e:
            print(f"Warning: Dynamic importance calculation failed; use random initialization: {e}")
            sample_importance = torch.randn(sample_data['num_nodes'], device=device).abs()

        self.memory_bank_stats['total_uses'] += 1
        self.memory_bank_stats['uses_per_class'][target_label] += 1
        
        return sample_data, sample_importance, True
    
    def print_memory_bank_stats(self):
        for label, samples in self.memory_bank.items():
            print(f"  Class {label}: {len(samples)} samples")
        print(f"Usage count per class:")
        for label, count in self.memory_bank_stats['uses_per_class'].items():
            print(f"  Class {label}: {count} times")
        print("=" * 30)
    
    def initialize_memory_bank_from_dataset(self, train_loader, device, tau=None):
        if not self.memory_bank_enabled:
            print("Memory Bank is not enabled, skipping pre-filling")
            return
        
        print("\n" + "="*60)
        print("Starting Memory Bank pre-filling (fast mode: only collecting samples)...")
        print("="*60)
        
        samples_by_class = defaultdict(list)
        
        expected_classes = set()
        print("Scanning dataset to determine all classes...")
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                labels = data.y.cpu()
                for label in labels:
                    expected_classes.add(label.item())
                if batch_idx >= 10:
                    break
        

        
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                batch_size = data.num_graphs
                labels = data.y.cpu()

                for i in range(batch_size):
                    label = labels[i].item()

                    if len(samples_by_class[label]) >= self.memory_bank_size:
                        continue

                    sample_start = data.ptr[i]
                    sample_end = data.ptr[i + 1]
                    
                    sample_data = {
                        'raw_ts': data.raw_ts[sample_start:sample_end].cpu().clone(),
                        'y': data.y[i].cpu().clone(),
                        'num_nodes': (sample_end - sample_start).item()
                    }
                    
                    samples_by_class[label].append(sample_data)
                
                # Check if all expected classes have been collected
                all_classes_filled = all(
                    len(samples_by_class[label]) >= self.memory_bank_size 
                    for label in expected_classes
                )

                if all_classes_filled:
                    print(f" All class samples collected! Iterated through {batch_idx + 1} batches")
                    break
                elif batch_idx >= 200:
                    print(f" Iterated through {batch_idx + 1} batches, stopping collection")
                    for label in sorted(expected_classes):
                        count = len(samples_by_class[label]) if label in samples_by_class else 0
                        status = "✓" if count >= self.memory_bank_size else "✗"
                        print(f"  Class {label}: {count}/{self.memory_bank_size} {status}")
                    break

  
        for label in sorted(samples_by_class.keys()):
            print(f"  Class {label}: {len(samples_by_class[label])} samples")
        
      
        
        for label, sample_list in samples_by_class.items():
            self.memory_bank[label] = []
            
            for sample_data in sample_list[:self.memory_bank_size]:
                # Directly add to Memory Bank (without calculating importance)
                self.memory_bank[label].append(sample_data)
                self.memory_bank_stats['updates_per_class'][label] += 1
        
        self.memory_bank_stats['total_updates'] += 1
        
       
    
    def compute_node_importance(self, data, tau=None, device=None):

        if device is None:
            device = next(self.parameters()).device

        self._init_importance_calculator(device)

        importance_scores = self.importance_calculator.compute_gradient_importance(
            data, tau, method=self.gradient_method, normalize=True
        )
        
        return importance_scores
    
    def _apply_mixing_strategy(self, data, importance_scores, mixing_type='background'):

        if self.mixing_strategy == 'anatomical':
            return self._anatomical_mixing(data, importance_scores, mixing_type)
        elif self.mixing_strategy == 'element_wise':
            return self._element_wise_mixing(data, importance_scores, mixing_type)
        elif self.mixing_strategy == 'hybrid':
            return self._hybrid_mixing(data, importance_scores, mixing_type)
        elif self.mixing_strategy == 'union':
            return self._union_mixing(data, importance_scores, mixing_type)
        else:
            return self._anatomical_mixing(data, importance_scores, mixing_type)
    
    def _anatomical_mixing(self, data, importance_scores, mixing_type='background'):
        batch_size = data.num_graphs
        raw_ts_batched = data.raw_ts.clone()
        
        if batch_size < 2:
            return data

        if mixing_type == 'causal':

            labels = data.y
            unique_labels = torch.unique(labels)
            same_label_indices = []
            for label in unique_labels:
                label_mask = (labels == label)
                label_indices = torch.where(label_mask)[0]
                if len(label_indices) >= 2:
                    same_label_indices.append(label_indices[torch.randperm(len(label_indices))[:2]])
            
            if not same_label_indices:
                return data
            
            selected_group = same_label_indices[torch.randint(0, len(same_label_indices), (1,)).item()]
            sample1_idx, sample2_idx = selected_group[0].item(), selected_group[1].item()
        else:
            sample_indices = torch.randperm(batch_size)[:2]
            sample1_idx, sample2_idx = sample_indices[0].item(), sample_indices[1].item()

        sample1_start = data.ptr[sample1_idx]
        sample2_start = data.ptr[sample2_idx]

        sample1_importance = importance_scores[sample1_idx]
        sample2_importance = importance_scores[sample2_idx]

        if mixing_type == 'causal':
            sample1_nodes = torch.argsort(sample1_importance, descending=True)[:self.top_k_important]
            sample2_nodes = torch.argsort(sample2_importance, descending=True)[:self.top_k_important]
        else:
            sample1_nodes = torch.argsort(sample1_importance)[:self.top_k_background]
            sample2_nodes = torch.argsort(sample2_importance)[:self.top_k_background]

        sample1_set = set(sample1_nodes.cpu().numpy())
        sample2_set = set(sample2_nodes.cpu().numpy())

        common_nodes = sample1_set & sample2_set
        
        if len(common_nodes) > 0:

            for node_idx in common_nodes:
                if (node_idx < len(sample1_importance) and 
                    node_idx < len(sample2_importance)):
                    temp_ts = raw_ts_batched[sample1_start + node_idx].clone()
                    raw_ts_batched[sample1_start + node_idx] = raw_ts_batched[sample2_start + node_idx]
                    raw_ts_batched[sample2_start + node_idx] = temp_ts

        mixed_data = data.clone()
        mixed_data.raw_ts = raw_ts_batched

        if hasattr(mixed_data, 'edge_index'):
            mixed_data.edge_index = None
        if hasattr(mixed_data, 'edge_attr'):
            mixed_data.edge_attr = None
        if hasattr(mixed_data, 'edge_type'):
            mixed_data.edge_type = None
        
        return mixed_data
    
    def _element_wise_mixing(self, data, importance_scores, mixing_type='background'):
        batch_size = data.num_graphs
        raw_ts_batched = data.raw_ts.clone()
        
        if batch_size < 2:
            return data

        if mixing_type == 'causal':
            labels = data.y
            unique_labels = torch.unique(labels)
            same_label_indices = []
            for label in unique_labels:
                label_mask = (labels == label)
                label_indices = torch.where(label_mask)[0]
                if len(label_indices) >= 2:
                    same_label_indices.append(label_indices[torch.randperm(len(label_indices))[:2]])
            
            if not same_label_indices:
                return data
            
            selected_group = same_label_indices[torch.randint(0, len(same_label_indices), (1,)).item()]
            sample1_idx, sample2_idx = selected_group[0].item(), selected_group[1].item()
        else:
            sample_indices = torch.randperm(batch_size)[:2]
            sample1_idx, sample2_idx = sample_indices[0].item(), sample_indices[1].item()

        sample1_start = data.ptr[sample1_idx]
        sample2_start = data.ptr[sample2_idx]

        sample1_importance = importance_scores[sample1_idx]
        sample2_importance = importance_scores[sample2_idx]

        if mixing_type == 'causal':
            sample1_nodes = torch.argsort(sample1_importance, descending=True)[:self.top_k_important]
            sample2_nodes = torch.argsort(sample2_importance, descending=True)[:self.top_k_important]
        else:
            sample1_nodes = torch.argsort(sample1_importance)[:self.top_k_background]
            sample2_nodes = torch.argsort(sample2_importance)[:self.top_k_background]

        for node1, node2 in zip(sample1_nodes, sample2_nodes):
            if node1 < len(sample1_importance) and node2 < len(sample2_importance):
                temp_ts = raw_ts_batched[sample1_start + node1].clone()
                raw_ts_batched[sample1_start + node1] = raw_ts_batched[sample2_start + node2]
                raw_ts_batched[sample2_start + node2] = temp_ts

        mixed_data = data.clone()
        mixed_data.raw_ts = raw_ts_batched

        if hasattr(mixed_data, 'edge_index'):
            mixed_data.edge_index = None
        if hasattr(mixed_data, 'edge_attr'):
            mixed_data.edge_attr = None
        if hasattr(mixed_data, 'edge_type'):
            mixed_data.edge_type = None
        
        return mixed_data
    
    def _hybrid_mixing(self, data, importance_scores, mixing_type='background'):
    
        anatomical_result = self._anatomical_mixing(data, importance_scores, mixing_type)
        
        if mixing_type == 'causal':
            sample1_nodes = torch.argsort(importance_scores[0], descending=True)[:self.top_k_important]
            sample2_nodes = torch.argsort(importance_scores[1], descending=True)[:self.top_k_important]
        else:
            sample1_nodes = torch.argsort(importance_scores[0])[:self.top_k_background]
            sample2_nodes = torch.argsort(importance_scores[1])[:self.top_k_background]
        
        sample1_set = set(sample1_nodes.cpu().numpy())
        sample2_set = set(sample2_nodes.cpu().numpy())
        common_nodes = sample1_set & sample2_set
        
        if len(common_nodes) < self.top_k_background // 2:
            return self._element_wise_mixing(data, importance_scores, mixing_type)
        else:
            return anatomical_result
    
    def _union_mixing(self, data, importance_scores, mixing_type='background'):
        batch_size = data.num_graphs
        raw_ts_batched = data.raw_ts.clone()
        device = raw_ts_batched.device
        
        if batch_size < 1:
            return data
    
        use_memory_bank = False
        memory_bank_sample = None
        memory_bank_importance = None
        
        if mixing_type == 'causal':
            labels = data.y
            unique_labels = torch.unique(labels)
            same_label_indices = []
            for label in unique_labels:
                label_mask = (labels == label)
                label_indices = torch.where(label_mask)[0]
                if len(label_indices) >= 2:
                    same_label_indices.append(label_indices[torch.randperm(len(label_indices))[:2]])
            
            if same_label_indices:
                selected_group = same_label_indices[torch.randint(0, len(same_label_indices), (1,)).item()]
                sample1_idx, sample2_idx = selected_group[0].item(), selected_group[1].item()
            else:
                if self.memory_bank_enabled:
                    sample1_idx = torch.randint(0, batch_size, (1,)).item()
                    target_label = labels[sample1_idx].item()
                    

                    memory_bank_sample, memory_bank_importance, success = self.get_sample_from_memory_bank(
                        target_label, device, tau=None  
                    )
                    
                    if success:
                        use_memory_bank = True
                        sample2_idx = None  
                    else:
                        return data
                else:
                    return data
        else:
            if batch_size < 2:
                return data
            sample_indices = torch.randperm(batch_size)[:2]
            sample1_idx, sample2_idx = sample_indices[0].item(), sample_indices[1].item()
        
        sample1_start = data.ptr[sample1_idx]
        sample1_importance = importance_scores[sample1_idx]
        
        if use_memory_bank:
            sample2_raw_ts = memory_bank_sample['raw_ts']  
            sample2_importance = memory_bank_importance
        else:
            sample2_start = data.ptr[sample2_idx]
            sample2_importance = importance_scores[sample2_idx]
        
        # 选择重要脑区
        if mixing_type == 'causal':
            sample1_nodes = torch.argsort(sample1_importance, descending=True)[:self.top_k_important]
            sample2_nodes = torch.argsort(sample2_importance, descending=True)[:self.top_k_important]
        else:
            sample1_nodes = torch.argsort(sample1_importance)[:self.top_k_background]
            sample2_nodes = torch.argsort(sample2_importance)[:self.top_k_background]
        
        sample1_set = set(sample1_nodes.cpu().numpy())
        sample2_set = set(sample2_nodes.cpu().numpy())
        
        union_nodes = sample1_set | sample2_set
        
        if len(union_nodes) > 0:
            for node_idx in union_nodes:
                if use_memory_bank:
                    if (node_idx < len(sample1_importance) and 
                        node_idx < len(sample2_importance)):
                        raw_ts_batched[sample1_start + node_idx] = sample2_raw_ts[node_idx].clone()
                else:
                    if (node_idx < len(sample1_importance) and 
                        node_idx < len(sample2_importance)):
                        temp_ts = raw_ts_batched[sample1_start + node_idx].clone()
                        raw_ts_batched[sample1_start + node_idx] = raw_ts_batched[sample2_start + node_idx]
                        raw_ts_batched[sample2_start + node_idx] = temp_ts
                    elif node_idx < len(sample1_importance):
                        raw_ts_batched[sample2_start + node_idx] = raw_ts_batched[sample1_start + node_idx].clone()
                    elif node_idx < len(sample2_importance):
                        raw_ts_batched[sample1_start + node_idx] = raw_ts_batched[sample2_start + node_idx].clone()
        
        mixed_data = data.clone()
        mixed_data.raw_ts = raw_ts_batched
        
        if hasattr(mixed_data, 'edge_index'):
            mixed_data.edge_index = None
        if hasattr(mixed_data, 'edge_attr'):
            mixed_data.edge_attr = None
        if hasattr(mixed_data, 'edge_type'):
            mixed_data.edge_type = None
        
        return mixed_data
    
    def compute_kl_consistency_loss(self, original_logits, background_logits):

        eps = 1e-8
        

        p_original = F.softmax(original_logits / self.kl_temperature, dim=1)
        p_background = F.softmax(background_logits / self.kl_temperature, dim=1)
        
        p_original = torch.clamp(p_original, min=eps, max=1.0)
        p_background = torch.clamp(p_background, min=eps, max=1.0)
        
        kl_forward = F.kl_div(
            F.log_softmax(original_logits / self.kl_temperature, dim=1),
            p_background,
            reduction='batchmean'
        )
        
        kl_backward = F.kl_div(
            F.log_softmax(background_logits / self.kl_temperature, dim=1),
            p_original,
            reduction='batchmean'
        )

        kl_sym = 0.5 * (kl_forward + kl_backward)

        if torch.isnan(kl_sym):
            print(f"Warning: KL divergence is NaN. Original logits range: [{original_logits.min():.4f}, {original_logits.max():.4f}]")
            print(f"Background logits range: [{background_logits.min():.4f}, {background_logits.max():.4f}]")
            kl_sym = torch.tensor(0.0, device=kl_sym.device, dtype=kl_sym.dtype)

        kl_loss = self.kl_temperature ** 2 * kl_sym
        
        return kl_loss
    
    def _apply_mixing_strategy_multi_samples(self, data, importance_scores, mixing_type='causal'):

        batch_size = data.num_graphs
        raw_ts_batched = data.raw_ts.clone()
        device = raw_ts_batched.device
        
        if batch_size < 1:
            return data
        
        labels = data.y
        mixed_count = 0  

        for sample_idx in range(batch_size):
            sample_label = labels[sample_idx].item()
            
    
            if self.memory_bank_enabled:
                memory_bank_sample, memory_bank_importance, success = self.get_sample_from_memory_bank(
                    sample_label, device, tau=None
                )
                
                if success:
                 
                    sample_start = data.ptr[sample_idx]
                    sample_importance = importance_scores[sample_idx]
                    
            
                    if mixing_type == 'causal':
                        sample1_nodes = torch.argsort(sample_importance, descending=True)[:self.top_k_important]
                        sample2_nodes = torch.argsort(memory_bank_importance, descending=True)[:self.top_k_important]
                    else:
                        sample1_nodes = torch.argsort(sample_importance)[:self.top_k_background]
                        sample2_nodes = torch.argsort(memory_bank_importance)[:self.top_k_background]
                    
                    sample1_set = set(sample1_nodes.cpu().numpy())
                    sample2_set = set(sample2_nodes.cpu().numpy())
                    union_nodes = sample1_set | sample2_set

                    if len(union_nodes) > 0:
                        memory_bank_raw_ts = memory_bank_sample['raw_ts']
                        for node_idx in union_nodes:
                            if (node_idx < len(sample_importance) and 
                                node_idx < len(memory_bank_importance)):
                                raw_ts_batched[sample_start + node_idx] = memory_bank_raw_ts[node_idx].clone()
                        
                        mixed_count += 1

        mixed_data = data.clone()
        mixed_data.raw_ts = raw_ts_batched

        if hasattr(mixed_data, 'edge_index'):
            mixed_data.edge_index = None
        if hasattr(mixed_data, 'edge_attr'):
            mixed_data.edge_attr = None
        if hasattr(mixed_data, 'edge_type'):
            mixed_data.edge_type = None

        if mixed_count > 0 and hasattr(self, 'mixing_stats'):
            self.mixing_stats[f'{mixing_type}_total'] += mixed_count
            self.mixing_stats[f'{mixing_type}_batches'] += 1
        
        return mixed_data
    
    def background_mixing(self, data, importance_scores):
       
        batch_size = data.num_graphs
        raw_ts_batched = data.raw_ts.clone()
        device = raw_ts_batched.device
        
        if batch_size < 1:
            return data
        
        labels = data.y
        mixed_count = 0  
        memory_bank_usage = 0 
        batch_usage = 0  

        for sample_idx in range(batch_size):
            sample_label = labels[sample_idx].item()
            sample_start = data.ptr[sample_idx]
            sample_importance = importance_scores[sample_idx]
            
            different_class_sample = None
            different_class_importance = None
            use_memory_bank = False
            
            if self.memory_bank_enabled and len(self.memory_bank) > 0:

                available_classes = list(self.memory_bank.keys())
                different_classes = [c for c in available_classes if c != sample_label]
                
                if different_classes:
                    target_class = random.choice(different_classes)
                    different_class_sample, different_class_importance, success = self.get_sample_from_memory_bank(
                        target_class, device, tau=None
                    )
                    
                    if success:
                        use_memory_bank = True
                        memory_bank_usage += 1

            if not use_memory_bank:
                different_class_indices = [i for i in range(batch_size) 
                                          if labels[i].item() != sample_label and i != sample_idx]
                
                if different_class_indices:
                    other_idx = random.choice(different_class_indices)
                    other_start = data.ptr[other_idx]
                    different_class_importance = importance_scores[other_idx]
                    batch_usage += 1
                else:
                    continue
            
            sample_nodes = torch.argsort(sample_importance)[:self.top_k_background]
            other_nodes = torch.argsort(different_class_importance)[:self.top_k_background]
            
            sample_set = set(sample_nodes.cpu().numpy())
            other_set = set(other_nodes.cpu().numpy())
            union_nodes = sample_set | other_set
            
            if len(union_nodes) > 0:
                for node_idx in union_nodes:
                    if use_memory_bank:
                        if (node_idx < len(sample_importance) and 
                            node_idx < len(different_class_importance)):
                            raw_ts_batched[sample_start + node_idx] = different_class_sample['raw_ts'][node_idx].clone()
                    else:
                        if (node_idx < len(sample_importance) and 
                            node_idx < len(different_class_importance)):
                            raw_ts_batched[sample_start + node_idx] = raw_ts_batched[other_start + node_idx].clone()
                
                mixed_count += 1
        
        mixed_data = data.clone()
        mixed_data.raw_ts = raw_ts_batched
        
        if hasattr(mixed_data, 'edge_index'):
            mixed_data.edge_index = None
        if hasattr(mixed_data, 'edge_attr'):
            mixed_data.edge_attr = None
        if hasattr(mixed_data, 'edge_type'):
            mixed_data.edge_type = None
        
        if hasattr(self, 'mixing_stats'):
            self.mixing_stats['background_total'] += mixed_count
            if mixed_count > 0:
                self.mixing_stats['background_batches'] += 1
        
        return mixed_data
    
    def causal_mixing(self, data, importance_scores):
        return self._apply_mixing_strategy_multi_samples(data, importance_scores, 'causal')
    
    def forward(self, data, tau=None, return_importance=False):
        if return_importance:     
            importance_scores = self.compute_node_importance(data, tau)
            base_model = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
            return base_model(data, tau, use_predefined_edges=False), importance_scores
        else:
            base_model = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
            return base_model(data, tau, use_predefined_edges=False)
    
    def causal_training_step(self, data, tau=None, batch_idx=0):
        base_model = self.base_model.module if hasattr(self.base_model, 'module') else self.base_model
        original_output = base_model(data, tau, use_predefined_edges=False)
        loss_orig = F.cross_entropy(original_output, data.y)
        skip_causal_enhancement = (
            not self.enable_causal_enhancement or
            loss_orig.item() > 50.0 or
            batch_idx % self.causal_enhancement_frequency != 0
        )
        
        if skip_causal_enhancement:
            return {
                'loss': loss_orig,
                'loss_orig': loss_orig,
                'loss_intra': loss_orig * 0,  
                'loss_inter': loss_orig * 0,  
                'importance_scores': [],
                'logits': original_output  
            }
        
        try:
            importance_scores = self.compute_node_importance(data, tau)

            if self.memory_bank_enabled:
                self.memory_bank_counter += 1
                if self.memory_bank_counter % self.memory_bank_update_freq == 0:
                    self.update_memory_bank(data, importance_scores)

            background_mixed_data = self.background_mixing(data, importance_scores)
            background_output = base_model(background_mixed_data, tau, use_predefined_edges=False)
            
            if self.use_kl_consistency:
                loss_inter = self.compute_kl_consistency_loss(original_output, background_output)
            else:
                loss_inter = F.cross_entropy(background_output, data.y)
            del background_mixed_data, background_output
            torch.cuda.empty_cache()

            causal_mixed_data = self.causal_mixing(data, importance_scores)
            causal_output = base_model(causal_mixed_data, tau, use_predefined_edges=False)
            loss_intra = F.cross_entropy(causal_output, data.y)
            del causal_mixed_data, causal_output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in causal reinforcement calculation: {e}, Revert to the original training")
            return {
                'loss': loss_orig,
                'loss_orig': loss_orig,
                'loss_intra': loss_orig * 0,
                'loss_inter': loss_orig * 0,
                'importance_scores': [],
                'logits': original_output 
            }
        
        if self.use_kl_consistency:
            total_loss = loss_orig + self.alpha * loss_intra + self.kl_gamma * loss_inter
        else:
            total_loss = loss_orig + self.alpha * loss_intra + self.beta * loss_inter
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Anomalyloss detected")
            print(f"  loss_orig: {loss_orig.item():.6f}")
            print(f"  loss_intra: {loss_intra.item():.6f}")
            print(f"  loss_inter: {loss_inter.item():.6f}")
            print(f"  total_loss: {total_loss.item():.6f}")
            print(f"  alpha: {self.alpha}, beta: {self.beta}")
        
            total_loss = loss_orig
        
        if total_loss.item() > 100.0:
            print(f"Warning: Loss value is too large ({total_loss.item():.2f}), there may be a problem")
            print(f"  Original loss: {loss_orig.item():.6f}")
            print(f"  Causal mixing loss: {loss_intra.item():.6f}")
            print(f"  Background mixing loss: {loss_inter.item():.6f}")
        
        return {
            'loss': total_loss,
            'loss_orig': loss_orig,
            'loss_intra': loss_intra,
            'loss_inter': loss_inter,
            'importance_scores': importance_scores,
            'logits': original_output 
        }
    
    def loss(self, pred, label, weights_fold=None, train=True):
        if weights_fold is None:
            criterion = nn.CrossEntropyLoss()
        else:
            weights_fold = torch.tensor(weights_fold, dtype=torch.float32).to(pred.device)
            criterion = nn.CrossEntropyLoss(weight=weights_fold)
        
        return criterion(pred, label)


def monitor_importance_stability(importance_scores_history, epoch, sample_idx=0, top_k=10):
    if len(importance_scores_history) < 2:
        return {"status": "insufficient_data"}
    
    current_scores = importance_scores_history[-1][sample_idx]
    previous_scores = importance_scores_history[-2][sample_idx]

    current_top_k = torch.argsort(current_scores, descending=True)[:top_k]
    previous_top_k = torch.argsort(previous_scores, descending=True)[:top_k]

    overlap = len(set(current_top_k.cpu().numpy()) & set(previous_top_k.cpu().numpy()))
    overlap_ratio = overlap / top_k

    rank_changes = []
    for node_idx in current_top_k:
        if node_idx in previous_top_k:
            current_rank = (current_top_k == node_idx).nonzero(as_tuple=True)[0].item()
            previous_rank = (previous_top_k == node_idx).nonzero(as_tuple=True)[0].item()
            rank_changes.append(abs(current_rank - previous_rank))
    
    avg_rank_change = np.mean(rank_changes) if rank_changes else 0
    
    stability_info = {
        "epoch": epoch,
        "overlap_ratio": overlap_ratio,
        "avg_rank_change": avg_rank_change,
        "current_top_k": current_top_k.cpu().numpy(),
        "previous_top_k": previous_top_k.cpu().numpy(),
        "status": "stable" if overlap_ratio > 0.7 and avg_rank_change < 2 else "unstable"
    }
    
    return stability_info


def log_importance_analysis(importance_scores, labels, epoch, top_k=10):
    
    analysis_log = {
        "epoch": epoch,
        "class_analysis": {},
        "overall_stats": {}
    }
    
    if len(importance_scores) == 0:
        analysis_log["overall_stats"] = {
            "total_samples": 0,
            "total_nodes": 0,
            "mean_importance": 0.0,
            "std_importance": 0.0,
            "max_importance": 0.0,
            "min_importance": 0.0
        }
        return analysis_log
    
    unique_labels = torch.unique(labels)
    
    all_importance = torch.cat(importance_scores, dim=0)
    analysis_log["overall_stats"] = {
        "total_samples": len(importance_scores),
        "total_nodes": all_importance.shape[0],
        "mean_importance": all_importance.mean().item(),
        "std_importance": all_importance.std().item(),
        "max_importance": all_importance.max().item(),
        "min_importance": all_importance.min().item()
    }
    
    for label in unique_labels:
        label_mask = (labels == label)
        label_indices = torch.where(label_mask)[0]
        
        if len(label_indices) > 0:
            class_importance = []
            for idx in label_indices:
                class_importance.append(importance_scores[idx.item()])
            
            avg_importance = torch.stack(class_importance).mean(dim=0)

            top_k_nodes = torch.argsort(avg_importance, descending=True)[:top_k]
            top_k_scores = avg_importance[top_k_nodes]

            class_all_importance = torch.cat(class_importance, dim=0)
            
            analysis_log["class_analysis"][f"class_{label.item()}"] = {
                "num_samples": len(label_indices),
                "top_k_nodes": top_k_nodes.cpu().numpy(),
                "top_k_scores": top_k_scores.cpu().numpy(),
                "avg_importance": avg_importance.mean().item(),
                "std_importance": avg_importance.std().item(),
                "max_importance": avg_importance.max().item(),
                "min_importance": avg_importance.min().item(),
                "class_all_mean": class_all_importance.mean().item(),
                "class_all_std": class_all_importance.std().item()
            }
    
    return analysis_log


def analyze_importance_consistency(importance_scores_history, labels, top_k=20):
   
    if len(importance_scores_history) < 2:
        return {"status": "insufficient_history"}
    
    unique_labels = torch.unique(labels)
    consistency_analysis = {
        "overall_consistency": {},
        "class_consistency": {}
    }

    recent_scores = importance_scores_history[-1]
    previous_scores = importance_scores_history[-2]

    recent_avg = torch.stack(recent_scores).mean(dim=0)
    previous_avg = torch.stack(previous_scores).mean(dim=0)

    recent_top_k = torch.argsort(recent_avg, descending=True)[:top_k]
    previous_top_k = torch.argsort(previous_avg, descending=True)[:top_k]
    
    overlap = len(set(recent_top_k.cpu().numpy()) & set(previous_top_k.cpu().numpy()))
    overlap_ratio = overlap / top_k

    correlation = torch.corrcoef(torch.stack([recent_avg, previous_avg]))[0, 1].item()
    
    consistency_analysis["overall_consistency"] = {
        "overlap_ratio": overlap_ratio,
        "correlation": correlation,
        "recent_top_k": recent_top_k.cpu().numpy(),
        "previous_top_k": previous_top_k.cpu().numpy()
    }

    for label in unique_labels:
        label_mask = (labels == label)
        label_indices = torch.where(label_mask)[0]
        
        if len(label_indices) > 0:
            recent_class_importance = []
            previous_class_importance = []
            
            for idx in label_indices:
                recent_class_importance.append(recent_scores[idx.item()])
                previous_class_importance.append(previous_scores[idx.item()])
            
            recent_class_avg = torch.stack(recent_class_importance).mean(dim=0)
            previous_class_avg = torch.stack(previous_class_importance).mean(dim=0)

            recent_class_top_k = torch.argsort(recent_class_avg, descending=True)[:top_k]
            previous_class_top_k = torch.argsort(previous_class_avg, descending=True)[:top_k]
            
            class_overlap = len(set(recent_class_top_k.cpu().numpy()) & set(previous_class_top_k.cpu().numpy()))
            class_overlap_ratio = class_overlap / top_k

            class_correlation = torch.corrcoef(torch.stack([recent_class_avg, previous_class_avg]))[0, 1].item()
            
            consistency_analysis["class_consistency"][f"class_{label.item()}"] = {
                "overlap_ratio": class_overlap_ratio,
                "correlation": class_correlation,
                "recent_top_k": recent_class_top_k.cpu().numpy(),
                "previous_top_k": previous_class_top_k.cpu().numpy()
            }
    
    return consistency_analysis
