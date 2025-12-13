import torch
import torch.utils.data
from torch.nn import functional as F
import json

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from scipy.signal import welch
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, filtfilt
from torch_geometric.data import Dataset, Data

import time
import pickle
import os
from sklearn.covariance import GraphicalLassoCV
import numpy as np
import csv
import dgl
from dgl.data.utils import load_graphs
import networkx as nx
from tqdm import tqdm
import random
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from numpy.linalg import inv, pinv
from torch_geometric.data import Batch

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

class CustomGraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['T_indices', 'adj_v_edge_index', 'adj_e_edge_index']:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)
        



class DGLFormDataset(torch.utils.data.Dataset):
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
    

def convert_graph_to_numpy(g, label):
    graph_dict = {
        'node_feat': g.ndata['feat'].numpy(),
        'time_series': g.ndata['time_series'].numpy(),
        'label': label.item() if torch.is_tensor(label) else label,
        'layer_adjs': [adj.astype(np.float32) for adj in g.layer_adjs],
        'layer_features': [adj.astype(np.float32) for adj in g.layer_features],
        'layer_ts': [ts.astype(np.float32) for ts in g.layer_ts]
    }
    
    if hasattr(g, 'edges'):
        src, dst = g.edges()
        graph_dict['edge_index'] = np.stack([src.numpy(), dst.numpy()], axis=0)
    
    return graph_dict

def self_loop(g):
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g
def generate_timestamps(TR=2.0, time_length=197):
    return torch.arange(0, TR*time_length, TR)
name2path = {
    'abide_full_schaefer100': 'multi-freq_NEU/data/abide_schaefer100/abide_full_schaefer100.bin',
    'abide_full_aal116': 'multi-FCN_RGCN_Censnet_multi_freq/data/abide_AAL116/abide_full_aal116.bin',
    'ppmi_schaefer100': 'Contrasformer/data/ppmi_schaefer100/ppmi_schaefer100.bin',
    'ppmi_AAL116': 'multi-FCN_RGCN_Censnet_multi_freq/data/ppmi_AAL116/ppmi_AAL116.bin'
}

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, name, threshold=0.3, edge_ratio=0, node_feat_transform='original', 
                 use_padding=False, max_length=None, split_ratios=(0.7, 0.15, 0.15)):
        self.name = name
        self.use_padding = use_padding
        self.max_length = max_length
        self.split_ratios = split_ratios
        print(f"[!] Initializing Dataset: {self.name}")
        print(f"[!] Padding mode: {'Enabled' if use_padding else 'Disabled'}")
        print(f"[!] Split ratios: Train={split_ratios[0]:.1f}, Val={split_ratios[1]:.1f}, Test={split_ratios[2]:.1f}")

        print(f"--- Pre-processed data not found. Starting data processing... ---")
        t0 = time.time()
        
        G_dataset, Labels = load_graphs(name2path[self.name])
        
        original_labels = Labels['glabel']

        min_length = 210
        
        temp_g_list = []
        labels_list = []
        
        if self.use_padding:
            if self.max_length is None:
                self.max_length = self.determine_max_length(G_dataset, min_length)
                print(f"[!] Auto-determined max length: {self.max_length}")
            else:
                print(f"[!] Using specified max length: {self.max_length}")
        else:
            print(f"[!] Using truncation mode with min_length: {min_length}")
        
        for i in range(len(G_dataset)):
            original_label = original_labels[i].item()

            g = G_dataset[i]
            features = g.ndata['N_features']
            T = features.shape[-1]
            
            if self.use_padding:
                if len(((features != 0).sum(dim=-1) == 0).nonzero()) > 0:
                    continue
            else:
                if T < min_length or len(((features != 0).sum(dim=-1) == 0).nonzero()) > 0:
                    continue

            new_label = -1
            if original_label == 0:
                new_label = -1
            elif original_label == 2:
                new_label = 0
            elif original_label == 3:
                new_label = -1
            elif original_label == 1:
                new_label = 1
            
            if new_label != -1:
                if self.use_padding:
                    g.ndata['time_series'] = self.pad_time_series(features, self.max_length)
                    if node_feat_transform == 'original':
                        g.ndata['feat'] = self.pad_time_series(g.ndata['N_features'], 120)
                    elif node_feat_transform == 'one_hot':
                        g.ndata['feat'] = torch.eye(features.shape[0]).clone()
                else:
                    g.ndata['time_series'] = features[:, :min_length].clone().float()
                    if node_feat_transform == 'original':
                        g.ndata['feat'] = g.ndata['N_features'][:, :120].clone()
                    elif node_feat_transform == 'one_hot':
                        g.ndata['feat'] = torch.eye(features.shape[0]).clone()
                
                temp_g_list.append(g)
                labels_list.append(new_label)

        
        print(f"Filtered out SMC samples and processed remaining data.")
        G_dataset_filtered = temp_g_list

        dataset = self.prepare_raw_data_for_model(G_dataset_filtered, labels_list)

        
        self.all_idx = self.get_all_split_idx(dataset, split_ratios)
        num_folds = len(self.all_idx['train'])

        self.train = [[dataset[idx] for idx in fold_idx] for fold_idx in self.all_idx['train']]
        self.val   = [[dataset[idx] for idx in fold_idx] for fold_idx in self.all_idx['val']]
        self.test  = [[dataset[idx] for idx in fold_idx] for fold_idx in self.all_idx['test']]
        self.train_weights = [self.calculate_class_weights(train_data) for train_data in self.train]

        print("Calculating weights for WeightedRandomSampler...")
        self.train_sampler_weights = self._calculate_sampler_weights(self.train)
        print("Sampler weights calculated.")

        print("Time taken: {:.4f}s".format(time.time()-t0))
        print(f"Number of folds in 'train': {num_folds}")
    
    def pad_time_series(self, features, target_length):
        current_length = features.shape[-1]
        
        if current_length >= target_length:
            return features[:, :target_length].clone().float()
        else:
            num_nodes = features.shape[0]
            padded_features = torch.zeros(num_nodes, target_length, dtype=torch.float32)
            padded_features[:, :current_length] = features.clone().float()
            return padded_features
    
    def determine_max_length(self, G_dataset, min_length):
        max_length = min_length
        for g in G_dataset:
            features = g.ndata['N_features']
            T = features.shape[-1]
            if T >= min_length:
                max_length = max(max_length, T)
        return max_length
    
    def _calculate_sampler_weights(self, train_data_per_fold):
        all_fold_weights = []
        for fold_data in train_data_per_fold:
            labels = np.array([item.y.item() for item in fold_data])
            
            class_counts = np.bincount(labels)
            
            class_weights = 1. / class_counts
            
            sample_weights = np.array([class_weights[label] for label in labels])
            
            all_fold_weights.append(torch.from_numpy(sample_weights).double())
            
        return all_fold_weights

    
    def calculate_class_weights(self, data):
        labels = np.array([item['y'] for item in data])
        if len(labels.shape) > 1:
            labels = labels.reshape(-1)
        
        unique_classes = np.unique(labels)
        
        try:
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=labels
            )
            print(f"Original balanced weights: {weights}")
            return weights
        except Exception as e:
            print("Labels shape:", labels.shape)
            print("Unique classes:", unique_classes)
            print("Labels unique values:", np.unique(labels))
            raise e


    
    def print_label_counts(self, labels_tensor):
        label_counts = torch.bincount(labels_tensor)
        for label, count in enumerate(label_counts):
            print(f"Label {label}: {count.item()} samples")

   

    def get_all_split_idx(self, dataset, split_ratios=(0.7, 0.15, 0.15)):
        train_ratio, val_ratio, test_ratio = split_ratios
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        ratio_str = f"{train_ratio:.2f}_{val_ratio:.2f}_{test_ratio:.2f}"
        root_idx_dir = 'multi-FCN_RGCN_Censnet_multi_freq/data/ppmi_AAL116/split/split_0.70_0.15_0.15_ppmi_210_42_2class'
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}
        
        if not os.path.exists(os.path.join(root_idx_dir, 'train.index')):
            print(f"[!] Splitting the data into train/val/test with ratios {split_ratios}...")
            k_splits = 5
            labels = np.array([d['y'] for d in dataset])
            indices = np.arange(len(dataset))
            
            train_idx_list = []
            val_idx_list = []
            test_idx_list = []
            
            for fold in range(k_splits):
                fold_random_state = 42 + fold
                
                train_val_idx, test_idx = train_test_split(
                    indices,
                    test_size=test_ratio,
                    stratify=labels,
                    random_state=fold_random_state
                )
                
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_ratio/(train_ratio + val_ratio),
                    stratify=labels[train_val_idx],
                    random_state=fold_random_state
                )
                
                train_idx_list.append(list(train_idx))
                val_idx_list.append(list(val_idx))
                test_idx_list.append(list(test_idx))
            
            with open(os.path.join(root_idx_dir, 'train.index'), 'w', newline='') as f_train:
                writer = csv.writer(f_train)
                for fold in train_idx_list:
                    writer.writerow(fold)
            with open(os.path.join(root_idx_dir, 'val.index'), 'w', newline='') as f_val:
                writer = csv.writer(f_val)
                for fold in val_idx_list:
                    writer.writerow(fold)
            with open(os.path.join(root_idx_dir, 'test.index'), 'w', newline='') as f_test:
                writer = csv.writer(f_test)
                for fold in test_idx_list:
                    writer.writerow(fold)
            print("[!] Splitting done!")
            
            self._verify_split_ratios(train_idx_list, val_idx_list, test_idx_list, len(dataset), split_ratios)
            
         
        
        for section in ['train', 'val', 'test']:
            with open(os.path.join(root_idx_dir, section + '.index'), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, row)) for row in reader]
        
        return all_idx
    
    def _verify_split_ratios(self, train_idx_list, val_idx_list, test_idx_list, total_samples, split_ratios):
        train_ratio, val_ratio, test_ratio = split_ratios
        
        train_size = len(train_idx_list[0])
        val_size = len(val_idx_list[0])
        test_size = len(test_idx_list[0])
        fold_total = train_size + val_size + test_size
        
        actual_train_ratio = train_size / fold_total
        actual_val_ratio = val_size / fold_total
        actual_test_ratio = test_size / fold_total
        
        print(f"[!] Actual split ratios in first fold:")
        print(f"    Train: {actual_train_ratio:.3f} (expected: {train_ratio:.3f})")
        print(f"    Val:   {actual_val_ratio:.3f} (expected: {val_ratio:.3f})")
        print(f"    Test:  {actual_test_ratio:.3f} (expected: {test_ratio:.3f})")
        
        tolerance = 0.01
        if (abs(actual_train_ratio - train_ratio) > tolerance or 
            abs(actual_val_ratio - val_ratio) > tolerance or 
            abs(actual_test_ratio - test_ratio) > tolerance):
            print(f"[!] Warning: Split ratios deviate from expected values by more than {tolerance*100}%")
    

    def prepare_raw_data_for_model(self, G_dataset, labels_list):
        dataset = []
        print("Preparing raw time series data for dynamic graph construction in model.")
        for i, g in tqdm(enumerate(G_dataset), total=len(G_dataset), desc="Preparing subjects data"):
            ts_tensor = g.ndata['time_series']
            num_nodes = ts_tensor.shape[0]

            data_obj = CustomGraphData(
                raw_ts=ts_tensor,       
                y=torch.tensor([labels_list[i]]),
                num_nodes=num_nodes,
            )
            dataset.append(data_obj)
        return dataset