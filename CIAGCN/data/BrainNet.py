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
    'abide_full_aal116': 'data/abide_full_aal116.bin',
    'adni_schaefer100': 'multi-freq_NEU/data/andi_schaefer100/adni_schaefer100.bin',
    'adni_AAL116': 'multi-FCN_RGCN_Censnet_multi_freq/data/adni_AAL116/adni_AAL116.bin',
    'adni_AAL116_partial': 'multi-freq_GCN/data/adni_AAL116/partial/adni_AAL116.bin'
}

     
class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, name, threshold=0.3, edge_ratio=0, node_feat_transform='original'):
        self.name = name
        print(f"[!] Initializing Dataset: {self.name}")

      
        save_dir = 'multi-FCN_RGCN_Censnet_multi_freq/data/adni_AAL116/raw_ts_only_3class' 
        processed_data_path = os.path.join(save_dir, 'adni_raw_ts_3class_filtered.pt') 
       

        if os.path.exists(processed_data_path):
            print(f"--- Loading pre-processed data from: {processed_data_path} ---")
            t0 = time.time()
            dataset = torch.load(processed_data_path)
            labels_list = [data.y.item() for data in dataset]
            print(f"--- Data loaded successfully in {time.time() - t0:.2f}s ---")

        else:
            print(f"--- Pre-processed data not found. Starting data processing... ---")
            t0 = time.time()
            
            G_dataset, Labels = load_graphs(name2path[self.name])
            
            original_labels = Labels['glabel']

            temp_g_list = []
            labels_list = []
            min_length = 197
            
            for i in range(len(G_dataset)):
                original_label = original_labels[i].item()

                if original_label == 2:  
                    continue

                g = G_dataset[i]
                features = g.ndata['N_features']
                T = features.shape[-1]

                if T < min_length or len(((features != 0).sum(dim=-1) == 0).nonzero()) > 0:
                    continue

                new_label = -1 
                if original_label == 0:   
                    new_label = 0
                elif original_label == 3 or original_label == 4: 
                    new_label = 1
                elif original_label == 1: 
                    new_label = 2
                if new_label != -1:
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

            print(f"--- Saving processed 3-class data to: {processed_data_path} ---")
            os.makedirs(save_dir, exist_ok=True) 
            torch.save(dataset, processed_data_path)
            print(f"--- Initial processing and saving finished in {time.time() - t0:.2f}s ---")


        
        
        self.all_idx = self.get_all_split_idx(dataset)
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

    def _calculate_sampler_weights(self, train_data_per_fold):

        all_fold_weights = []
        for fold_data in train_data_per_fold:
            labels = np.array([item.y.item() for item in fold_data])
            
            class_counts = np.bincount(labels)
            

            class_weights = 1. / np.sqrt(class_counts)
            

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

            power = 0.5
            smoothed_weights_power = weights ** power
            print(f"Smoothed weights (power={power}): {smoothed_weights_power}")
            smoothed_weights_log = np.log1p(weights)
            smoothed_weights_log = np.where(smoothed_weights_log < 1.0, 1.0, smoothed_weights_log)
            print(f"Smoothed weights (log1p): {smoothed_weights_log}")

            return smoothed_weights_log
        except Exception as e:
            print("Labels shape:", labels.shape)
            print("Unique classes:", unique_classes)
            print("Labels unique values:", np.unique(labels))
            raise e


    
    def print_label_counts(self, labels_tensor):
        label_counts = torch.bincount(labels_tensor) 
        for label, count in enumerate(label_counts):
            print(f"Label {label}: {count.item()} samples")

   

    def get_all_split_idx(self, dataset):
        
        root_idx_dir = 'multi-FCN_RGCN_Censnet_multi_freq/data/adni_AAL116/split/split_3class'
        
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        if not os.path.exists(os.path.join(root_idx_dir, 'train.index')):
            print("[!] Splitting the data into train/val/test ...")
            k_splits = 5

            labels = np.array([d['y'] for d in dataset])
            indices = np.arange(len(dataset))
            skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)
            
            train_idx_list = []
            val_idx_list = []
            test_idx_list = []
            

            for train_val_idx, test_idx in skf.split(indices, labels):
   
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=0.111, 
                    stratify=labels[train_val_idx],
                    random_state=42
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

        for section in ['train', 'val', 'test']:
            with open(os.path.join(root_idx_dir, section + '.index'), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, row)) for row in reader]
        
        return all_idx

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