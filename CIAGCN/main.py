import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
import argparse
import glob
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from nets.gradient_importance import GradientBasedCausalTSLANet
from data.data import LoadData
from train_TUs_graph_classification import evaluate_network_all_metric
import torch.distributed as dist
from collections import defaultdict
import random
import gc
from torch.cuda.amp import GradScaler, autocast


def load_pretrained_weights(model, pretrained_path, device, freeze_backbone=False, target_classes=None):
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weight file not found: {pretrained_path}")
    
    try:
        pretrained_state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
    except Exception as e:
        raise e
    
    if hasattr(model, 'base_model'):
        current_state_dict = model.base_model.state_dict()
    else:
        current_state_dict = model.state_dict()
    
            
    loaded_info = {
        'total_pretrained_params': len(pretrained_state_dict),
        'total_current_params': len(current_state_dict),
        'loaded_params': 0,
        'skipped_params': 0,
        'reinitialized_params': 0,
        'frozen_params': 0,
        'trainable_params': 0,
        'loaded_layers': [],
        'skipped_layers': [],
        'reinitialized_layers': []
    }
    
                 
    if target_classes is not None:
                         
        pretrained_classifier_keys = [k for k in pretrained_state_dict.keys() if 'classifier' in k or 'fc' in k or 'head' in k]
        if pretrained_classifier_keys:
                            
            for key in pretrained_classifier_keys:
                if pretrained_state_dict[key].shape[-1] != target_classes:
                    print(f"Class count mismatch for {key}: pretrained={pretrained_state_dict[key].shape[-1]}, target={target_classes}")
                                      
                    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k not in pretrained_classifier_keys}
                    loaded_info['reinitialized_layers'] = pretrained_classifier_keys
                    loaded_info['reinitialized_params'] = len(pretrained_classifier_keys)
                    break
    
             
    for name, param in pretrained_state_dict.items():
        if name in current_state_dict:
            if current_state_dict[name].shape == param.shape:
                current_state_dict[name] = param
                loaded_info['loaded_params'] += 1
                loaded_info['loaded_layers'].append(name)
            else:
                loaded_info['skipped_params'] += 1
                loaded_info['skipped_layers'].append(name)
        else:
            loaded_info['skipped_params'] += 1
            loaded_info['skipped_layers'].append(name)
    
    if hasattr(model, 'base_model'):
        model.base_model.load_state_dict(current_state_dict)
    else:
        model.load_state_dict(current_state_dict)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not any(keyword in name.lower() for keyword in ['classifier', 'fc', 'head', 'output']):
                param.requires_grad = False
                loaded_info['frozen_params'] += 1
            else:
                param.requires_grad = True
                loaded_info['trainable_params'] += 1
    else:
        for param in model.parameters():
            param.requires_grad = True
            loaded_info['trainable_params'] += 1
    
    return model, loaded_info


def create_optimizer_for_finetuning(model, lr, weight_decay, freeze_backbone=False):
    
    if freeze_backbone:
                         
        classifier_params = []
        backbone_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(keyword in name.lower() for keyword in ['classifier', 'fc', 'head', 'output']):
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
                                            
        param_groups = [
            {'params': classifier_params, 'lr': lr, 'weight_decay': weight_decay, 'name': 'classifier'},
        ]
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params, 
                'lr': lr * 0.1,                   
                'weight_decay': weight_decay, 
                'name': 'backbone'
            })
        
        optimizer = optim.Adam(param_groups)
    else:
        param_groups = [{'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
        optimizer = optim.Adam(param_groups)
    
    return optimizer, param_groups

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def train_epoch_gradient_causal(model, optimizer, device, train_loader, epoch, weights_fold, tau=None, scaler=None, use_amp=True, importance_log_file=None):
    
    model.train()
    total_loss = 0
    total_orig_loss = 0
    total_intra_loss = 0
    total_inter_loss = 0
    correct = 0
    total = 0
    
                
    importance_history = []
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
                          
        actual_model = model.module if hasattr(model, 'module') else model
        
                  
        if use_amp and scaler is not None:
            with autocast():
                                 
                results = actual_model.causal_training_step(data, tau, batch_idx)
                
                loss = results['loss']
                loss_orig = results['loss_orig']
                loss_intra = results['loss_intra']
                loss_inter = results['loss_inter']
                importance_scores = results['importance_scores']
                logits = results['logits']                   
            
                            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
                             
            results = actual_model.causal_training_step(data, tau, batch_idx)
            
            loss = results['loss']
            loss_orig = results['loss_orig']
            loss_intra = results['loss_intra']
            loss_inter = results['loss_inter']
            importance_scores = results['importance_scores']
            logits = results['logits']                   
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_orig_loss += loss_orig.item()
        total_intra_loss += loss_intra.item()
        total_inter_loss += loss_inter.item()
        
                                      
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
        
    
    avg_loss = total_loss / len(train_loader)
    avg_orig_loss = total_orig_loss / len(train_loader)
    avg_intra_loss = total_intra_loss / len(train_loader)
    avg_inter_loss = total_inter_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    
    return avg_loss, accuracy, avg_orig_loss, avg_intra_loss, avg_inter_loss, optimizer


def train_val_pipeline_gradient_causal(MODEL_NAME, DATASET_NAME, params, net_params, dirs, args):
    
    avg_test_acc, avg_test_precision, avg_test_recall, avg_test_f1, avg_test_roc_auc, avg_test_conf_matrix = [], [], [], [], [], []
    avg_test_precision_macro, avg_test_recall_macro, avg_test_f1_macro, avg_test_report = [], [], [], []
    avg_convergence_epochs = []
    
                   
    val_acc_best_results_all_folds = []                                       
    last_epoch_results_all_folds = []                         
    
    t0 = time.time()
    per_epoch_time = []
    device = net_params['device']
    
                             
    if dirs is None:
        root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_pircture_file = None, None, None, None, None
    else:
        root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_pircture_file = dirs
    
            
    os.environ['PYTHONHASHSEED'] = str(params['seed'])
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
          
                                   
    multi_site_mode = getattr(args, 'multi_site_mode', False)
    selected_sites = getattr(args, 'selected_sites', None)
    
    dataset = LoadData(
        DATASET_NAME, 
        threshold=params['threshold'], 
        node_feat_transform=params['node_feat_transform'],
        multi_site_mode=multi_site_mode,
        selected_sites=selected_sites
    )
    weights = dataset.train_weights
    weights_for_sampler = dataset.train_sampler_weights
    
                  
    num_folds = 10 if multi_site_mode else 5
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_pircture_file = dirs
    
    
                
    pretrained_loaded_info = None
    
    try:
        for split_number in range(num_folds):
            
                          
            if 'model' in locals():
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
                         
            best_val_acc_epoch = -1
            best_val_acc = 0.0
            last_epoch_num = -1
            
            weights_fold = weights[split_number]
            weights_fold_sampler = weights_for_sampler[split_number]
            
                            
            if is_main_process() and write_pircture_file is not None:
                pic_dir = os.path.join(write_pircture_file, "RUN_" + str(split_number))
                if not os.path.exists(pic_dir):
                    os.makedirs(pic_dir)
            else:
                pic_dir = None
            
            t0_split = time.time()
            if is_main_process() and root_log_dir is not None:
                log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
                writer = SummaryWriter(log_dir=log_dir)
            else:
                log_dir = None
                writer = None
            
                    
            torch.manual_seed(params['seed'])
            torch.cuda.manual_seed_all(params['seed'])
            torch.use_deterministic_algorithms(True)
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])
                torch.cuda.manual_seed_all(params['seed'])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            
                  
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
            
                   
            try:
                if isinstance(weights_fold, (list, tuple)):
                    num_classes_split = len(weights_fold)
                elif hasattr(weights_fold, 'shape'):
                    num_classes_split = int(weights_fold.shape[0])
                else:
                    num_classes_split = net_params.get('n_classes', 2)
            except Exception:
                num_classes_split = net_params.get('n_classes', 2)
            net_params['n_classes'] = int(num_classes_split)
            
            
                          
            model = GradientBasedCausalTSLANet(
                input_length=params.get('seq_len', 120),
                num_nodes=net_params.get('node_num', 116),
                nhid=net_params.get('hidden_dim', 128),
                nclass=net_params['n_classes'],
                dropout=net_params.get('dropout', 0.3),
                node_feature_dim=net_params.get('hidden_dim', 128),
                num_layers=net_params.get('num_layers', 3),
                leaky_slope=net_params.get('leaky_slope', 0.1),
                alpha=args.alpha,
                beta=args.beta,
                top_k_important=args.top_k_important,
                top_k_background=getattr(args, 'top_k_background', 40),
                gradient_method=getattr(args, 'gradient_method', 'abs'),
                use_kl_consistency=getattr(args, 'use_kl_consistency', True),
                kl_temperature=getattr(args, 'kl_temperature', 1.5),
                kl_gamma=getattr(args, 'kl_gamma', 0.1),
                topk_strategy=getattr(args, 'topk_strategy', 'local'),
                global_topk_ratio=getattr(args, 'global_topk_ratio', 0.3)
            )
            
                                        
            pretrained_path = getattr(args, 'pretrained_path', None)
            if pretrained_path and pretrained_path != 'None' and pretrained_path.strip():
                if split_number == 0:
                    try:
                        model, pretrained_loaded_info = load_pretrained_weights(
                            model, 
                            pretrained_path, 
                            device, 
                            freeze_backbone=getattr(args, 'freeze_backbone', False),
                            target_classes=net_params['n_classes']
                        )
                    except Exception as e:
                        pretrained_loaded_info = None
                elif split_number > 0:
                    if pretrained_loaded_info is not None:
                        pass
            else:
                pretrained_loaded_info = None
            
                                     
            model.mixing_strategy = getattr(args, 'mixing_strategy', 'anatomical')
            model.enable_causal_enhancement = getattr(args, 'enable_causal_enhancement', True)
            model.causal_enhancement_frequency = getattr(args, 'causal_enhancement_frequency', 1)
            
                                                                      
            model.memory_bank_enabled = getattr(args, 'memory_bank_enabled', True)
            model.memory_bank_size = getattr(args, 'memory_bank_size', 3)
            model.memory_bank_update_freq = getattr(args, 'memory_bank_update_freq', 5)
            
            
            model = model.to(device)
            
                   
            if is_dist_avail_and_initialized() and device.type == 'cuda':
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False
                )
            
                     
            if pretrained_loaded_info is not None:
                                     
                freeze_backbone = getattr(args, 'freeze_backbone', False)
                          
                pretrained_lr = params.get('pretrained_lr', params['init_lr'])
                optimizer, param_groups = create_optimizer_for_finetuning(
                    model, pretrained_lr, params['weight_decay'], freeze_backbone
                )
            else:
                              
                optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
                param_groups = [{'params': model.parameters(), 'lr': params['init_lr'], 'weight_decay': params['weight_decay']}]
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=params['lr_reduce_factor'], patience=params['lr_schedule_patience'])
            
                      
            use_amp = getattr(args, 'use_amp', True)
            scaler = GradScaler() if use_amp else None
            
            
                   
            drop_last = True if MODEL_NAME in ['DiffPool', 'ContrastPool', 'Transformer', 'GPS', 'Contrasformer'] else False
            
                                                                                     
                          
            use_weighted_sampler = getattr(args, 'use_weighted_sampler', True)        
            
            if is_dist_avail_and_initialized():
                world_size = dist.get_world_size()
                rank = get_rank()
                train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last)
            elif use_weighted_sampler and weights_fold_sampler is not None:
                                      
                train_sampler = WeightedRandomSampler(
                    weights=weights_fold_sampler,
                    num_samples=len(trainset),
                    replacement=True                       
                )
                if is_main_process():
                    print(f"\n=== Using WeightedRandomSampler for class imbalance ===")
                    print(f"Training samples: {len(trainset)}")
                    print(f"Weight range: {weights_fold_sampler.min():.4f} - {weights_fold_sampler.max():.4f}")
                              
                    labels = [data.y.item() for data in trainset]
                    for label in sorted(set(labels)):
                        label_indices = [i for i, l in enumerate(labels) if l == label]
                        avg_weight = weights_fold_sampler[label_indices].mean()
                        print(f"  Class {label}: {len(label_indices)} samples, avg weight: {avg_weight:.4f}")
                    print("=" * 50)
            else:
                train_sampler = None
            
            from torch_geometric.loader import DataLoader as PyGDataLoader
            train_loader = PyGDataLoader(
                trainset,
                batch_size=params['batch_size'],
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                drop_last=drop_last,
                pin_memory=True,                     
                num_workers=6,                   
                persistent_workers=True
            )
            
                                                                      
                                                      
            if is_main_process() and getattr(args, 'memory_bank_enabled', True):
                actual_model = model.module if hasattr(model, 'module') else model
                tau_init = 1.0
                actual_model.initialize_memory_bank_from_dataset(
                    train_loader=train_loader,
                    device=device,
                    tau=tau_init
                )
            
                              
            if is_main_process():
                val_loader = PyGDataLoader(
                    valset,
                    batch_size=params['batch_size'],
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=8,
                    persistent_workers=True
                )
                test_loader = PyGDataLoader(
                    testset,
                    batch_size=params['batch_size'],
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    num_workers=8,
                    persistent_workers=True
                )
            else:
                val_loader, test_loader = None, None
            
                  
            tau_init = 1.0
            tau_final = 0.1
            anneal_rate = 0.95
            
            for epoch in range(params['epochs']):
                tau = max(tau_final, tau_init * (anneal_rate ** epoch))
                start = time.time()
                
                if is_dist_avail_and_initialized() and isinstance(train_loader.sampler, DistributedSampler):
                    train_loader.sampler.set_epoch(epoch)
                
                    
                epoch_train_loss, epoch_train_acc, epoch_train_orig_loss, epoch_train_intra_loss, epoch_train_inter_loss, optimizer = train_epoch_gradient_causal(
                    model, optimizer, device, train_loader, epoch, weights_fold, tau=tau, scaler=scaler, use_amp=use_amp, importance_log_file=None)
                
                                   
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                             
                if is_main_process():
                                      
                    eval_model = model.module.base_model if hasattr(model, 'module') else model.base_model
                    
                                
                    if use_amp:
                        with autocast():
                            epoch_val_loss, val_precision_macro, val_recall_macro, val_f1_macro, val_report, _, val_acc, val_balanced_acc, val_precision, val_recall, val_f1, val_roc_auc, conf_matrix = evaluate_network_all_metric(
                                eval_model, device, val_loader, epoch, False, metric='val', fold=split_number)
                            
                            epoch_test_loss, test_precision_macro, test_recall_macro, test_f1_macro, test_report, _, test_acc, test_balanced_acc, test_precision, test_recall, test_f1, test_roc_auc, conf_matrix = evaluate_network_all_metric(
                                eval_model, device, test_loader, epoch, False, metric='test', fold=split_number)
                    else:
                        epoch_val_loss, val_precision_macro, val_recall_macro, val_f1_macro, val_report, _, val_acc, val_balanced_acc, val_precision, val_recall, val_f1, val_roc_auc, conf_matrix = evaluate_network_all_metric(
                            eval_model, device, val_loader, epoch, False, metric='val', fold=split_number)
                        
                        epoch_test_loss, test_precision_macro, test_recall_macro, test_f1_macro, test_report, _, test_acc, test_balanced_acc, test_precision, test_recall, test_f1, test_roc_auc, conf_matrix = evaluate_network_all_metric(
                            eval_model, device, test_loader, epoch, False, metric='test', fold=split_number)
                    
                    val_acc_tensor = torch.tensor([val_acc], dtype=torch.float32, device=device)
                    test_acc_tensor = torch.tensor([test_acc], dtype=torch.float32, device=device)
                else:
                    val_acc_tensor = torch.zeros(1, dtype=torch.float32, device=device)
                    test_acc_tensor = torch.zeros(1, dtype=torch.float32, device=device)
                
                if is_dist_avail_and_initialized():
                    dist.broadcast(val_acc_tensor, src=0)
                    dist.broadcast(test_acc_tensor, src=0)
                    dist.barrier()
                val_acc_synced = float(val_acc_tensor.item())
                test_acc_synced = float(test_acc_tensor.item())
                
                      
                if is_main_process():
                    if writer is not None:
                        writer.add_scalar('train/loss', epoch_train_loss, epoch)
                        writer.add_scalar('train/loss_orig', epoch_train_orig_loss, epoch)
                        writer.add_scalar('train/loss_intra', epoch_train_intra_loss, epoch)
                        writer.add_scalar('val/loss', epoch_val_loss, epoch)
                        writer.add_scalar('train/acc', epoch_train_acc, epoch)
                        writer.add_scalar('val/acc', val_acc_synced, epoch)
                        writer.add_scalar('test/acc', test_acc_synced, epoch)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                        
                                    
                        actual_model = model.module if hasattr(model, 'module') else model
                        if getattr(actual_model, 'use_kl_consistency', True):
                            writer.add_scalar('train/loss_inter_kl', epoch_train_inter_loss, epoch)
                        else:
                            writer.add_scalar('train/loss_inter_ce', epoch_train_inter_loss, epoch)
                
                        
                if is_main_process() and root_ckpt_dir is not None:
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                else:
                    ckpt_dir = None
                
                                               
                if is_main_process() and val_acc_synced > best_val_acc and ckpt_dir is not None:
                    best_val_acc = val_acc_synced
                    best_val_acc_epoch = epoch
                    
                    old_val_acc_best_list = glob.glob(os.path.join(ckpt_dir, "best_val_acc_model_*.pt"))
                    for old_best in old_val_acc_best_list:
                        os.remove(old_best)
                    
                    best_val_acc_model_path = os.path.join(ckpt_dir, f"best_val_acc_model_epoch_{epoch}.pt")
                    to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(to_save, best_val_acc_model_path)
                
                                 
                if is_main_process():
                    last_epoch_num = epoch
                
                per_epoch_time.append(time.time() - start)
                                
                if is_main_process():
                    scheduler.step(val_acc_synced)
                
                                
                if is_main_process() and optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    if is_dist_avail_and_initialized():
                        stop_tensor = torch.tensor([1], dtype=torch.int32, device=device)
                        dist.broadcast(stop_tensor, src=0)
                    break
                elif not is_main_process() and is_dist_avail_and_initialized():
                    stop_tensor = torch.tensor([0], dtype=torch.int32, device=device)
                    dist.broadcast(stop_tensor, src=0)
                    if stop_tensor.item() == 1:
                        break
                
                                  
                if is_main_process() and time.time() - t0_split > params['max_time'] * 3600 / 10:
                    print('-' * 89)
                    print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                    if is_dist_avail_and_initialized():
                        stop_tensor = torch.tensor([1], dtype=torch.int32, device=device)
                        dist.broadcast(stop_tensor, src=0)
                    break
                elif not is_main_process() and is_dist_avail_and_initialized():
                    stop_tensor = torch.tensor([0], dtype=torch.int32, device=device)
                    dist.broadcast(stop_tensor, src=0)
                    if stop_tensor.item() == 1:
                        break
            
                       
            if is_main_process() and ckpt_dir is not None:
                target_model = model.module if hasattr(model, 'module') else model
                eval_model = model.module.base_model if hasattr(model, 'module') else model.base_model
                
                               
                val_acc_best_results = {}
                last_epoch_results = {}
                
                
                                               
                if best_val_acc_epoch >= 0:
                    best_val_acc_model_path = os.path.join(ckpt_dir, f"best_val_acc_model_epoch_{best_val_acc_epoch}.pt")
                    if os.path.exists(best_val_acc_model_path):
                        target_model.load_state_dict(torch.load(best_val_acc_model_path, map_location=device, weights_only=True))
                        
                                    
                        if use_amp:
                            with autocast():
                                _, val_acc_test_precision_macro, val_acc_test_recall_macro, val_acc_test_f1_macro, val_acc_test_report, _, val_acc_test_acc, val_acc_test_balanced_acc, val_acc_test_precision, val_acc_test_recall, val_acc_test_f1, val_acc_test_roc_auc, val_acc_conf_matrix = evaluate_network_all_metric(
                                    eval_model, device, test_loader, best_val_acc_epoch, True, 'val_acc', pic_dir)
                        else:
                            _, val_acc_test_precision_macro, val_acc_test_recall_macro, val_acc_test_f1_macro, val_acc_test_report, _, val_acc_test_acc, val_acc_test_balanced_acc, val_acc_test_precision, val_acc_test_recall, val_acc_test_f1, val_acc_test_roc_auc, val_acc_conf_matrix = evaluate_network_all_metric(
                                eval_model, device, test_loader, best_val_acc_epoch, True, 'val_acc', pic_dir)
                        
                        val_acc_best_results = {
                            'acc': val_acc_test_acc,
                            'precision': val_acc_test_precision,
                            'recall': val_acc_test_recall,
                            'f1': val_acc_test_f1,
                            'roc_auc': val_acc_test_roc_auc,
                            'conf_matrix': val_acc_conf_matrix,
                            'precision_macro': val_acc_test_precision_macro,
                            'recall_macro': val_acc_test_recall_macro,
                            'f1_macro': val_acc_test_f1_macro,
                            'report': val_acc_test_report,
                            'epoch': best_val_acc_epoch,
                            'balanced_acc': val_acc_test_balanced_acc
                        }
                
                if last_epoch_num >= 0:
                                    
                    last_epoch_model_path = os.path.join(ckpt_dir, f"last_epoch_model_epoch_{last_epoch_num}.pt")
                    to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(to_save, last_epoch_model_path)
                    
                                       
                    target_model.load_state_dict(torch.load(last_epoch_model_path, map_location=device, weights_only=True))
                    
                                
                    if use_amp:
                        with autocast():
                            _, last_test_precision_macro, last_test_recall_macro, last_test_f1_macro, last_test_report, _, last_test_acc, last_test_balanced_acc, last_test_precision, last_test_recall, last_test_f1, last_test_roc_auc, last_conf_matrix = evaluate_network_all_metric(
                                eval_model, device, test_loader, last_epoch_num, True, 'last', pic_dir)
                    else:
                        _, last_test_precision_macro, last_test_recall_macro, last_test_f1_macro, last_test_report, _, last_test_acc, last_test_balanced_acc, last_test_precision, last_test_recall, last_test_f1, last_test_roc_auc, last_conf_matrix = evaluate_network_all_metric(
                            eval_model, device, test_loader, last_epoch_num, True, 'last', pic_dir)
                    
                    last_epoch_results = {
                        'acc': last_test_acc,
                        'precision': last_test_precision,
                        'recall': last_test_recall,
                        'f1': last_test_f1,
                        'roc_auc': last_test_roc_auc,
                        'conf_matrix': last_conf_matrix,
                        'precision_macro': last_test_precision_macro,
                        'recall_macro': last_test_recall_macro,
                        'f1_macro': last_test_f1_macro,
                        'report': last_test_report,
                        'epoch': last_epoch_num,
                        'balanced_acc': last_test_balanced_acc
                    }
                
                if val_acc_best_results:
                    val_acc_best_results_all_folds.append(val_acc_best_results)
                if last_epoch_results:
                    last_epoch_results_all_folds.append(last_epoch_results)
                
                                                          
                if val_acc_best_results:
                    avg_test_acc.append(val_acc_best_results['acc'])
                    avg_test_precision.append(val_acc_best_results['precision'])
                    avg_test_recall.append(val_acc_best_results['recall'])
                    avg_test_f1.append(val_acc_best_results['f1'])
                    avg_test_roc_auc.append(val_acc_best_results['roc_auc'])
                    avg_test_conf_matrix.append(val_acc_best_results['conf_matrix'])
                    avg_test_precision_macro.append(val_acc_best_results['precision_macro'])
                    avg_test_recall_macro.append(val_acc_best_results['recall_macro'])
                    avg_test_f1_macro.append(val_acc_best_results['f1_macro'])
                
            
            avg_convergence_epochs.append(epoch)
            
                                                                         
            if writer is not None:
                writer.close()
            
                          
            if is_dist_avail_and_initialized():
                dist.barrier()
            
            del optimizer
            del scheduler
            del train_loader
            if is_main_process():
                del val_loader
                del test_loader
            
            torch.cuda.empty_cache()
            gc.collect()
            
    
    except KeyboardInterrupt:
        if is_main_process():
            print('-' * 89)
            print('Exiting from training early because of KeyboardInterrupt')
        if is_dist_avail_and_initialized():
            dist.barrier()
    
                       
    if is_main_process() and write_file_name is not None:
        pass
    
                  
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
        
    parser = argparse.ArgumentParser()
                                                                                                                    
                                                                                                                                                                           
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details", 
                       default='multi-FCN_RGCN_Censnet_multi_freq/configs/adni_schaefer116/TUs_graph_classification_CausalTSLANet_adni_aal116_100k.json')
                                                                                                                                                                     
    parser.add_argument('--gpu_id', help="Please give a value for gpu id", default='7')
    parser.add_argument('--gpu', type=int, nargs='+', default=[7], help='GPU list for cpu')
    parser.add_argument('--cpus_per_gpu', type=int, default=10, help='Number of CPUs per GPU')
    parser.add_argument('--num_workers', type=int, default=8, help='number_worker')
    parser.add_argument('--model', help="Please give a value for model name", default='GradientBasedCausalTSLANet')
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size", default=128)
    parser.add_argument('--init_lr', type=float, help="Please give a value for init_lr", default=0.0001)
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay", type=float, default=0.0)
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--threshold', type=float, help="Please give a threshold to drop edge", default=0.3)
    parser.add_argument('--edge_ratio', type=float, help="Please give a ratio to drop edge", default=0)
    parser.add_argument('--node_feat_transform', help="Please give a value for node feature transform", default='pearson')
    parser.add_argument('--pos_enc', type=str, default='identity', help="Please give a value for positional encoding")
    parser.add_argument('--contrast', default=True, action='store_true')
    parser.add_argument('--multi_site_mode', action='store_true', default=False, help='Enable multi-site leave-one-site-out evaluation')
    parser.add_argument('--no_multi_site_mode', action='store_false', dest='multi_site_mode', help='Disable multi-site mode')
    parser.add_argument('--selected_sites', type=str, nargs='+', default=None, help='Explicit site list when multi-site mode is enabled')
    parser.add_argument('--auto_select_sites', action='store_true', default=True, help='Automatically select sites that cover all classes')
    parser.add_argument('--manual_select_sites', action='store_false', dest='auto_select_sites', help='Disable auto site selection')
    
            
    parser.add_argument('--lambda1', type=float, help='Weight for causal mixing loss')
    parser.add_argument('--lambda2', type=float, help='Weight for background mixing loss (CE loss)')
    parser.add_argument('--top_k_important', type=int, help='Number of important nodes for causal mixing')
    parser.add_argument('--top_k_background', type=int, help='Number of background nodes for background mixing')
    
                 
    parser.add_argument('--use_kl_consistency', action='store_true', default=False, help='Use KL divergence consistency loss for background mixing')
    parser.add_argument('--disable_kl_consistency', action='store_false', dest='use_kl_consistency', help='Disable KL divergence consistency loss')
    parser.add_argument('--kl_temperature', type=float, help='Temperature parameter T for KL divergence (1.0-2.0)')
    parser.add_argument('--kl_gamma', type=float, help='Weight gamma for KL divergence loss (0.05-0.2)')
            
    parser.add_argument('--gradient_method', type=str, default='abs', 
                       choices=['abs', 'square', 'raw'], 
                       help='Gradient processing method: abs, square, or raw')
          
    parser.add_argument('--mixing_strategy', type=str, default='union', 
                       choices=['anatomical', 'element_wise', 'hybrid', 'union'], 
                       help='Mixing strategy: anatomical (preserve brain structure), element_wise (original), hybrid, or union (aggressive intervention)')
      
    parser.add_argument('--use_amp', action='store_true', default=False, help='Enable Automatic Mixed Precision training')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='Disable Automatic Mixed Precision training')
            
    parser.add_argument('--enable_causal_enhancement', action='store_true', default=True, help='Enable causal enhancement training')
    parser.add_argument('--disable_causal_enhancement', action='store_false', dest='enable_causal_enhancement', help='Disable causal enhancement training to save memory')
    parser.add_argument('--causal_enhancement_frequency', type=int, default=1, help='Frequency of causal enhancement (every N batches)')
    
                   
    parser.add_argument('--memory_bank_enabled', action='store_true', default=True, help='Enable Memory Bank for causal mixing')
    parser.add_argument('--disable_memory_bank', action='store_false', dest='memory_bank_enabled', help='Disable Memory Bank')
    parser.add_argument('--memory_bank_size', type=int, help='Number of samples per class in Memory Bank')
    parser.add_argument('--memory_bank_update_freq', type=int, help='Update Memory Bank every N batches')
    
               
    parser.add_argument('--use_weighted_sampler', action='store_true', default=True, help='Enable WeightedRandomSampler for imbalanced datasets')
    parser.add_argument('--disable_weighted_sampler', action='store_false', dest='use_weighted_sampler', help='Disable WeightedRandomSampler')
    
                  
    parser.add_argument('--topk_strategy', type=str, default='global', 
                       choices=['local', 'global'], 
                       help='Top-K edge selection strategy: local (per-node top-k) or global (graph-level top-k)')
    parser.add_argument('--global_topk_ratio', type=float, default=0.3, 
                       help='Ratio of edges to keep for global top-k strategy (e.g., 0.3 means keep top 30% edges)')
    
                
    parser.add_argument('--pretrained_path', type=str, default='best_model_epoch_97.pt', 
                       help='Pretrained weight file path')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, 
                       help='Whether to freeze backbone and train classifier head only')
    parser.add_argument('--finetune_all', action='store_true', default=True, 
                       help='Whether to finetune the entire model')
    parser.add_argument('--pretrained_lr', type=float, default=None, 
                       help='Learning rate for pretrained model; defaults to init_lr if unspecified')
    
            
    parser.add_argument('--enc_in', type=int, default=116, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=64, help='model input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--n_layers', type=int, default=2, help='n_layers of DEFT Block')
    parser.add_argument('--pe_type', type=str, default='no', help='position embedding type')
    parser.add_argument('--dropout', type=float, default=0., help='dropout ratio')
    parser.add_argument('--revin', type=bool, default=True, help='using revin from non-stationary transformer')
    parser.add_argument('--seq_len', type=int, default=140, help='input sequence length of backbone model')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
            
    try:
        world_size_env = int(os.environ.get('WORLD_SIZE', '1'))
    except Exception:
        world_size_env = 1
    if torch.cuda.is_available() and world_size_env > 1 and not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend='nccl', init_method='env://')
    
          
    if is_dist_avail_and_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f"cuda:{local_rank}")
        config['gpu']['id'] = local_rank
        config['gpu']['use'] = True
    elif args.gpu_id is not None and config['gpu']['use']:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
        device = torch.device(f"cuda:{config['gpu']['id']}")
    else:
        config['gpu']['id'] = 0
        device = torch.device('cpu')
    
            
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    
    args.multi_site_mode = bool(getattr(args, 'multi_site_mode', config['params'].get('multi_site_mode', False)))
    args.auto_select_sites = bool(getattr(args, 'auto_select_sites', config['params'].get('auto_select_sites', True)))
    if getattr(args, 'selected_sites', None) is None:
        args.selected_sites = config['params'].get('selected_sites')
    
    if args.multi_site_mode:
        if not args.auto_select_sites:
            if args.selected_sites is None:
                if 'adni' in DATASET_NAME.lower():
                    args.selected_sites = ['55', '52', '5', '28', '25', '27', '47', '2', '59', '58']
                    print(f"[!] Detected ADNI dataset, using default ADNI site list: {args.selected_sites}")
                elif 'abide' in DATASET_NAME.lower():
                    args.selected_sites = ['PITT', 'KKI', 'YALE', 'MAX_MUN', 'TRINITY', 'STANFORD', 'OLIN', 'SDSU', 'CALTECH', 'SBL']
                    print(f"[!] Detected ABIDE dataset, using default ABIDE site list: {args.selected_sites}")
                else:
                    print(f"[!] Warning: dataset {DATASET_NAME} unknown, please provide selected_sites explicitly")
            else:
                print(f"[!] Using provided site list: {args.selected_sites}")
        else:
            print(f"[!] Detected dataset {DATASET_NAME}, will auto-select sites containing all classes")
    
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    
          
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    
                 
    if args.pretrained_lr is not None:
        params['pretrained_lr'] = float(args.pretrained_lr)
    else:
        params['pretrained_lr'] = params['init_lr']
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.threshold is not None:
        params['threshold'] = float(args.threshold)
    if args.edge_ratio is not None:
        params['edge_ratio'] = float(args.edge_ratio)
    if args.node_feat_transform is not None:
        params['node_feat_transform'] = args.node_feat_transform
    
          
    net_params = config['net_params']
    params['num_workers'] = args.num_workers
    net_params['node_num'] = 116
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    
              
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.lambda1 is not None:
        net_params['lambda1'] = float(args.lambda1)
    if args.lambda2 is not None:
        net_params['lambda2'] = float(args.lambda2)
    if args.lambda3 is not None:
        net_params['lambda3'] = float(args.lambda3)
    if args.pos_enc is not None:
        net_params['pos_enc'] = args.pos_enc
    
    net_params['in_dim'] = 116
    net_params['edge_dim'] = 116
    num_classes = 2
    net_params['n_classes'] = num_classes
    
                      
    if is_main_process():
        root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        write_pircture_file = out_dir + 'pics/pic_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file, write_pircture_file
        
        if not os.path.exists(out_dir + 'results'):
            os.makedirs(out_dir + 'results')
        if not os.path.exists(out_dir + 'configs'):
            os.makedirs(out_dir + 'configs')
    else:
        dirs = None, None, None, None, None
    
              
    if is_dist_avail_and_initialized():
        dist.barrier()
    
          
    if is_main_process():
        train_val_pipeline_gradient_causal(MODEL_NAME, DATASET_NAME, params, net_params, dirs, args)
    else:
        train_val_pipeline_gradient_causal(MODEL_NAME, DATASET_NAME, params, net_params, dirs, args)


if __name__ == '__main__':
    main()
