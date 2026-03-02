
                                             
import torch
import torch.nn as nn
                                         
from torch.cuda.amp import autocast
import numpy as np
                                                                             
from metrics import accuracy_TU as accuracy
from metrics import per_class_accuracy
                                              
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, classification_report

from sklearn.metrics import confusion_matrix

def add_gaussian_noise_to_correlation(correlation_matrix, noise_std=0.01):
    
    noise = torch.randn_like(correlation_matrix) * noise_std                           
    augmented_matrix = correlation_matrix + noise             
    return augmented_matrix

def print_tensor_stats(tensor, name):
    
    with torch.no_grad():                             
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
                                                               
        percentiles = np.percentile(tensor.cpu().numpy(), [25, 50, 75])  
        print(f"{name}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}, percentiles={percentiles}")
def remap_predictions(y_pred):
    
    remapped_y_pred = np.zeros_like(y_pred)                                                                           
    for i in range(len(y_pred)):
        if y_pred[i] in [0, 1, 2]:
            remapped_y_pred[i] = 0
        elif y_pred[i] in [3, 4, 5]:
            remapped_y_pred[i] = 1
        elif y_pred[i] in [6, 7, 8]:
            remapped_y_pred[i] = 2
        elif y_pred[i] in [9, 10, 11]:
            remapped_y_pred[i] = 3
        else:
            raise ValueError(f"Unexpected prediction value: {y_pred[i]}")

    return remapped_y_pred

def group_scores_max_vectorized(y_pred_scores):
    
    remapped_y_pred_scores = np.zeros((y_pred_scores.shape[0], 4))
    remapped_y_pred_scores[:, 0] = np.max(y_pred_scores[:, 0:3], axis=1)
    remapped_y_pred_scores[:, 1] = np.max(y_pred_scores[:, 3:6], axis=1)
    remapped_y_pred_scores[:, 2] = np.max(y_pred_scores[:, 6:9], axis=1)
    remapped_y_pred_scores[:, 3] = np.max(y_pred_scores[:, 9:12], axis=1)
    return remapped_y_pred_scores

def train_epoch_sparse(model, optimizer, device, data_loader, epoch, ifvis, weights_fold=None, tau=None):
                                                                                                    
    y_true = []                 
    y_pred_scores = []                     
    y_pred = []
    ifvis = ifvis
    iftrain = True
    weights_fold = weights_fold
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    
    tau = tau
                                                                           
    for iter, batch in enumerate(data_loader):     
                                                   
        batch = batch.to(device)
                                                
                                                                        
        optimizer.zero_grad()
        batch_scores= model.forward(batch, tau)
                                                                               
        _mod = model.module if hasattr(model, 'module') else model
        loss = _mod.loss(batch_scores, batch.y, weights_fold)
                                                              
                                          
        loss.backward()

        optimizer.step()
        epoch_loss += loss.detach().item()
                                                                 
        nb_data += batch.y.size(0)
                                  
                                             
        y_true.append(batch.y.detach().cpu())
                 
        batch_probabilities = torch.softmax(batch_scores, dim=1)                   
        y_pred_scores.append(batch_probabilities.detach().cpu())
    
    epoch_loss /= (iter + 1)
                                        
    y_true = torch.cat(y_true, dim=0).numpy()
                                               
    y_pred_scores = torch.cat(y_pred_scores, dim=0).numpy()
                    
    y_pred = np.argmax(y_pred_scores, axis=1)
      
    epoch_train_acc = accuracy_score(y_true, y_pred)
    epoch_train_balanced_acc = per_class_accuracy(y_true, y_pred)

    return epoch_loss, epoch_train_acc, epoch_train_balanced_acc, optimizer

                                                   

def evaluate_network_all_metric(model, device, data_loader, epoch, ifvis=None, metric=None, pic_dir=None, fold=None):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    labels = []
    ifvis = ifvis
    metric = metric
    pic_dir = pic_dir
    with torch.no_grad():
        
        y_true = []        
        y_pred_scores = []        
        y_pred = []   
                                                                                                                                
                                                                                        
        for iter, batch in enumerate(data_loader):   
                                             
            batch = batch.to(device)
            batch_scores= model.forward(batch)
                                                      
                                                                   
            _mod = model.module if hasattr(model, 'module') else model
            loss = _mod.loss(batch_scores, batch.y)
 
            labels.append(batch.y.detach().cpu())
            y_true.append(batch.y.detach().cpu())
                                                        
                             
            batch_probabilities = torch.softmax(batch_scores, dim=1)                   
            y_pred_scores.append(batch_probabilities.detach().cpu())

            epoch_test_loss += loss.detach().item()
                                                                    
            nb_data += batch.y.size(0)
        epoch_test_loss /= (iter + 1)
                                   
                                      

        y_true = torch.cat(y_true, dim=0).numpy()
                                                   
        y_pred_scores = torch.cat(y_pred_scores, dim=0).numpy()
              
                      
        y_pred = np.argmax(y_pred_scores, axis=1)
                                            
        epoch_test_acc = accuracy_score(y_true, y_pred)
        epoch_test_balanced_acc = per_class_accuracy(y_true, y_pred)

              
        test_precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
        test_recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
        test_f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')

                   
        precision_macro = precision_score(y_true, y_pred, zero_division=0, average='macro')
                                                
        recall_macro = recall_score(y_true, y_pred, zero_division=0, average='macro')
        f1_macro = f1_score(y_true, y_pred, zero_division=0, average='macro')
                                                             
            
                                   
        unique_y_true = np.unique(y_true)
        unique_y_pred = np.unique(y_pred)
        all_classes = np.unique(np.concatenate([unique_y_true, unique_y_pred]))
        
                                           
        report = classification_report(
            y_true, y_pred, 
            zero_division=0, 
            labels=all_classes,              
            target_names=[f"Class {i}" for i in all_classes], 
            output_dict=True
        )
                  
                
                                                
        conf_matrix = confusion_matrix(y_true, y_pred, labels=all_classes)
                                     
                                                                    
              

            
                                               
                                             
                                                       
        
                       
        unique_classes = np.unique(y_true)
        num_classes_in_data = len(unique_classes)
        num_classes_in_scores = y_pred_scores.shape[1]
        
              
        print("\n" + "="*80)
        fold_info = f"Fold {fold}" if fold is not None else "Unknown fold"
        metric_info = f" ({metric})" if metric else ""
        print(f"[ROC AUC] {fold_info} | Epoch {epoch}{metric_info}")
        print("="*80)
        print(f"Model output classes: {num_classes_in_scores}")
        print(f"Classes present in test set: {unique_classes} (total {num_classes_in_data})")
        
                    
        from collections import Counter
        class_counts = Counter(y_true)
        print(f"\nTest-set sample count per class:")
        for cls in sorted(class_counts.keys()):
            print(f"  Class {cls}: {class_counts[cls]} samples")
        
                    
        all_model_classes = set(range(num_classes_in_scores))
        present_classes = set(unique_classes)
        missing_classes = all_model_classes - present_classes
        
        if missing_classes:
            print(f"\nWarning:  Warning: test set missing classes: {sorted(missing_classes)}")
            print(f"  This can happen (e.g., leave-one-site-out may miss classes).")
            print(f"  Will compute ROC AUC using only classes present in test set.")
        
                             
                                    
        y_pred_scores_adjusted = y_pred_scores[:, unique_classes]
        
                                            
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        y_true_adjusted = np.array([class_to_index[cls] for cls in y_true])
        
        print(f"\nAdjusted data:")
        print(f"  y_true_adjusted: {len(np.unique(y_true_adjusted))} classes, unique: {np.unique(y_true_adjusted)}")
        print(f"  y_pred_scores_adjusted shape: {y_pred_scores_adjusted.shape}")
        
                              
                                 
        if num_classes_in_data < 2:
            print(f"\nWarning:  Warning: only {num_classes_in_data} class present in test set, cannot compute ROC AUC")
            print(f"  Returning 0.0 as default")
            test_roc_auc = 0.0
        elif num_classes_in_data == 2:
            print(f"\nUsing binary ROC AUC")
            print(f"  y_true_adjusted uniques: {np.unique(y_true_adjusted)}")
            print(f"  y_pred_scores_adjusted shape: {y_pred_scores_adjusted.shape}")
            if not np.array_equal(np.unique(y_true_adjusted), np.array([0, 1])):
                print(f"  Warning:  Warning: y_true_adjusted not in [0,1], actual: {np.unique(y_true_adjusted)}")
                print(f"  Remapping to [0,1]...")
                min_val = y_true_adjusted.min()
                max_val = y_true_adjusted.max()
                y_true_binary = (y_true_adjusted == max_val).astype(int)
                print(f"  After remap: {np.unique(y_true_binary)}")
            else:
                y_true_binary = y_true_adjusted
            y_score_binary = y_pred_scores_adjusted[:, 1]
            print(f"  y_score_binary shape: {y_score_binary.shape}, range: [{y_score_binary.min():.6f}, {y_score_binary.max():.6f}]")
            test_roc_auc = roc_auc_score(y_true_binary, y_score_binary)
            print(f"ROC AUC computed (binary): {test_roc_auc:.6f}")
        else:
            print(f"\nUsing multiclass ROC AUC (OVR)")
            test_roc_auc = roc_auc_score(y_true_adjusted, y_pred_scores_adjusted, multi_class='ovr')
            print(f"ROC AUC computed (multiclass): {test_roc_auc:.6f}")
        
        print("="*80 + "\n")
        return epoch_test_loss, precision_macro, recall_macro, f1_macro, report, epoch_test_loss, epoch_test_acc, epoch_test_balanced_acc, test_precision, test_recall, test_f1, test_roc_auc, conf_matrix

                                                                                        
                               
                     
                               
                                 
           
                      
                                           
