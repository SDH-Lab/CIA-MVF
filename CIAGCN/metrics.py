import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
def per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    accuracies = []
    
    for i in range(num_classes):
        TP = cm[i, i]
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        FN = cm[i, :].sum() - cm[i, i]
        
        if (TP + TN + FP + FN) == 0:
            accuracy = 0.0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracies.append(accuracy)
    
    average_accuracy = np.mean(accuracies)
    return average_accuracy


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def sensitivity(scores, targets):
    y_true = targets.cpu().numpy()
    y_pred = scores.cpu().numpy()

    threshold = 0.5
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def precision(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return precision_score(y_true, y_pred)


def recall(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return recall_score(y_true, y_pred)


def f1(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return f1_score(y_true, y_pred)


def roc_auc(scores, targets):
    y_true = targets
    y_pred = scores.argmax(axis=1)
    return roc_auc_score(y_true, y_pred)


def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    acc = acc / len(targets)
    return acc


def accuracy_SBM(scores, targets):
    if not isinstance(scores, np.ndarray):
        raise TypeError("scores 应该是 numpy.ndarray")
    if not isinstance(targets, np.ndarray):
        raise TypeError("targets 应该是 numpy.ndarray")
    S = targets
    C = np.argmax(scores, axis=1)
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    class_counts = np.sum(CM, axis=1)
    total_samples = np.sum(class_counts)
    class_proportions = class_counts / float(total_samples)

    acc = 100. * np.sum(pr_classes * class_proportions)
    return acc


def binary_f1_score(scores, targets):
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

  
def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc
