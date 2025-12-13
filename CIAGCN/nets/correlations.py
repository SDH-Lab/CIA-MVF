import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression
from sklearn.covariance import graphical_lasso


PINV_COUNTER = 0
TOTAL_PARTIAL_CALLS = 0

def _calculate_ties_single_vector(vec):
    unique_vals, counts = torch.unique(vec, return_counts=True)
    counts_gt_1 = counts[counts > 1] 
    if counts_gt_1.numel() == 0:
        return torch.tensor(0.0, device=vec.device, dtype=torch.float32)
    
    tie_adjust = (counts_gt_1.float() * (counts_gt_1.float() - 1) / 2).sum()
    return tie_adjust

def kendall_tau_correlation(x_time_series):
    num_regions, time_length = x_time_series.shape
    device = x_time_series.device

    if time_length < 2:
        return torch.eye(num_regions, device=device, dtype=torch.float32)

    x_t1_expanded = x_time_series.unsqueeze(2)
    x_t2_expanded = x_time_series.unsqueeze(1)
    
    diff_within_ts = x_t1_expanded - x_t2_expanded

    sgn_within_ts = torch.sign(diff_within_ts)

    mask_upper = torch.triu(torch.ones(time_length, time_length, dtype=torch.bool, device=device), diagonal=1)

    S_masked_flat = sgn_within_ts[:, mask_upper]
    numerator_matrix = torch.matmul(S_masked_flat, S_masked_flat.T) 

    N_x_per_region = torch.zeros(num_regions, device=device, dtype=torch.float32)
   
    for i in range(num_regions):
        N_x_per_region[i] = _calculate_ties_single_vector(x_time_series[i, :])
    
    N0 = time_length * (time_length - 1) / 2.0

    denominator_factor_x = (N0 - N_x_per_region).sqrt().clamp(min=1e-10) 

    denominator_matrix = torch.outer(denominator_factor_x, denominator_factor_x)

    kendall_tau_matrix = numerator_matrix / denominator_matrix
 
    kendall_tau_matrix.fill_diagonal_(1.0)
 
    kendall_tau_matrix = torch.clamp(kendall_tau_matrix, -1.0, 1.0)

    return kendall_tau_matrix

def pearson_correlation(x, eps=1e-8):
    mean_x = torch.mean(x, dim=1, keepdim=True)
    x_centered = x - mean_x

    cov_matrix = torch.matmul(x_centered, x_centered.T) / (x.shape[1] - 1)

    std_dev_stable = torch.sqrt(torch.diag(cov_matrix) + eps)
    
    corr_matrix = cov_matrix / torch.outer(std_dev_stable, std_dev_stable)
    
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
    
    corr_matrix.fill_diagonal_(1.0)
    
    return corr_matrix
    
def partial_correlation(x, reg_param=1e-6):
    global TOTAL_PARTIAL_CALLS
    TOTAL_PARTIAL_CALLS += 1

    num_regions, time_length = x.shape
    
    x_centered = x - x.mean(dim=1, keepdim=True)
    cov_matrix = (x_centered @ x_centered.T) / (time_length - 1)
    
    eye = torch.eye(num_regions, device=x.device)
    cov_matrix_reg = cov_matrix + reg_param * eye
    
    try:
        precision_matrix = torch.linalg.inv(cov_matrix_reg)
    except torch.linalg.LinAlgError:
        global PINV_COUNTER
        PINV_COUNTER += 1
        
        if PINV_COUNTER % 10 == 0 or PINV_COUNTER == 1:
             print(f"[WARNING] Pseudo-inverse triggered! Total count: {PINV_COUNTER} / {TOTAL_PARTIAL_CALLS} calls.")

        precision_matrix = torch.linalg.pinv(cov_matrix_reg)
    
    diag_precision = torch.diag(precision_matrix)
    diag_inv_sqrt = 1.0 / torch.sqrt(diag_precision)
    partial_corr = -precision_matrix * torch.outer(diag_inv_sqrt, diag_inv_sqrt)
    
    partial_corr = torch.clamp(partial_corr, -1.0, 1.0)
    
    partial_corr.fill_diagonal_(1.0)
    
    return partial_corr


def mutual_information(x, k=3):
    num_regions, time_length = x.shape
    x_np = x.detach().cpu().numpy()
    
    nmi_matrix = np.zeros((num_regions, num_regions))
    
    entropies = np.zeros(num_regions)
    for i in range(num_regions):
        points = x_np[i, :].reshape(-1, 1)
        neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
        distances, _ = neighbors.kneighbors(points)
        epsilon = distances[:, -1]
        
        epsilon[epsilon == 0] = 1e-10

        entropies[i] = -digamma(k) + digamma(time_length) + np.log(2 * epsilon).mean()

    for i in range(num_regions):
        for j in range(i, num_regions):
            if i == j:
                continue

            mi_value = mutual_info_regression(
                x_np[i, :].reshape(-1, 1), 
                x_np[j, :], 
                discrete_features=False, 
                n_neighbors=k,
                random_state=42
            )[0]
            
            mi_value = max(0, mi_value)

            denominator = entropies[i] + entropies[j]
            if denominator > 1e-8:
                nmi = (2 * mi_value) / denominator
            else:
                nmi = 0
            
            nmi_matrix[i, j] = np.clip(nmi, 0, 1)
            nmi_matrix[j, i] = nmi_matrix[i, j]

    nmi_tensor = torch.tensor(nmi_matrix, dtype=torch.float32, device=x.device)
    
    nmi_tensor.fill_diagonal_(1.0)
    
    return nmi_tensor


def spearman_correlation(x, eps=1e-8):
    num_regions, time_length = x.shape
    
    ranks = x.argsort(dim=1).argsort(dim=1).float()
    
    ranks_centered = ranks - ranks.mean(dim=1, keepdim=True)
    
    cov_matrix_rank = (ranks_centered @ ranks_centered.T) / (time_length - 1)
    
    std_dev_rank = torch.sqrt(torch.diag(cov_matrix_rank) + eps)
    
    std_dev_rank[std_dev_rank == 0] = 1.0
    
    corr_matrix = cov_matrix_rank / torch.outer(std_dev_rank, std_dev_rank)
    
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
    
    corr_matrix.fill_diagonal_(1.0)
    
    return corr_matrix

def partial_spearman_correlation(x, reg_param=1e-6):
    num_regions, time_length = x.shape

    ranks = x.argsort(dim=1).argsort(dim=1).float()

    ranks_centered = ranks - ranks.mean(dim=1, keepdim=True)
    cov_matrix_rank = (ranks_centered @ ranks_centered.T) / (time_length - 1)

    eye = torch.eye(num_regions, device=x.device)
    cov_matrix_rank_reg = cov_matrix_rank + reg_param * eye

    try:
        precision_matrix_rank = torch.linalg.inv(cov_matrix_rank_reg)
    except torch.linalg.LinAlgError:
        precision_matrix_rank = torch.linalg.pinv(cov_matrix_rank_reg)

    diag_precision_rank = torch.diag(precision_matrix_rank)
    diag_inv_sqrt_rank = 1.0 / torch.sqrt(diag_precision_rank)
    partial_corr = -precision_matrix_rank * torch.outer(diag_inv_sqrt_rank, diag_inv_sqrt_rank)
    
    partial_corr = torch.clamp(partial_corr, -1.0, 1.0)
    
    partial_corr.fill_diagonal_(1.0)
    
    return partial_corr

def compute_fc(time_series):
    return np.corrcoef(time_series)