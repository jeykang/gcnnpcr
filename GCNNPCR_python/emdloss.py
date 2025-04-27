import torch
import torch.nn.functional as F

def pairwise_distance_sq(x, y):
    """
    x: [B, N, 3]
    y: [B, N, 3]
    returns cost: [B, N, N] pairwise squared distances
    """
    # x2: [B, N, 1]
    x2 = (x * x).sum(dim=-1, keepdim=True)
    # y2: [B, 1, N]
    y2 = (y * y).sum(dim=-1).unsqueeze(1)
    # xy: [B, N, N]
    xy = torch.matmul(x, y.transpose(1,2))

    # dist_sq = x^2 + y^2 - 2xy
    dist_sq = x2 + y2 - 2*xy
    dist_sq = F.relu(dist_sq)  # numerical stability
    return dist_sq


def sinkhorn_knopp(K, max_iters=50, eps=1e-9):
    """
    Sinkhorn-Knopp iteration to row/col normalize the kernel K
    so that each row, col sums to 1/N.
    K: [B, N, N] (non-negative)
    returns P: [B, N, N] doubly-stochastic approx. of K
    """
    B, N, _ = K.shape

    # We want each row, col to sum to 1/N => row_sum=col_sum=1/N
    # Equivalently row_sum=col_sum=1 if we want uniform marginals,
    # then we interpret them as P*(1/N).
    # For convenience, we'll aim for row_sum=col_sum=1, so the final
    # "P" is s.t. sum(P)=N. We'll incorporate 1/N in cost later if we like.

    # init
    Q = K + eps  # add small eps to avoid zeros

    for _ in range(max_iters):
        # row normalization => sum along dim=2
        row_sum = Q.sum(dim=2, keepdim=True)  # [B, N, 1]
        row_inv = 1.0 / (row_sum + eps)
        Q = Q * row_inv

        # col normalization => sum along dim=1
        col_sum = Q.sum(dim=1, keepdim=True)  # [B, 1, N]
        col_inv = 1.0 / (col_sum + eps)
        Q = Q * col_inv

    return Q


def emd_loss_sinkhorn(pred, gt, reg=0.1, max_iters=50):
    """
    Approximate EMD (Sinkhorn) between pred, gt:
    pred: [B, N, 3]
    gt:   [B, N, 3]
    reg:  Entropy regularization coefficient. Larger => smoother matrix.
    returns: scalar EMD loss
    """
    B, N, _ = pred.shape

    # 1) cost matrix [B, N, N]
    cost = pairwise_distance_sq(pred, gt)  # squared Euclidian
    # shape => [B, N, N]

    # 2) compute kernel K = exp(-cost / reg)
    K = torch.exp(- cost / reg)  # [B, N, N]

    # 3) do sinkhorn
    P = sinkhorn_knopp(K, max_iters=max_iters)  # => [B, N, N]

    # 4) approximate EMD = sum_{i,j} cost[i,j]* P[i,j]
    # But note that each row, col sums to 1 => sum(P)=N
    # If we want the average EMD, we might scale by 1/N
    # We'll just produce the mean over the batch
    emd_batch = (P * cost).sum(dim=(1,2)) / N  # -> [B]
    emd_mean = emd_batch.mean()
    return emd_mean
