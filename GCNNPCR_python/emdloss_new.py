import torch
import torch.nn.functional as F
import random

def pairwise_distance_sq(x, y):
    """
    x: [B, N, 3]
    y: [B, M, 3] (N and M can be different if we downsample differently)
    Returns:
        dist_sq: [B, N, M] pairwise squared distances
    """
    # x2: [B, N, 1]
    x2 = (x * x).sum(dim=-1, keepdim=True)
    # y2: [B, 1, M]
    y2 = (y * y).sum(dim=-1).unsqueeze(1)
    # xy: [B, N, M]
    xy = torch.matmul(x, y.transpose(1,2))
    
    dist_sq = x2 + y2 - 2*xy
    dist_sq = F.relu(dist_sq)  # for numerical stability, clamp negative to 0
    return dist_sq


def sinkhorn_log_stabilized(logK, max_iters=50, eps=1e-9, warm_start_logP=None):
    """
    logK: [B, N, N] = log(exp(-cost/reg)) = - cost/reg
    max_iters: number of Sinkhorn iterations
    warm_start_logP: if provided, it's a previous solution [B, N, N] in log-space
                     that we can use as an initialization

    Returns:
      logP: [B, N, N], the stabilized log of the doubly stochastic matrix
    """
    B, N, N2 = logK.shape
    assert N == N2, "logK must be square"

    # If warm_start provided, incorporate it by combining with logK
    if warm_start_logP is not None and warm_start_logP.shape == logK.shape:
        # We combine logK with warm_start_logP in a way that still ensures
        # the cost structure is respected.
        # One simple approach: logP_0 = average of logK and warm_start
        # so that we're not ignoring the new cost, and not ignoring the old solution.
        logP = 0.5 * (logK + warm_start_logP)
    else:
        logP = logK.clone()

    for _ in range(max_iters):
        # Row normalization in log-space
        #   We want each row to sum to 1 => row-wise log-sum-exp = 0
        row_max = logP.max(dim=2, keepdim=True).values
        log_row_sum = torch.logsumexp(logP - row_max, dim=2, keepdim=True) + row_max
        logP = logP - log_row_sum

        # Column normalization in log-space
        col_max = logP.max(dim=1, keepdim=True).values
        log_col_sum = torch.logsumexp(logP - col_max, dim=1, keepdim=True) + col_max
        logP = logP - log_col_sum

    return logP


def downsample_points(points, num_samples=1024):
    """
    points: [B, N, 3]
    num_samples: how many points to sample from each cloud
    
    Returns downsampled: [B, num_samples, 3]
    """
    B, N, C = points.shape
    if num_samples >= N:
        return points

    # Random sampling for demonstration.
    # If you want more structured sampling, e.g. Farthest Point Sampling, you could implement that.
    idx = torch.randint(low=0, high=N, size=(B, num_samples), device=points.device)
    # Gather per batch
    batch_indices = torch.arange(B, device=points.device).unsqueeze(-1).expand(-1, num_samples)
    downsampled = points[batch_indices, idx, :]
    return downsampled


class SinkhornEMDLoss(torch.nn.Module):
    """
    A module to encapsulate the EMD loss with:
      - downsampling
      - log-stabilized Sinkhorn
      - optional warm starts
    """
    def __init__(self, reg=0.1, max_iters=50, num_samples=1024, use_warm_start=True):
        """
        Args:
          reg: Entropy regularization coefficient
          max_iters: number of Sinkhorn iterations
          num_samples: how many points to downsample to
          use_warm_start: whether to store and use warm starts across forward calls
        """
        super().__init__()
        self.reg = reg
        self.max_iters = max_iters
        self.num_samples = num_samples
        self.use_warm_start = use_warm_start
        
        # We'll store the last solution (in log space) for warm starts
        self.last_logP = None

    def forward(self, pred, gt):
        """
        pred: [B, N, 3]
        gt:   [B, N, 3]
        Returns:
          emd_loss: scalar
        """
        B, N, C = pred.shape
        
        # 1) Downsample if needed
        pred_ds = downsample_points(pred, self.num_samples)  # [B, K, 3] (K = num_samples)
        gt_ds   = downsample_points(gt, self.num_samples)    # [B, K, 3]

        # 2) Compute cost matrix
        cost = pairwise_distance_sq(pred_ds, gt_ds)  # [B, K, K], squared Euclidean

        # 3) logK = - cost / reg
        logK = -cost / self.reg

        # 4) Stabilized Sinkhorn
        if self.use_warm_start and self.last_logP is not None:
            logP = sinkhorn_log_stabilized(
                logK,
                max_iters=self.max_iters,
                warm_start_logP=self.last_logP
            )
        else:
            logP = sinkhorn_log_stabilized(
                logK,
                max_iters=self.max_iters
            )

        # 5) Convert logP back to P
        P = logP.exp()  # [B, K, K], doubly-stochastic approx.

        # 6) EMD ~ sum_{i,j} cost[i,j]*P[i,j], scaled by 1/K if you want average per point
        emd_batch = (P * cost).sum(dim=(1,2)) / float(self.num_samples)
        emd_mean = emd_batch.mean()

        # 7) Update warm start memory
        if self.use_warm_start:
            self.last_logP = logP.detach()

        return emd_mean

