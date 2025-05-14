import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from minimal_main_4 import repulsion_loss, compute_normals_pca

class NormalConsistencyLoss(nn.Module):
    def forward(self, pred, gt):
        n_pred = compute_normals_pca(pred, k=16)
        n_gt   = compute_normals_pca(gt,  k=16)
        return 1 - (n_pred * n_gt).sum(-1).mean()

class CombinedLoss(nn.Module):
    def __init__(self, cd_w=1., rep_w=0.1, norm_w=0.01):
        super().__init__()
        self.cd_w, self.rep_w, self.norm_w = cd_w, rep_w, norm_w
    def forward(self, pred, gt):
        cd, _ = chamfer_distance(pred, gt)
        rep   = repulsion_loss(pred, k=4, threshold=0.02)
        normc = NormalConsistencyLoss()(pred, gt)
        return self.cd_w*cd + self.rep_w*rep + self.norm_w*normc
