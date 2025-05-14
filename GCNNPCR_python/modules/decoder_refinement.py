import torch.nn as nn
from minimal_main_4 import SPDBasedDecoder  # reuse your SPD decoder

class CoarseToFineDecoder(nn.Module):
    def __init__(self, base_decoder, refine_stages=None):
        super().__init__()
        self.base = base_decoder
        # refine_stages: list of up_factors to enable
        self.refine = nn.ModuleList([
            stage for i,stage in enumerate(self.base.stages) if i in (refine_stages or [])
        ])
    def forward(self, partial_6d, global_feat):
        # do coarse decode
        pcd = self.base.init_fc(global_feat.squeeze(-1)).view(*global_feat.shape[:2],3,-1)
        # selectively run some SPD stages
        for stage in self.refine:
            pcd, _ = stage(pcd, None, global_feat, partial_6d[..., :3])
        return pcd
