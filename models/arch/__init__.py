# Add your custom network here
from .default import DRNet, NewDRNet, AttDRNet, DRNetTR, ImprovedDRNet  # 只添加 ImprovedDRNet 到导入
import torch.nn as nn

def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


#def errnet(in_channels, out_channels, **kwargs):
#    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)

# NAF AND LKA
# def errnet(in_channels, out_channels, **kwargs):
#     return NewDRNet(
#         in_channels,
#         out_channels,
#         n_feats=256,
#         n_blocks=13,
#         pyramid=True
#     )


# # ========== 原代码（保持不变，不注释） ==========
#def errnet(in_channels, out_channels, **kwargs):
#    return AttDRNet(in_channels, out_channels, n_feats=96, n_blocks=8)
# ===============================================

 def errnet(in_channels, out_channels, **kwargs):
     return DRNetTR(in_channels, out_channels, n_feats=96, n_blocks=8)


# ========== 新增：使用 ImprovedDRNet 的版本（可选，通过 --inet 指定） ==========
def improved_errnet(in_channels, out_channels, **kwargs):
    return ImprovedDRNet(in_channels, out_channels, n_feats=256, n_resblocks=13, pyramid=True)
# =================================================================================