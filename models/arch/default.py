# Define network components here
import torch
from torch import nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     

class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act, 
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)
    

###########################################################################################################################
###########################################################################################################################



# =====================================================
# Existing PyramidPooling (keep yours if already present)
# =====================================================

class PyramidPoolingNew(nn.Module):
    def __init__(self, in_channels, out_channels,
                 scales=(4, 8, 16, 32),
                 ct_channels=1):
        super().__init__()

        self.stages = nn.ModuleList([
            self._make_stage(in_channels, scale, ct_channels)
            for scale in scales
        ])

        self.bottleneck = nn.Conv2d(
            in_channels + len(scales) * ct_channels,
            out_channels,
            kernel_size=1
        )

        self.relu = nn.GELU()

    def _make_stage(self, in_channels, scale, ct_channels):
        return nn.Sequential(
            nn.AvgPool2d(scale),
            nn.Conv2d(in_channels, ct_channels, 1, bias=False),
            nn.GELU()
        )

    def forward(self, feats):

        h, w = feats.shape[2:]

        priors = torch.cat(
            [
                F.interpolate(
                    stage(feats),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )
                for stage in self.stages
            ] + [feats],
            dim=1
        )

        return self.relu(self.bottleneck(priors))


# =====================================================
# LayerNorm2d
# =====================================================

class LayerNorm2d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        return self.norm(x)


# =====================================================
# NAFNet Components
# =====================================================

class SimpleGate(nn.Module):

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LargeKernelAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(
            dim,
            dim,
            5,
            padding=2,
            groups=dim,
            bias=False
        )

        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            7,
            padding=9,
            dilation=3,
            groups=dim,
            bias=False
        )

        self.conv1 = nn.Conv2d(
            dim,
            dim,
            1,
            bias=False
        )

    def forward(self, x):

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return x * attn


# =====================================================
# Hybrid NAF + LKA Block
# =====================================================

class HybridNAFBlock(nn.Module):

    def __init__(
        self,
        channels,
        expand=2,
        ffn_expand=2,
        use_lka=False
    ):
        super().__init__()

        dw_channel = channels * expand

        self.norm1 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(
            channels,
            dw_channel,
            1,
            bias=True
        )

        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            3,
            padding=1,
            groups=dw_channel,
            bias=True
        )

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                1,
                bias=True
            )
        )

        self.conv3 = nn.Conv2d(
            dw_channel // 2,
            channels,
            1,
            bias=True
        )

        self.beta = nn.Parameter(
            torch.zeros((1, channels, 1, 1))
        )

        self.norm2 = LayerNorm2d(channels)

        ffn_channel = ffn_expand * channels

        self.conv4 = nn.Conv2d(
            channels,
            ffn_channel,
            1,
            bias=True
        )

        self.conv5 = nn.Conv2d(
            ffn_channel // 2,
            channels,
            1,
            bias=True
        )

        self.gamma = nn.Parameter(
            torch.zeros((1, channels, 1, 1))
        )

        self.use_lka = use_lka

        if use_lka:
            self.lka = LargeKernelAttention(channels)

    def forward(self, inp):

        x = self.norm1(inp)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.sg(x)

        x = x * self.sca(x)

        x = self.conv3(x)

        if self.use_lka:
            x = self.lka(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        out = y + x * self.gamma

        return out


# =====================================================
# NewDRNet
# =====================================================

class NewDRNet(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        n_feats=256,
        n_blocks=24,
        pyramid=True
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                n_feats,
                3,
                padding=1
            ),
            nn.GELU()
        )

        self.down = nn.Sequential(
            nn.Conv2d(
                n_feats,
                n_feats,
                3,
                stride=2,
                padding=1
            ),
            nn.GELU()
        )

        blocks = []

        for i in range(n_blocks):

            blocks.append(
                HybridNAFBlock(
                    n_feats,
                    use_lka=(i % 4 == 0)
                )
            )

        self.body = nn.Sequential(*blocks)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                n_feats,
                n_feats,
                4,
                stride=2,
                padding=1
            ),
            nn.GELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                n_feats,
                n_feats,
                3,
                padding=1
            ),
            nn.GELU()
        )

        self.pyramid_module = None

        if pyramid:
            self.pyramid_module = PyramidPoolingNew(
                n_feats,
                n_feats,
                scales=(4, 8, 16, 32),
                ct_channels=n_feats // 4
            )

        self.tail = nn.Conv2d(
            n_feats,
            out_channels,
            3,
            padding=1
        )

    def forward(self, x):

        shallow = self.head(x)

        feat = self.down(shallow)

        feat = self.body(feat)

        feat = self.up(feat)

        if feat.shape[2:] != shallow.shape[2:]:
            feat = F.interpolate(
                feat,
                size=shallow.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        feat = feat + shallow

        feat = self.refine(feat)

        if self.pyramid_module is not None:
            feat = self.pyramid_module(feat)

        out = self.tail(feat)

        return out
    

########################################################################################################################
########################################################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -------------------------
# # Utility: LayerNorm for images
# # -------------------------
# class LayerNorm2dAtt(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.ln = nn.LayerNorm(c)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.permute(0, 2, 3, 1)
#         x = self.ln(x)
#         return x.permute(0, 3, 1, 2)


# # -------------------------
# # ECA Attention (safe, no heads)
# # -------------------------
# class ECALayerAtt(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)                  # B,C,1,1
#         y = y.squeeze(-1).transpose(-1, -2)  # B,1,C
#         y = self.conv(y)
#         y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
#         return x * y


# # -------------------------
# # NAF-style Gated Block (modern restoration core)
# # -------------------------
# class NAFBlockAtt(nn.Module):
#     def __init__(self, c):
#         super().__init__()

#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.conv2 = nn.Conv2d(c, c, 3, padding=1, groups=c)
#         self.conv3 = nn.Conv2d(c, c, 1)

#         self.beta = nn.Parameter(torch.zeros(1))
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.norm = LayerNorm2dAtt(c)
#         self.eca = ECALayerAtt(c)

#         self.ffn = nn.Sequential(
#             nn.Conv2d(c, c * 2, 1),
#             nn.GELU(),
#             nn.Conv2d(c * 2, c, 1)
#         )

#     def forward(self, x):
#         res = x

#         x = self.norm(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)

#         x = self.eca(x)

#         x = res + self.beta * x

#         # FFN
#         x = x + self.gamma * self.ffn(self.norm(x))
#         return x


# # -------------------------
# # Multi-scale feature fusion (safe replacement for pyramid pooling)
# # -------------------------
# class MSFusionAtt(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.scales = [1, 2, 4]
#         self.convs = nn.ModuleList([
#             nn.Conv2d(c, c, 3, padding=1, groups=c) for _ in self.scales
#         ])
#         self.fuse = nn.Conv2d(c * len(self.scales), c, 1)

#     def forward(self, x):
#         outs = []
#         for s, conv in zip(self.scales, self.convs):
#             if s == 1:
#                 outs.append(conv(x))
#             else:
#                 y = F.adaptive_avg_pool2d(x, x.shape[-1] // s)
#                 y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
#                 outs.append(conv(y))
#         return self.fuse(torch.cat(outs, dim=1))


# # -------------------------
# # Down / Up blocks
# # -------------------------
# class DownAtt(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.conv = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)

#     def forward(self, x):
#         return self.conv(x)


# class UpAtt(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.conv = nn.Conv2d(c, c // 2, 3, padding=1)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
#         return self.conv(x)


# # -------------------------
# # NEW DRNET (MAIN MODEL)
# # -------------------------
# class AttDRNet(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n_feats=128,
#         n_blocks=8,
#         pyramid=True
#     ):
#         super().__init__()

#         self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)

#         # encoder
#         self.enc1 = nn.Sequential(*[NAFBlockAtt(n_feats) for _ in range(n_blocks)])
#         self.down1 = DownAtt(n_feats)

#         self.enc2 = nn.Sequential(*[NAFBlockAtt(n_feats * 2) for _ in range(n_blocks)])
#         self.down2 = DownAtt(n_feats * 2)

#         self.mid = nn.Sequential(*[NAFBlockAtt(n_feats * 4) for _ in range(n_blocks)])

#         # decoder
#         self.up2 = UpAtt(n_feats * 4)
#         self.dec2 = nn.Sequential(*[NAFBlockAtt(n_feats * 2) for _ in range(n_blocks)])

#         self.up1 = UpAtt(n_feats * 2)
#         self.dec1 = nn.Sequential(*[NAFBlockAtt(n_feats) for _ in range(n_blocks)])

#         self.fusion = MSFusionAtt(n_feats) if pyramid else None

#         self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)

#     def forward(self, x):
#         x = self.head(x)

#         e1 = self.enc1(x)
#         e2 = self.enc2(self.down1(e1))
#         m = self.mid(self.down2(e2))

#         d2_up = self.up2(m)
#         d2_up = F.interpolate(d2_up, size=e2.shape[-2:], mode='bilinear', align_corners=False)
#         d2 = self.dec2(d2_up + e2)

#         d1_up = self.up1(d2)
#         d1_up = F.interpolate(d1_up, size=e1.shape[-2:], mode='bilinear', align_corners=False)
#         d1 = self.dec1(d1_up + e1)

#         if self.fusion is not None:
#             d1 = self.fusion(d1)

#         return self.tail(d1)

# ###############################################################################################################
# ###############################################################################################################




import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utility: LayerNorm for images
# -------------------------
class LayerNorm2dAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


# -------------------------
# ECA Attention and Improved SE Attention
# -------------------------
class ECALayerAtt(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                  # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)  # B,1,C
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y

class ImprovedSEAtt(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))

# -------------------------
# NAF-style Gated Block (modern restoration core)
# -------------------------
class SimpleGateAtt(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlockAtt(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv1 = nn.Conv2d(c, c*2, 1)
        self.conv2 = nn.Conv2d(c*2, c*2, 3, padding=1, groups=c*2)
        self.sg = SimpleGateAtt()
        self.conv3 = nn.Conv2d(c, c, 1)

        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.norm = LayerNorm2dAtt(c)
        self.eca = ImprovedSEAtt(c)

        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 4, 1),
            nn.GELU(),
            nn.Conv2d(c * 4, c, 1)
        )

    def forward(self, x):
        res = x

        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)

        x = self.eca(x)

        x = res + self.beta * x

        # FFN
        x = x + self.gamma * self.ffn(self.norm(x))
        return x


# -------------------------
# Multi-scale feature fusion (safe replacement for pyramid pooling)
# -------------------------
class MSFusionAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scales = [1, 2, 4, 8]
        self.convs = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=1, groups=c) for _ in self.scales
        ])
        self.fuse = nn.Conv2d(c * len(self.scales), c, 1)

    def forward(self, x):
        outs = []
        for s, conv in zip(self.scales, self.convs):
            if s == 1:
                outs.append(conv(x))
            else:
                y = F.adaptive_avg_pool2d(x, x.shape[-1] // s)
                y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
                outs.append(conv(y))
        return self.fuse(torch.cat(outs, dim=1))


# -------------------------
# Down / Up blocks
# -------------------------
class DownAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c // 2, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


# -------------------------
# NEW DRNET (MAIN MODEL)
# -------------------------
class AttDRNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_feats=128,
        n_blocks=8,
        pyramid=True
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_feats, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

        # encoder
        self.enc1 = nn.Sequential(*[NAFBlockAtt(n_feats) for _ in range(n_blocks)])
        self.down1 = DownAtt(n_feats)

        self.enc2 = nn.Sequential(*[NAFBlockAtt(n_feats * 2) for _ in range(n_blocks)])
        self.down2 = DownAtt(n_feats * 2)

        self.mid = nn.Sequential(*[NAFBlockAtt(n_feats * 4) for _ in range(n_blocks)])

        # decoder
        self.up2 = UpAtt(n_feats * 4)
        self.dec2 = nn.Sequential(*[NAFBlockAtt(n_feats * 2) for _ in range(n_blocks)])

        self.up1 = UpAtt(n_feats * 2)
        self.dec1 = nn.Sequential(*[NAFBlockAtt(n_feats) for _ in range(n_blocks)])

        self.fusion = MSFusionAtt(n_feats) if pyramid else None

        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)

        self.skip_fuse2 = nn.Conv2d(n_feats*4, n_feats*2, 1)
        self.skip_fuse1 = nn.Conv2d(n_feats*2, n_feats, 1)

    def forward(self, x):
        inp = x[:, :3]
        x = self.head(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))

        d2_up = self.up2(m)
        d2_up = F.interpolate(d2_up, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(
            self.skip_fuse2(
                torch.cat([d2_up, e2], dim=1)
            )
        )

        d1_up = self.up1(d2)
        d1_up = F.interpolate(d1_up, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(
            self.skip_fuse1(
                torch.cat([d1_up, e1], dim=1)
            )
        )

        if self.fusion is not None:
            d1 = self.fusion(d1)

        return self.tail(d1) + inp

###############################################################################################################
###############################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. LayerNorm (token-style but for images)
# =========================================================
class LayerNorm2dTR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


# =========================================================
# 2. Window Partitioning (Swin-style)
# =========================================================
import torch.nn.functional as F

def window_partition(x, win_size):
    b, c, h, w = x.shape

    # ✅ FORCE SAFE PAD
    pad_h = (win_size - h % win_size) % win_size
    pad_w = (win_size - w % win_size) % win_size

    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    b, c, h, w = x.shape  # updated after padding

    x = x.view(
        b,
        c,
        h // win_size,
        win_size,
        w // win_size,
        win_size
    )

    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return windows.view(-1, win_size * win_size, c), (h, w)


def window_reverse(windows, win_size, h, w, c):
    b = windows.shape[0] // ((h // win_size) * (w // win_size))

    x = windows.view(
        b,
        h // win_size,
        w // win_size,
        win_size,
        win_size,
        c
    )

    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(b, c, h, w)

    return x


# =========================================================
# 3. REAL Multi-Head Window Attention
# =========================================================
class WindowAttentionTR(nn.Module):
    def __init__(self, dim, num_heads=4, win_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape

        xw, (hp, wp) = window_partition(x, self.win_size)  # (B*nW, N, C)

        qkv = self.qkv(xw).reshape(
            xw.shape[0],
            xw.shape[1],
            3,
            self.num_heads,
            c // self.num_heads
        )

        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(xw.shape[0], xw.shape[1], c)

        out = self.proj(out)

        out = window_reverse(out, self.win_size, hp, wp, c)
        out = out[:, :, :h, :w]
        return out



# =========================================================
# 4. Transformer Block (Swin-style)
# =========================================================
class SwinBlockTR(nn.Module):
    def __init__(self, dim, heads=4, win_size=8):
        super().__init__()

        self.norm1 = LayerNorm2dTR(dim)
        self.attn = WindowAttentionTR(dim, heads, win_size)

        self.norm2 = LayerNorm2dTR(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = res + self.gamma1 * x

        b, c, h, w = x.shape
        x2 = self.norm2(x).permute(0, 2, 3, 1)
        x2 = self.ffn(x2).permute(0, 3, 1, 2)

        return x + self.gamma2 * x2


# =========================================================
# 5. CNN Down/Up (keep inductive bias)
# =========================================================
class DownTR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c * 2, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UpTR(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c // 2, 3, 1, 1)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.conv(x)


# =========================================================
# 6. CNN-Transformer Hybrid Block (Restormer-style refinement)
# =========================================================
class HybridBlockTR(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c),
            nn.Conv2d(c, c, 1)
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            nn.GELU(),
            nn.Conv2d(c * 2, c, 1)
        )

        self.norm = LayerNorm2dTR(c)

        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = x

        x = self.conv(x)
        x = res + self.g1 * x

        x = x + self.g2 * self.ffn(self.norm(x))
        return x


# =========================================================
# 7. FULL MODEL
# =========================================================
class DRNetTR(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=96, n_blocks=4):
        super().__init__()

        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)

        # encoder
        self.enc1 = nn.Sequential(*[HybridBlockTR(n_feats) for _ in range(n_blocks)])
        self.down1 = DownTR(n_feats)

        self.enc2 = nn.Sequential(*[HybridBlockTR(n_feats * 2) for _ in range(n_blocks)])
        self.down2 = DownTR(n_feats * 2)

        # transformer bottleneck (REAL attention)
        self.mid = nn.Sequential(
            *[SwinBlockTR(n_feats * 4, heads=4, win_size=8) for _ in range(n_blocks)]
        )

        # decoder
        self.up2 = UpTR(n_feats * 4)
        self.dec2 = nn.Sequential(*[HybridBlockTR(n_feats * 2) for _ in range(n_blocks)])

        self.up1 = UpTR(n_feats * 2)
        self.dec1 = nn.Sequential(*[HybridBlockTR(n_feats) for _ in range(n_blocks)])

        self.tail = nn.Conv2d(n_feats, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))

        d2 = self.dec2(self.up2(m, e2.shape[-2:]) + e2)
        d1 = self.dec1(self.up1(d2, e1.shape[-2:]) + e1)

        return self.tail(d1)

###############################################################################################################
###############################################################################################################
###############################################################################################################
# Improved DRNet with SE Attention + PyramidPooling (Based on R²SFD and CERRN)
# This is a lightweight improvement over the original DRNet
###############################################################################################################

class ImprovedResidualBlock(nn.Module):
    """Residual block with SE attention - improved version of original ResidualBlock"""
    def __init__(self, channels, dilation=1, norm=None, act=nn.ReLU(True), se_reduction=16, res_scale=1):
        super(ImprovedResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        
        # Use ImprovedSEAtt (GELU + Conv2d based) for better performance
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = ImprovedSEAtt(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer is not None:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)


class ImprovedDRNet(nn.Module):
    """
    Improved DRNet with SE attention in every residual block and pyramid pooling.
    This is a drop-in replacement for the original DRNet with better performance.
    Usage: use --inet ImprovedDRNet in training command
    """
    def __init__(self, in_channels, out_channels, n_feats=64, n_resblocks=13, 
                 norm=nn.BatchNorm2d, pyramid=True):
        super(ImprovedDRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=3, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers with SE attention (improved)
        dilation_config = [1] * n_resblocks
        self.res_module = nn.Sequential(*[
            ImprovedResidualBlock(
                n_feats, dilation=dilation_config[i], norm=norm, act=act, 
                se_reduction=16, res_scale=1) 
            for i in range(n_resblocks)
        ])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)
        self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        
        # Pyramid Pooling for multi-scale context (improved from original)
        if pyramid:
            self.pyramid_module = PyramidPoolingNew(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats//4)
        
        self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class LightweightImprovedDRNet(nn.Module):
    """
    Lightweight version of ImprovedDRNet with fewer parameters.
    Suitable for faster training with limited GPU memory.
    """
    def __init__(self, in_channels, out_channels, n_feats=32, n_resblocks=8, pyramid=True):
        super(LightweightImprovedDRNet, self).__init__()
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=3, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=None, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=None, act=act)

        # Residual layers with SE attention
        self.res_module = nn.Sequential(*[
            ImprovedResidualBlock(n_feats, se_reduction=8, res_scale=1) 
            for _ in range(n_resblocks)
        ])

        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=None, act=act)
        self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=None, act=act)
        
        if pyramid:
            self.pyramid_module = PyramidPoolingNew(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats//4)
        
        self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)
        return x