import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UpSample
from timm.models import create_model
from timm.models.convnext import ConvNeXtBlock
from types import MethodType

# Basic blocks
class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, norm=nn.InstanceNorm2d, act=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            norm(out_channels),
            act(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock2d(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=nn.Identity, attention_type=None, intermediate_conv=False, upsample_mode="deconv", scale_factor=2):
        super().__init__()
        if upsample_mode == "deconv":
            self.upsample = UpSample(2, in_channels, in_channels, scale_factor=scale_factor, mode=upsample_mode)
        elif upsample_mode == "pixelshuffle":
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * (scale_factor**2), 1),
                nn.PixelShuffle(scale_factor)
            )
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        self.intermediate_conv = nn.Sequential(
            ConvBnAct2d(skip_channels, skip_channels, 3, padding=1),
            ConvBnAct2d(skip_channels, skip_channels, 3, padding=1),
        ) if intermediate_conv and skip_channels > 0 else None

        self.att1 = SCSEModule2d(in_channels + skip_channels) if attention_type == "scse" else nn.Identity()
        self.conv1 = ConvBnAct2d(in_channels + skip_channels, out_channels, 3, padding=1, norm=nn.InstanceNorm2d)
        self.conv2 = ConvBnAct2d(out_channels, out_channels, 3, padding=1, norm=nn.InstanceNorm2d)
        self.att2 = SCSEModule2d(out_channels) if attention_type == "scse" else nn.Identity()

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if self.intermediate_conv:
                skip = self.intermediate_conv(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.att1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.att2(x)
        return x


class UnetDecoder2d_backbone1(nn.Module):
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels : tuple[int] =None,
        decoder_channels: tuple[int] = (256, 128, 64, 32), 
        scale_factors: tuple[int] = (2, 2, 2, 2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = "scse",
        intermediate_conv: bool = False,
        upsample_mode: str = "pixelshuffle",
    ):
        super().__init__()

        if len(encoder_channels) == 4:
            decoder_channels = decoder_channels[1:]
        self.decoder_channels = decoder_channels

        skip_channels = list(encoder_channels[1:]) + [0]  
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])

        self.blocks = nn.ModuleList([
            DecoderBlock2d(
                ic, sc, dc,
                norm_layer=norm_layer,
                attention_type=attention_type,
                intermediate_conv=intermediate_conv,
                upsample_mode=upsample_mode,
                scale_factor=scale_factors[i],
            )
            for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels))
        ])

    def forward(self, feats: list[torch.Tensor]):
        x = feats[0]
        results = [x]
        feats= feats[1:]
        for i, b in enumerate(self.blocks):
            skip = feats[i] if i < len(feats) else None
            x = b(results[-1], skip)
            results.append(x)
        return results 
    
class SegmentationHead2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2), kernel_size=3, mode="nontrainable"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsample = UpSample(2, out_channels, out_channels, scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.conv(x)
        return self.upsample(x)

def _convnext_block_forward(self, x):
    shortcut = x
    x = self.conv_dw(x)

    if self.use_conv_mlp:
        x = self.norm(x)
        x = self.mlp(x)
    else:
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

    if self.gamma is not None:
        x = x * self.gamma.reshape(1, -1, 1, 1)

    x = self.drop_path(x) + self.shortcut(shortcut)
    return x


class EnsembleNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384"
        self.backbone = create_model(backbone, pretrained=True, in_chans=5, features_only=True)
        chs = [self.backbone.feature_info[i]['num_chs'] for i in range(-1, -5, -1)]
        self.decoder1 = UnetDecoder2d_backbone1(chs,norm_layer=nn.InstanceNorm2d)
        self.head1 = SegmentationHead2d(self.decoder1.decoder_channels[-1], 1,1)

        self._update_stem(self.backbone)
        self.replace_activations(self.backbone, log=True)
        self.replace_norms(self.backbone, log=True)
        self.replace_forwards(self.backbone, log=True)

    def _update_stem(self, backbone):
        if True:
            backbone.stem_0.stride = (1, 10)
            backbone.stem_0.padding = (1, 1)
            backbone.stem_0.kernel_size = (4, 4)
            backbone.stages_1.downsample[1].stride = (2, 2)
            backbone.stages_1.downsample[1].padding = (0, 0)
            backbone.stages_2.downsample[1].stride = (2, 2)
            backbone.stages_2.downsample[1].padding = (0, 0)
            backbone.stages_3.downsample[1].stride = (2, 2)
            backbone.stages_3.downsample[1].padding = (0, 0)
            with torch.no_grad():
                w = backbone.stem_0.weight
                new_conv = nn.Conv2d(w.shape[0], w.shape[0], kernel_size=(4, 4), stride=(1, 7), padding=(1, 1))
                target_in_ch = new_conv.weight.shape[1]
                repeats = target_in_ch // w.shape[1] + 1
                expanded_w = w.repeat(1, repeats, 1, 1)[:, :target_in_ch, :, :]
                new_conv.weight.copy_(expanded_w)
                new_conv.bias.copy_(backbone.stem_0.bias)
            backbone.stem_0 = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                backbone.stem_0,
                new_conv,
            )
        else:
            raise ValueError("Custom striding not implemented.")

    def replace_activations(self, module, log=False):
        if log:
            print("Replacing all activations with GELU...")
        for name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.LeakyReLU, nn.Mish, nn.Sigmoid, nn.Tanh, nn.Softmax,
                                   nn.Hardtanh, nn.ELU, nn.SELU, nn.PReLU, nn.CELU, nn.GELU, nn.SiLU)):
                setattr(module, name, nn.GELU())
            else:
                self.replace_activations(child)

    def replace_norms(self, mod, log=False):
        if log:
            print("Replacing all norms with InstanceNorm...")
        for name, c in mod.named_children():
            n_feats = None
            if isinstance(c, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                n_feats = c.num_features
            elif isinstance(c, nn.GroupNorm):
                n_feats = c.num_channels
            elif isinstance(c, nn.LayerNorm):
                n_feats = c.normalized_shape[0]
            if n_feats is not None:
                new = nn.InstanceNorm2d(n_feats, affine=True)
                setattr(mod, name, new)
            else:
                self.replace_norms(c)

    def replace_forwards(self, mod, log=False):
        if log:
            print("Replacing forward functions...")
        for name, c in mod.named_children():
            if isinstance(c, ConvNeXtBlock):
                c.forward = MethodType(_convnext_block_forward, c)
            else:
                self.replace_forwards(c)

    def forward(self, x):
        feats = self.backbone(x)[-4:][::-1]
        x = self.decoder1(feats)[-1]
        x = self.head1(x)
        x = x[..., 1:-1, 1:-1] * 1500 + 3000
        return x
