from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm .models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn * u


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layeer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layeer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, image_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Module):
    def __init__(self,
                 image_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 flag=False,
                 ):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]   #  随机深度衰减法则   stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(image_size=image_size if i == 0 else image_size // (2 ** (i+1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i], mlp_ratio=mlp_ratio[i], drop=drop_rate, drop_path=dpr[cur + j]
                ) for j in range(depths[i])
            ])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        x = x.mean(dim=1)
        x = self.head(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x


# %%
class bayes_van(VAN):
    def __init__(self,
                 image_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 flag=False):
        super().__init__(image_size,
                         in_chans,
                         num_classes,
                         embed_dims,
                         mlp_ratio,
                         drop_rate,
                         drop_path_rate,
                         norm_layer,
                         depths,
                         num_stages,
                         flag)

        # embed_dims=[64, 128, 256, 512],
        head1 = nn.Linear(embed_dims[0], embed_dims[1]) if num_classes > 0 else nn.Identity()
        head2 = nn.Linear(embed_dims[1], embed_dims[2]) if num_classes > 0 else nn.Identity()
        head3 = nn.Linear(embed_dims[2], embed_dims[3]) if num_classes > 0 else nn.Identity()
        self.heads = nn.ModuleList([
            head1,
            head2,
            head3,
            self.head
        ])

        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        #         self.logits_bias = nn.Parameter(torch.zeros(1, num_classes))
        embed_dim = [128, 256, 512, num_classes]
        self.embed = embed_dim
        for i in range(num_stages):
            logits_layer_norm = nn.LayerNorm(self.embed[i])
            # log_prior = torch.zeros(1, self.embed[i])
            # setattr(self, f"log_prior{i + 1}", log_prior)
            setattr(self, f"logits_layer_norm{i + 1}", logits_layer_norm)
        # embed_dims=[128, 256, 512, num_classes]
        # self.logits_layer_norm = nn.LayerNorm(num_classes)
        # self.norm = None
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        # log_prior = repeat(self.log_prior, '1 n -> b n ', b=B)

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            logits_layer_norm = getattr(self, f"logits_layer_norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
                logits = x.flatten(2).transpose(1, 2)
                logits = logits.mean(dim=1)
                logits = self.heads[i](logits)
                log_prior = logits_layer_norm(logits)
                log_prior = log_prior + logits

            # x = x.flatten(2).transpose(1, 2)
            # x = norm(x)
            # if i != self.num_stages - 1:
            #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # logits = x.mean(dim=1)
            # logits = self.head(logits)

        # x = x.mean(dim=1)
        # x = self.head(x)

        return log_prior

# %%
class bayes_van2(VAN):
    def __init__(self,
                 image_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 flag=False):
        super().__init__(image_size,
                         in_chans,
                         num_classes,
                         embed_dims,
                         mlp_ratio,
                         drop_rate,
                         drop_path_rate,
                         norm_layer,
                         depths,
                         num_stages,
                         flag)

        # embed_dims=[64, 128, 256, 512],
        head1 = nn.Linear(embed_dims[0], embed_dims[1]) if num_classes > 0 else nn.Identity()
        head2 = nn.Linear(embed_dims[1], embed_dims[2]) if num_classes > 0 else nn.Identity()
        head3 = nn.Linear(embed_dims[2], embed_dims[3]) if num_classes > 0 else nn.Identity()
        self.heads = nn.ModuleList([
            head1,
            head2,
            head3,
            self.head
        ])
        embed_dim = [128, 256, 512, num_classes]
        self.embed = embed_dim
        for i in range(num_stages):
            log_prior = torch.zeros(1, self.embed[i])
            setattr(self, f"log_prior{i + 1}", log_prior)

        # log_prior = torch.zeros(1, num_classes)
        # self.register_buffer('log_prior', log_prior)
        #         self.logits_bias = nn.Parameter(torch.zeros(1, num_classes))

        # self.logits_layer_norm = nn.LayerNorm(num_classes)
        # self.norm = None
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            log_prior1 = getattr(self, f"log_prior{i + 1}")
            # log_prior = repeat(self.log_prior, '1 n -> b n ', b=B)
            print(log_prior1.shape)
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)

            for blk in block:
                x = blk(x)
                logits = x.flatten(2).transpose(1, 2)
                logits = logits.mean(dim=1)
                logits = self.heads[i](logits)
                # log_prior = self.logits_layer_norm(logits)
                log_prior = logits + log_prior1
                log_prior = F.log_softmax(log_prior, dim=-1)
                log_prior = log_prior + math.log(self.embed[i])

        #     x = x.flatten(2).transpose(1, 2)
        #     x = norm(x)
        #     if i != self.num_stages - 1:
        #         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #
        # x = x.mean(dim=1)
        # x = self.head(x)

        return log_prior


        






















