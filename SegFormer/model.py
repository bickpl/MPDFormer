from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SEGFORMER_CONFIGS = {
    "b0": {
        "embed_dims": [32, 64, 160, 256],
        "num_heads": [1, 2, 5, 8],
        "depths": [2, 2, 2, 2],
        "sr_ratios": [8, 4, 2, 1],
        "decoder_dim": 256,
        "drop_path_rate": 0.1,
    },
    "b1": {
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [2, 2, 2, 2],
        "sr_ratios": [8, 4, 2, 1],
        "decoder_dim": 256,
        "drop_path_rate": 0.1,
    },
    "b2": {
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 4, 6, 3],
        "sr_ratios": [8, 4, 2, 1],
        "decoder_dim": 768,
        "drop_path_rate": 0.1,
    },
    "b3": {
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 4, 18, 3],
        "sr_ratios": [8, 4, 2, 1],
        "decoder_dim": 768,
        "drop_path_rate": 0.1,
    },
    "b4": {
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 8, 27, 3],
        "sr_ratios": [8, 4, 2, 1],
        "decoder_dim": 768,
        "drop_path_rate": 0.1,
    },
    "b5": {
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "depths": [3, 6, 40, 3],
        "sr_ratios": [8, 4, 2, 1],
        "decoder_dim": 768,
        "drop_path_rate": 0.1,
    },
}


PRETRAINED_URLS = {
    "b0": "https://hf-mirror.com/nvidia/mit-b0/resolve/main/pytorch_model.bin",
    "b1": "https://hf-mirror.com/nvidia/mit-b1/resolve/main/pytorch_model.bin",
    "b2": "https://hf-mirror.com/nvidia/mit-b2/resolve/main/pytorch_model.bin",
    "b3": "https://hf-mirror.com/nvidia/mit-b3/resolve/main/pytorch_model.bin",
    "b4": "https://hf-mirror.com/nvidia/mit-b4/resolve/main/pytorch_model.bin",
    "b5": "https://hf-mirror.com/nvidia/mit-b5/resolve/main/pytorch_model.bin",
}

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        self.block1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], drop=drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        cur += depths[0]
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], drop=drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        cur += depths[1]
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], drop=drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        cur += depths[2]
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], drop=drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        self.embed_dims = embed_dims

    def forward_features(self, x):
        outs = []
        for patch_embed, blocks, norm in [
            (self.patch_embed1, self.block1, self.norm1),
            (self.patch_embed2, self.block2, self.norm2),
            (self.patch_embed3, self.block3, self.norm3),
            (self.patch_embed4, self.block4, self.norm4),
        ]:
            x, h, w = patch_embed(x)
            for block in blocks:
                x = block(x, h, w)
            x = norm(x)
            x = x.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        return self.forward_features(x)


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes):
        super().__init__()
        self.linear_c = nn.ModuleList([MLP(ch, embedding_dim) for ch in in_channels])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        n = c4.shape[0]

        outs = []
        target_size = c1.shape[2:]
        for feat, linear in zip([c1, c2, c3, c4], self.linear_c):
            x = linear(feat).permute(0, 2, 1).reshape(n, -1, feat.shape[2], feat.shape[3])
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            outs.append(x)

        x = torch.cat(outs, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x


def _extract_state_dict(state):
    if "state_dict" in state:
        state = state["state_dict"]
    if "model" in state:
        state = state["model"]
    return state


def _adapt_pretrained_keys(state_dict):
    adapted = {}
    pending_kv = {}
    for key, value in state_dict.items():
        new_key = key
        prefixes = ["backbone.", "encoder.", "mit.", "module.backbone.", "module.encoder."]
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        if new_key.startswith("segformer.encoder."):
            new_key = new_key[len("segformer.encoder.") :]

        if new_key.startswith("patch_embeddings."):
            parts = new_key.split(".")
            stage = int(parts[1]) + 1
            if parts[2] == "proj":
                adapted["patch_embed%d.proj.%s" % (stage, parts[3])] = value
            elif parts[2] == "layer_norm":
                adapted["patch_embed%d.norm.%s" % (stage, parts[3])] = value
            continue

        if new_key.startswith("layer_norm."):
            parts = new_key.split(".")
            stage = int(parts[1]) + 1
            adapted["norm%d.%s" % (stage, parts[2])] = value
            continue

        if new_key.startswith("block."):
            parts = new_key.split(".")
            stage = int(parts[1]) + 1
            block_id = int(parts[2])
            rest = ".".join(parts[3:])
            base = "block%d.%d." % (stage, block_id)
            replacements = {
                "layer_norm_1.": "norm1.",
                "layer_norm_2.": "norm2.",
                "attention.self.query.": "attn.q.",
                "attention.self.sr.": "attn.sr.",
                "attention.self.layer_norm.": "attn.norm.",
                "attention.output.dense.": "attn.proj.",
                "mlp.dense1.": "mlp.fc1.",
                "mlp.dense2.": "mlp.fc2.",
                "mlp.dwconv.dwconv.": "mlp.dwconv.dwconv.",
            }
            for old, new in replacements.items():
                if rest.startswith(old):
                    adapted[base + new + rest[len(old) :]] = value
                    break
            else:
                if rest.startswith("attention.self.key.") or rest.startswith("attention.self.value."):
                    if not rest.endswith(".weight"):
                        continue
                    pending_kv.setdefault(base + "attn.kv.weight", {})[rest.split(".")[2]] = value
            continue

        adapted[new_key] = value

    for key, parts in pending_kv.items():
        if "key" in parts and "value" in parts:
            adapted[key] = torch.cat([parts["key"], parts["value"]], dim=0)
    return adapted


class SegFormer(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        variant="b0",
        pretrained_backbone=False,
        backbone_weights_path=None,
    ):
        super().__init__()
        if variant not in SEGFORMER_CONFIGS:
            raise ValueError("Unsupported SegFormer variant: %s" % variant)

        cfg = SEGFORMER_CONFIGS[variant]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.variant = variant
        self.pretrained_backbone = pretrained_backbone
        self.backbone_weights_path = backbone_weights_path

        self.backbone = MixVisionTransformer(
            in_chans=n_channels,
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            depths=cfg["depths"],
            sr_ratios=cfg["sr_ratios"],
            drop_path_rate=cfg["drop_path_rate"],
        )
        self.decode_head = SegFormerHead(cfg["embed_dims"], cfg["decoder_dim"], n_classes)

        if pretrained_backbone:
            self.load_pretrained_backbone(backbone_weights_path)

    def load_pretrained_backbone(self, backbone_weights_path=None):
        if backbone_weights_path:
            state = torch.load(backbone_weights_path, map_location="cpu")
        else:
            url = PRETRAINED_URLS[self.variant]
            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
        state = _extract_state_dict(state)
        state = _adapt_pretrained_keys(state)
        if self.n_channels != 3:
            weight = state.get("patch_embed1.proj.weight")
            if weight is not None:
                if self.n_channels == 1:
                    state["patch_embed1.proj.weight"] = weight.mean(dim=1, keepdim=True)
                elif self.n_channels > 3:
                    extra = weight.mean(dim=1, keepdim=True).repeat(1, self.n_channels - 3, 1, 1)
                    state["patch_embed1.proj.weight"] = torch.cat([weight, extra], dim=1)
                else:
                    state["patch_embed1.proj.weight"] = weight[:, : self.n_channels]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        loaded = len(self.backbone.state_dict()) - len(missing)
        print(
            "Loaded MiT-%s pretrained backbone keys: %d/%d, unexpected=%d"
            % (self.variant, loaded, len(self.backbone.state_dict()), len(unexpected))
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        x = self.decode_head(features)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x
