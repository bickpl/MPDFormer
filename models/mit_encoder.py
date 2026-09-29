import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def _load_segformer_model_module():
    model_file = Path(__file__).resolve().parents[1] / "SegFormer" / "model.py"
    spec = importlib.util.spec_from_file_location("mpdformer_segformer_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load SegFormer model definitions from %s" % model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_segformer_model = _load_segformer_model_module()
PRETRAINED_URLS = _segformer_model.PRETRAINED_URLS
SEGFORMER_CONFIGS = _segformer_model.SEGFORMER_CONFIGS
MixVisionTransformer = _segformer_model.MixVisionTransformer
_adapt_pretrained_keys = _segformer_model._adapt_pretrained_keys
_extract_state_dict = _segformer_model._extract_state_dict


MIT_CONFIGS = {key: SEGFORMER_CONFIGS[key] for key in ("b0", "b3", "b4", "b5")}


class MiTEncoder(nn.Module):
    def __init__(
        self,
        backbone="b3",
        pretrained_backbone=True,
        backbone_weights_path=None,
        freeze_encoder=False,
        freeze_stages=0,
    ):
        super().__init__()
        if backbone not in MIT_CONFIGS:
            raise ValueError("Unsupported backbone %r; use b0, b3, or b5" % backbone)

        cfg = MIT_CONFIGS[backbone]
        self.backbone = backbone
        self._frozen_stages = 0
        self.stage_channels = list(cfg["embed_dims"])
        self.encoder = MixVisionTransformer(
            in_chans=3,
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            depths=cfg["depths"],
            sr_ratios=cfg["sr_ratios"],
            drop_path_rate=cfg["drop_path_rate"],
        )

        if pretrained_backbone:
            self.load_pretrained_backbone(backbone_weights_path)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif int(freeze_stages) > 0:
            self.freeze_stages(int(freeze_stages))

    def freeze_stages(self, freeze_stages):
        self._frozen_stages = min(4, int(freeze_stages))
        for stage in range(1, self._frozen_stages + 1):
            for name in ("patch_embed%d" % stage, "block%d" % stage, "norm%d" % stage):
                module = getattr(self.encoder, name, None)
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self._frozen_stages > 0:
            for stage in range(1, self._frozen_stages + 1):
                for name in ("patch_embed%d" % stage, "block%d" % stage, "norm%d" % stage):
                    module = getattr(self.encoder, name, None)
                    if module is not None:
                        module.eval()
        return self

    def load_pretrained_backbone(self, backbone_weights_path=None):
        if backbone_weights_path:
            state = torch.load(str(backbone_weights_path), map_location="cpu")
        else:
            state = torch.hub.load_state_dict_from_url(PRETRAINED_URLS[self.backbone], map_location="cpu", progress=True)
        state = _adapt_pretrained_keys(_extract_state_dict(state))
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        loaded = len(self.encoder.state_dict()) - len(missing)
        print(
            "Loaded MiT-%s pretrained backbone keys: %d/%d, unexpected=%d"
            % (self.backbone, loaded, len(self.encoder.state_dict()), len(unexpected))
        )

    def forward(self, x):
        return self.encoder(x)

