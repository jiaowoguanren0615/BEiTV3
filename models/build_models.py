# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import torch
import torch.nn as nn
from timm.models import register_model
from models.model_utils import BEiT3Wrapper, _get_base_config, _get_large_config, _get_beit3_config


class TwoLayerMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            out_features,
            norm_layer,
            norm_input=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output




class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(
            self,
            args,
            num_classes,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiT3ForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        t = x[:, 1:, :]
        cls_x = self.fc_norm(t.mean(1))
        return self.head(cls_x)




@register_model
def beit3_base_patch16_224_imageclassification(pretrained=False, pretrained_cfg=None,
                                               pretrained_cfg_overlay=None, **kwargs):
    args = _get_base_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, **kwargs)
    return model


@register_model
def beit3_large_patch16_224_imageclassification(pretrained=False, pretrained_cfg=None,
                                                pretrained_cfg_overlay=None, **kwargs):
    args = _get_large_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, **kwargs)
    return model


@register_model
def beit3_giant_patch14_224_imageclassification(pretrained=False, pretrained_cfg=None,
                                                pretrained_cfg_overlay=None, **kwargs):
    args = _get_beit3_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, **kwargs)
    return model


# if __name__ == '__main__':
#     from torchinfo import summary
#     model = beit3_giant_patch14_224_imageclassification()
#     summary(model, input_size=(1, 3, 224, 224))