import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from utils.config import ConfigDict
import torchvision
from datasets.tokenizer import area_vocab
from typing import List, Optional, Tuple

class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer, non_linearity_layer).
    Examples:
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """
    def __init__(self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]) -> None:
        super(ProjectionHead, self).__init__()
        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])


    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class ClipNet(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(ClipNet, self).__init__()
        args = cfg.copy()

        backbone_args = cfg.backbone.copy()

        backbone_name = backbone_args.pop('type')
        self.num_classes = backbone_args.pop('num_classes')
        
        hidden_dim = 2048
        self.output_dim = 2048
        self.context_length = 6
        # text
        self.locLayer = nn.Linear(2, self.output_dim)
        self.token_embedding = nn.Embedding(len(area_vocab), self.output_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.output_dim))

        self.text_encoder = Transformer(width=2048, layers=3, heads=1, attn_mask = self.build_attention_mask())
        self.ln_final = LayerNorm(self.output_dim)
        self.text_projection = nn.Parameter(torch.empty(self.output_dim, self.output_dim))

        # image
        self.image_encoder = getattr(torchvision.models, backbone_name)(**backbone_args)
        input_dim = self.image_encoder.classifier[-1].in_features
        self.image_encoder.classifier = ProjectionHead([
            (input_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , self.output_dim, nn.BatchNorm1d(hidden_dim), None)
        ])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.text_encoder.width ** -0.5) * ((2 * self.text_encoder.layers) ** -0.5)
        attn_std = self.text_encoder.width ** -0.5
        fc_std = (2 * self.text_encoder.width) ** -0.5
        for block in self.text_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.text_encoder.width ** -0.5)    

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, x):
        x = self.image_encoder.features(x)
        x = self.image_encoder.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.image_encoder.classifier(x)
        return x

    def encode_text(self, text, loc):
        loc = self.locLayer(loc).unsqueeze(1) # [batch_size, 1, d_model]
        x = self.token_embedding(text) # [batch_size, n_ctx, d_model]

        x = torch.cat((x[:,:-1], loc, x[:,-1:]), dim=1)
        
        x = x + self.positional_embedding

        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
    
        x = x[:, -1] @ self.text_projection
        return x

    def forward(self, img, text, loc):
        image_features = self.encode_image(img)
        text_features = self.encode_text(text, loc)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        if self.training:
            logits_per_image = logit_scale * image_features @ text_features.t() #[B, embedding_dim] #[embedding_dim, B] -> #[B, B]
        else:
            logits_per_image = logit_scale * image_features @ text_features.t().view(self.output_dim, -1,self.num_classes).permute(1, 0, 2) #[B, embedding_dim] #[B, embedding_dim, num_classes] -> #[B, B, num_classes]
            B = logits_per_image.shape[0]
            logits_per_image = logits_per_image.view(-1, self.num_classes)[[i for i in range(0, B * B, B)], :]

        return logits_per_image
