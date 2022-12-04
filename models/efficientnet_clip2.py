import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from utils.config import ConfigDict
import torchvision
from datasets.tokenizer import area_vocab, vocab
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

class ClipNet(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(ClipNet, self).__init__()
        args = cfg.copy()

        backbone_args = cfg.backbone.copy()
        backbone_name = backbone_args.pop('type')
        self.num_classes = backbone_args.pop('num_classes')
        
        locDim = 128
        hidden_dim = 2048
        self.output_dim = 2048
        self.context_length = 2
        self.label_token = torch.tensor(range(self.num_classes), device='cuda')

        # text
        self.locLayer = nn.Linear(2, locDim)
        self.token_embedding = nn.Embedding(len(area_vocab), self.output_dim // self.context_length)
        self.label_embedding = nn.Embedding(len(vocab), self.output_dim)
        self.text_projection = nn.Parameter(torch.empty(self.output_dim, self.output_dim))

        # image
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_encoder = getattr(torchvision.models, backbone_name)(**backbone_args)
        input_dim = self.image_encoder.classifier[-1].in_features + locDim
        self.image_encoder.classifier = ProjectionHead([
            (input_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , self.output_dim, nn.BatchNorm1d(hidden_dim), None)
        ])   

    def encode_image(self, x, loc):
        x = self.image_encoder.features(x)
        x = self.image_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x, loc), dim=-1)
        x = self.image_encoder.classifier(x)
        return x

    def encode_text(self, text):
        x = text

        return x

    def init_parameter(self, loc, text, lam, index):
        loc = self.locLayer(loc)
        text = self.token_embedding(text)
        text = torch.flatten(text, 1)
        locs = torch.empty((0, loc.size(1)), device=loc.device)
        texts = torch.empty((0, text.size(1)), device=text.device)
        
        if lam is not None and index is not None:
            bs = loc.size(0) // len(lam)
            for i, b in enumerate(range(0, loc.size(0), bs)):

                loc_mix = lam[i] * loc[b:b+bs] + (1 - lam[i]) * loc[b:b+bs][index[b:b+bs], :]
                text_mix = lam[i] * text[b:b+bs] + (1 - lam[i]) * text[b:b+bs][index[b:b+bs], :]
                locs = torch.cat((locs, loc_mix), dim=0)
                texts = torch.cat((texts, text_mix), dim=0)
        else:
            locs = loc
            texts = text

        return locs, texts

    def forward(self, img, loc, text, lam=None, index=None):
        
        loc, text = self.init_parameter(loc, text, lam, index)

        image_features = self.encode_image(img, loc)
        text_features = self.encode_text(text)
        label_features = self.label_embedding(self.label_token)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        label_features = label_features / label_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_matrix = logit_scale * image_features @ label_features.t()
        
        # similarity matrix
        similarity_matrix = logit_scale * image_features @ text_features.t()

        return logits_matrix, similarity_matrix
