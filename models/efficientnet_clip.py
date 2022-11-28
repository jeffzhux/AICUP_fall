import numpy as np
import torch
import torch.nn as nn
from utils.config import ConfigDict
import torchvision
from datasets.tokenizer import vocab
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

class LocClipNet(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(LocClipNet, self).__init__()
        args = cfg.copy()

        backbone_args = cfg.backbone.copy()

        backbone_name = backbone_args.pop('type')
        num_classes = backbone_args.pop('num_classes')
        batch_size = backbone_args.pop('batch_size')
        dropout_rate =  backbone_args.pop('dropout_rate') if backbone_args.get('dropout_rate') != None else None
        self.backbone = getattr(torchvision.models, backbone_name)(**backbone_args)
        
        locDim = 128
        input_dim = self.backbone.classifier[-1].in_features + locDim
        hidden_dim = 2048
        output_dim = 2048
        self.label_token = torch.tensor(range(num_classes)).cuda()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.token_embedding = nn.Embedding(num_classes, output_dim)
        self.locLayer = nn.Linear(2, locDim)
        self.backbone.classifier = ProjectionHead([
            (input_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)),
            (hidden_dim , output_dim, nn.BatchNorm1d(hidden_dim), None)
        ])


        # if dropout_rate != None:
        #     self.backbone.classifier[-2].p = dropout_rate
        # self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features + locDim, num_classes)

    def image_encoding(self, x, loc):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        loc = self.locLayer(loc)
        x = torch.cat((x, loc), dim=-1)
        x = self.backbone.classifier(x)
        return x

    def forward(self, x, loc):
        image_features = self.image_encoding(x, loc)
        text_features = self.token_embedding(self.label_token)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return logits_per_image
