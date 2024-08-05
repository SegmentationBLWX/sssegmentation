'''
Function:
    Implementation of PromptEncoder
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
from ...backbones import BuildActivation
from ...backbones.samvit import LayerNorm2d
from ...backbones.hiera import PositionEmbeddingRandom


'''PromptEncoder'''
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans, act_cfg={'type': 'GELU'}):
        super(PromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans // 4), BuildActivation(act_cfg=act_cfg),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2), LayerNorm2d(mask_in_chans), BuildActivation(act_cfg=act_cfg),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    '''getdensepe'''
    def getdensepe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    '''embedpoints'''
    def embedpoints(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forwardwithcoords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding
    '''embedboxes'''
    def embedboxes(self, boxes):
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forwardwithcoords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding
    '''embedmasks'''
    def embedmasks(self, masks):
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding
    '''getbatchsize'''
    def getbatchsize(self, points, boxes, masks):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1
    '''getdevice'''
    def getdevice(self):
        return self.point_embeddings[0].weight.device
    '''forward'''
    def forward(self, points, boxes, masks):
        bs = self.getbatchsize(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.getdevice())
        if points is not None:
            coords, labels = points
            point_embeddings = self.embedpoints(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self.embedboxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self.embedmasks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings