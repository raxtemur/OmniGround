import torch
from torch import nn
from torch.nn import Linear, TransformerEncoderLayer
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor


def initialize_special_embs(cfg):
    special_embs = {}
    special_embs['USER'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.bfloat16)
    special_embs['BOT'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.bfloat16)
    special_embs['SOI'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.bfloat16)
    special_embs['EOI'] = torch.normal(torch.zeros(cfg.emb_dim), torch.ones(cfg.emb_dim) / cfg.emb_dim**0.5).to(dtype=torch.bfloat16)

    special_embs['SOI'].requires_grad_()
    special_embs['EOI'].requires_grad_()
    special_embs['USER'].requires_grad_()
    special_embs['BOT'].requires_grad_()
    return special_embs
    

class VisualToGPTMapping(nn.Module):
    def __init__(self, visual_emb_dim, gpt_emb_dim, num_gpt_embs, num_heads):
        super(VisualToGPTMapping, self).__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model=visual_emb_dim, nhead=num_heads, batch_first=True, norm_first=False)
        self.linear = Linear(visual_emb_dim, gpt_emb_dim)
        self.n_embeddings = num_gpt_embs
        self.embedding_dim = gpt_emb_dim
    def forward(self, visual_embs):
        out = self.transformer_layer(visual_embs)
        out = self.linear(out).view(-1, self.n_embeddings, self.embedding_dim)
        return out

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -2
        self.select_feature = 'patch'

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

class GroundingModel(nn.Module):
    def __init__(self, gpt_emb_dim, num_gpt_embs):
        super().__init__()        
        self.n_embeddings = num_gpt_embs
        self.embedding_dim = gpt_emb_dim
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(gpt_emb_dim, gpt_emb_dim)
        self.fc2 = nn.Linear(gpt_emb_dim, 4)
        
    def forward(self, image_features):
        # image_features: expected shape [batch_size, num_gpt_embs, gpt_emb_dim]
        x = image_features.transpose(1, 2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        
        return x