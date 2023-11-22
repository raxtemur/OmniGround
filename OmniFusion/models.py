import random

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import transformers
from IPython.display import clear_output
from PIL import Image
from torch import nn
from torch.nn import Linear, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer


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


# from transformers import BertModel, BertConfig
# class VisualToGPTMapping(nn.Module):
#     def __init__(self, visual_emb_dim, gpt_emb_dim, num_gpt_embs, num_heads):
#         super(VisualToGPTMapping, self).__init__()
        
#         config = BertConfig(
#             hidden_size=visual_emb_dim, 
#             num_attention_heads=num_heads,
#             intermediate_size=visual_emb_dim * 4, # This is typically set to 4x of hidden_size
#             num_hidden_layers=1,
#             vocab_size=1  # Not utilized but necessary to initialize the BertModel
#         )
        
#         self.encoder = BertModel(config)
#         self.linear = nn.Linear(visual_emb_dim, gpt_emb_dim)

#     def forward(self, visual_embs):

#         # Assuming all input embeddings are valid (no padding)
#         attention_mask = torch.ones(visual_embs.shape[:2], dtype=torch.long, device=visual_embs.device)
        
#         # Passing through the BertModel encoder
#         outputs = self.encoder(inputs_embeds=visual_embs, attention_mask=attention_mask)
#         out = outputs.last_hidden_state
        
#         # Linear layer
#         out = self.linear(out)
        
#         return out
