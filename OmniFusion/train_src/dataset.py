import json
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class OmniDataset(Dataset):
    def __init__(self, cfg, tokenizer, image_processor):
        with open(cfg.json_data_path) as f:
            json_data = json.load(f)
        self.cfg = cfg
        self.json_data = json_data
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, idx):
        data_item = self.json_data[idx]
        tokens = []
        masks = []
        positions = []

        prompt_tokens = self.tokenizer.encode(f"{self.cfg.prompt}", add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_tokens.shape[-1]
        tokens.append(prompt_tokens)
        mask = prompt_len * [False]
        
        image = None
        if 'image' in data_item.keys():
            image_path = f"{self.cfg.image_folder}/{data_item['image']}"
            image = self.image_processor((Image.open(image_path)).resize((336, 336)), return_tensors='pt', do_center_crop=False)['pixel_values'][0]

            tokens.append(
                torch.tensor(
                    [(self.cfg.vision_emb_num + 2)*[self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )
            
            positions += [
                {'type': 'SOI', 'position': prompt_len},
                {'type': 'IMG', 'position': (prompt_len + 1, prompt_len + 1 + self.cfg.vision_emb_num)},
                {'type': 'EOI', 'position': prompt_len + 1 + self.cfg.vision_emb_num}
            ]

            mask += (2 + self.cfg.vision_emb_num) * [False]

        for conversation in data_item['conversations']:
            if conversation['from'] == 'human':
                positions.append({'type': 'USER', 'position': len(mask)})
            else: # from gpt
                positions.append({'type': 'BOT', 'position': len(mask)})
            mask += [False]
            tokens.append(
                torch.tensor(
                    [[self.cfg.pad_id]],
                    dtype=torch.int64,
                )
            )
                
            text = conversation['value']
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            tokens.append(text_tokens)
            if conversation['from'] == 'human':
                mask += text_tokens.shape[-1] * [False]
            else:
                mask += text_tokens.shape[-1] * [True]
        coords = data_item['coords']

        tokens = torch.cat(tokens, dim = -1)[0]
        mask = torch.tensor(mask, dtype=bool)
        return image, tokens, mask, positions, coords
        

def get_dataset(cfg, tokenizer, image_processor):

    return OmniDataset(cfg, tokenizer, image_processor)


def get_collate_function(cfg):

    def colate_fn(data):
        images, tokens, masks, positions, coords = zip(*data)

        images_mask = torch.tensor([True if image is not None else False for image in images], dtype=bool)
        if images_mask.sum() > 0:
            images = torch.stack([image for image in images if image is not None])
        else:
            images = None
        tokens = list(tokens)
        masks = list(masks)
        positions = list(positions)
        coords = list(coords)
        max_len = max([token.shape[-1] for token in tokens])

        for i in range(len(tokens)):
            pad_len = max_len - tokens[i].shape[-1]
            masks[i] = torch.cat([masks[i], torch.tensor(pad_len*[False], dtype=bool)], dim=0)
            tokens[i] = torch.cat([tokens[i], torch.tensor(pad_len*[cfg.pad_id], dtype=int)], dim=0)
    
        
        masks = torch.stack(masks)
        tokens = torch.stack(tokens)
        coords = torch.tensor(coords)
        return images, images_mask, tokens, masks, positions, coords
    return colate_fn
