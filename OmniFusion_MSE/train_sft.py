import random
import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import CSVLogger

from models import VisualToGPTMapping, CLIPVisionTower, initialize_special_embs
from dataset import get_dataset, get_collate_function
        

class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Config(**v))
            else:
                setattr(self, k, v)
    
    def __str__(self):
        return '\n'.join(f'{key}: {value}' for key, value in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
        


class Model_pl(pl.LightningModule):
    def __init__(self, cfg, clip, special_embs, model, projection, train_dataset, collate_function):
        super().__init__()
        self.cfg = cfg
        self.clip = clip
        self.special_embs = special_embs
        self.projection = projection
        self.model = model
        self.n_embeddings = model.model.embed_tokens.weight.shape[0]
        self.loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=cfg.pad_id)
        self.train_dataset = train_dataset
        self.collate_function = collate_function
        self.n_iters = len(self.train_dataloader())
        self.save_hyperparameters('cfg')
        # self.automatic_optimization = False
        
    def configure_optimizers(self):
        
        optimizer = Adafactor(list(self.special_embs.values()) + list(self.projection.parameters()) + list(self.model.parameters()), lr=self.cfg.learning_rate, relative_step=False)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.n_iters // self.cfg.grad_accum * 0.01, num_training_steps=self.n_iters // self.cfg.grad_accum)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            
        }}
    

    def on_train_epoch_end(self):
        torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/projection.pt")
        torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/special_embeddings.pt")
        self.model.save_pretrained(f"ckpts/{self.cfg.exp_name}/tuned-model")

        
    def training_step(self, batch, batch_idx):
        images, images_mask, labels, mask, positions = batch
        if images_mask.sum() > 0:
            image_embedding = self.clip(images).to(dtype=torch.bfloat16) # preprocessing!!!
            projected_vision_embeddings = self.projection(image_embedding)
        
        embeddings = self.model.model.embed_tokens(labels)
        img_idx_counter = 0
        for i in range(len(embeddings)):
            for pos in positions[i]:
                
                if pos['type'] in self.special_embs.keys():
                    embeddings[i][pos['position']] = self.special_embs[pos['type']]
                if pos['type'] == 'IMG':
                    embeddings[i][pos['position'][0]:pos['position'][1]] = projected_vision_embeddings[img_idx_counter]
                    img_idx_counter += 1
        
        embeddings = embeddings[:, :self.cfg.max_context_len]
        labels = labels[:, :self.cfg.max_context_len]
        mask = mask[:, :self.cfg.max_context_len]
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.model(inputs_embeds=embeddings.to(dtype = torch.bfloat16), output_hidden_states=True).get("logits")[:, :-1]
        
        labels = labels[:, 1:]
        mask = mask[:, 1:]
          
        logits = logits[mask].contiguous().float()
        labels = labels[mask].contiguous()

        
        loss = self.loss_fct(logits.view(-1, self.n_embeddings), labels.view(-1)).mean()
            
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx % self.cfg.ckp_iterations_step == 0 and self.global_rank == 0:
            # print(self.model)
            os.makedirs(f"ckpts/{self.cfg.exp_name}/{batch_idx}", exist_ok=True)
            torch.save(self.projection, f"ckpts/{self.cfg.exp_name}/{batch_idx}/projection.pt")
            torch.save(self.special_embs, f"ckpts/{self.cfg.exp_name}/{batch_idx}/special_embeddings.pt")
            self.model.save_pretrained(f"ckpts/{self.cfg.exp_name}/{batch_idx}/tuned-model")
            # torch.save(self.model.state_dict(), f"ckpts/{self.cfg.exp_name}/{batch_idx}/tuned-model_state_dict.pth")
        return loss
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.collate_function, num_workers = self.cfg.num_workers, shuffle=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config-sft.json')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        config_dict = json.load(config_file)
    cfg = Config(**config_dict)
    
    ### Define models
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_ckp, use_fast=False)
    unk_id = tokenizer.encode("<unk>", add_special_tokens=False)[0]
    cfg.pad_id = unk_id
    os.makedirs(f"ckpts/{cfg.exp_name}", exist_ok=True)
    logger = CSVLogger("ckpts", name=cfg.exp_name)
    cfg.exp_name = os.path.join(cfg.exp_name, f'version_{logger.version}')
    
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model_ckp, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
    # print(model)

    clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
    clip.load_model()
    clip = clip.to(dtype=torch.bfloat16)
    
    projection = torch.load(f'{cfg.pretrain_path}/projection.pt', map_location=f'cpu')
    projection.transformer_layer.norm_first = False
    special_embs = torch.load(f'{cfg.pretrain_path}/special_embeddings.pt', map_location = f'cpu')
    special_embs['SOI'].requires_grad_()
    special_embs['EOI'].requires_grad_()
    special_embs['USER'].requires_grad_()
    special_embs['BOT'].requires_grad_()
    
    # special_embs = initialize_special_embs(cfg)
    freeze(clip)
    
    ### Work with data
    train_dataset = get_dataset(cfg, tokenizer, clip.image_processor)
    collate_function = get_collate_function(cfg)

    devices_count = torch.cuda.device_count()
    module = Model_pl(cfg, clip, special_embs, model, projection, train_dataset, collate_function)
    trainer = pl.Trainer(devices=devices_count, max_epochs=cfg.n_epochs, logger=logger, accumulate_grad_batches=cfg.grad_accum, strategy='ddp_find_unused_parameters_true')
    trainer.fit(module)
    
    # run()






