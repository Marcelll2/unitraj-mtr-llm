import transformers
from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn 
import numpy as np
import os
from unitraj.models.mtr.MTR_utils import build_mlps
from peft import LoftQConfig, LoraConfig, get_peft_model

class LLM_module(nn.Module):
    def __init__(self, dim_last_layer, in_pro_without_norm = False):
        super().__init__()
        self.hidden_dim = 768
        self.cache_path = "/home/woody/iwnt/iwnt113h/UniTraj/UniTraj_llm2/unitraj/models/mtr/llm_cache"
        self.model = GPT2Model.from_pretrained(pretrained_model_name_or_path='gpt2',
                                        cache_dir=self.cache_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2', 
                                        cache_dir=self.cache_path)
        
        # loftq_config = LoftQConfig(loftq_bits=4)           # set 4bit quantization
        # lora_config = LoraConfig(init_lora_weights="loftq", loftq_config=loftq_config)
        lora_config = LoraConfig(init_lora_weights="gaussian")
        self.model = get_peft_model(self.model, lora_config)
        self.pro_flag = False
        if dim_last_layer != 768:
            print(f'In LLm_module'
                  f'dim_last_layer: {dim_last_layer}')
            # self.in_pro = build_mlps(c_in=dim_last_layer, mlp_channels=[self.hidden_dim], without_norm=in_pro_without_norm)
            self.in_pro = nn.Sequential(
                nn.Linear(dim_last_layer, self.hidden_dim, bias=False),                 
                nn.ReLU()
            ) # nn.BatchNorm1d(self.hidden_dim), 
            self.pro_flag = True
        
        
    def forward(self, input_emb: torch.Tensor):
        if self.pro_flag:
            input_emb = self.in_pro(input_emb)
        assert input_emb.shape[-1] == 768, 'Input embedding shape [-1] should have dimension 768'
        output = self.model(input_emb)
        return output['last_hidden_state']
