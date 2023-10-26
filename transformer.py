import torch 
import numpy as np 
import pandas as pd 
import pickle
import math
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from utils import get_timing_signal_1d
from transformer_block import TransformerBlock

class Transformer(nn.Module):

    def __init__(self, sentence_length=5, imbd_size=512, hidden_size=64, attention_num_heads=8, ffnn_num_layers=3, ffnn_hidden_dim=1024, mask=False, num_blocks=3):
        super().__init__()
        
        self.imbd_size = imbd_size
        self.hidden_size = hidden_size
        self.sentence_length = sentence_length
        self.attention_num_heads = attention_num_heads
        self.ffnn_num_layers = ffnn_num_layers
        self.ffnn_hidden_dim = ffnn_hidden_dim
        self.mask = mask
        self.num_blocks = num_blocks


        self.t_blocks = [
            TransformerBlock(self.sentence_length, self.imbd_size, self.hidden_size, 
                             self.attention_num_heads, self.ffnn_num_layers, 
                             self.ffnn_hidden_dim, self.mask) for _ in self.num_blocks
            ]
        self.t_blocks = nn.Sequential(*self.t_blocks)

        self.output_probs = nn.Sequential(
            nn.Linear(self.imbd_size, self.vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        pass