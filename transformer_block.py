import torch 
import numpy as np 
import pandas as pd 
import pickle
import math
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from utils import get_timing_signal_1d


class TransformerBlock(nn.Module):
    def __init__(self, sequence_length=5, imbd_size=512, hidden_size=64, attention_num_heads=8, ffnn_num_layers=3, ffnn_hidden_dim=1024, mask=False):
        
        super().__init__()

        self.imbd_size = imbd_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.attention_num_heads = attention_num_heads
        self.ffnn_num_layers = ffnn_num_layers
        self.ffnn_hidden_dim = ffnn_hidden_dim
        self.mask = mask


        self.attention = MultiHeadAttention(sequence_length=self.sequence_length, 
                                            imbd_size=self.imbd_size, 
                                            hidden_size=self.hidden_size, 
                                            num_heads=self.attention_num_heads,
                                            mask=self.mask)
        
        self.norm1 = nn.LayerNorm(self.imbd_size)
        self.norm2 = nn.LayerNorm(self.imbd_size)
        self.norm3 = nn.LayerNorm(self.imbd_size)



        self.ffnn = nn.Sequential( 
                nn.Linear(self.imbd_size, self.ffnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.ffnn_hidden_dim, self.ffnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.ffnn_hidden_dim, self.imbd_size)
            )

    def forward(self, X): # X-> (sequence_length, imbd_size)
        r1 = self.attention.forward(X) # (sequence_length, imbd_size)

        r2 = self.norm1(X + r1) # (sequence_length, imbd_size)

        r3 = self.ffnn.forward(r2) # (sequence_length, imbd_size)

        r4 = self.norm1(r2 + r3) # (sequence_length, imbd_size)
        return r4





        
        
if __name__ == "__main__":
    pass