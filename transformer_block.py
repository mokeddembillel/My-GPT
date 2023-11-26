import torch 
import numpy as np 
import pandas as pd 
import pickle
import math
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from utils import get_timing_signal_1d


class TransformerBlock(nn.Module):
    def __init__(self, imbd_size, hidden_size, attention_num_heads, ffnn_hidden_dim, mask, dropout, device):
        
        super().__init__()

        self.attention = MultiHeadAttention(imbd_size=imbd_size, 
                                            hidden_size=hidden_size, 
                                            num_heads=attention_num_heads,
                                            mask=mask, dropout=dropout, device=device)
        
        self.norm1 = nn.LayerNorm(imbd_size)
        self.norm2 = nn.LayerNorm(imbd_size)

        self.ffnn = nn.Sequential( 
                nn.Linear(imbd_size, ffnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(ffnn_hidden_dim, imbd_size),
                nn.Dropout(dropout)
            )

    def forward(self, X): # X-> (batch_size, sequence_length, imbd_size)
        r1 = self.attention.forward(X) # (batch_size, sequence_length, imbd_size)

        r2 = self.norm1(X + r1) # (batch_size, sequence_length, imbd_size)

        r3 = self.ffnn.forward(r2) # (batch_size, sequence_length, imbd_size)

        r4 = self.norm1(r2 + r3) # (batch_size, sequence_length, imbd_size)
        return r4

        
if __name__ == "__main__":
    pass