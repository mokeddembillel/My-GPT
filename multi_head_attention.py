import torch 
import numpy as np 
import pandas as pd 
import pickle
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, imbd_size, hidden_size, num_heads, mask, dropout, device):
        
        super().__init__()


        self.imbd_size = imbd_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mask = mask
        self.device = device

        self.W_q_k_v = nn.Linear(self.imbd_size, self.hidden_size * 3 * self.num_heads)
        self.W_0 = nn.Linear(self.hidden_size * self.num_heads, self.imbd_size)
        self.dropout_layer_1 = nn.Dropout(dropout)
        self.dropout_layer_2 = nn.Dropout(dropout)


    def forward(self, X): # X-> (batch_size, sequence_length, imbd_size)
        batch_size, sequence_length, _ = X.shape
        q, k, v = self.W_q_k_v(X).split(self.hidden_size * self.num_heads, dim=-1) # (batch_size, sequence_length, hidden_size * num_heads) * 3

        q = q.view(batch_size, sequence_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, num_heads, sequence_length, hidden_size)
        k = k.view(batch_size, sequence_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, num_heads, sequence_length, hidden_size)
        v = v.view(batch_size, sequence_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, num_heads, sequence_length, hidden_size)

        r1 = q @ k.transpose(2,3) / math.sqrt(self.hidden_size) # (batch_size, num_heads, sequence_length, sequence_length)

        if self.mask: 
            indices = torch.triu_indices(sequence_length, sequence_length, offset=1, device=self.device)
            mask_mat = torch.zeros((1, 1, sequence_length, sequence_length), device=self.device) # (1, 1, sequence_length, sequence_length)
            mask_mat[:, :, indices[0], indices[1]] = float('-inf')
            r1 += mask_mat

        r2 = torch.softmax( r1, dim=3) # (batch_size, num_heads, sequence_length, sequence_length)
        r2 = self.dropout_layer_1(r2)
        r3 = r2 @ v # (batch_size, num_heads, sequence_length, hidden_size)

        Z = self.W_0(r3.permute((0, 2, 1, 3)).contiguous().view(batch_size, sequence_length, self.num_heads * self.hidden_size))  # (batch_size, sequence_length, imbd_size)
        Z = self.dropout_layer_2(Z)
        return Z


if __name__ == "__main__":
    pass