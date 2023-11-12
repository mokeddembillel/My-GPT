import torch 
import numpy as np 
import pandas as pd 
import pickle
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size=4, sequence_length=5, imbd_size=512, hidden_size=64, num_heads=8, mask=False):
        
        super().__init__()


        self.imbd_size = imbd_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.mask = mask
        self.batch_size = batch_size

        if self.mask: 
            self.mask_mat = torch.zeros((sequence_length, sequence_length))
            indices = torch.triu_indices(sequence_length, sequence_length, 1)
            self.mask_mat[indices[0], indices[1]] = float('-inf')
            self.mask_mat = self.mask_mat.view(1, 1, self.sequence_length, self.sequence_length) # (1, 1, sequence_length, sequence_length)


        self.W_q_k_v = nn.Linear(self.imbd_size, self.hidden_size * 3 * self.num_heads)
        self.W_0 = nn.Linear(self.hidden_size * self.num_heads, self.imbd_size)



    def forward(self, X): # X-> (batch_size, sequence_length, imbd_size)
        
        q, k, v = self.W_q_k_v(X).split(self.hidden_size * self.num_heads, dim=-1) # (batch_size, sequence_length, hidden_size * num_heads) * 3

        q = q.view(self.batch_size, self.sequence_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, num_heads, sequence_length, hidden_size)
        k = k.view(self.batch_size, self.sequence_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, num_heads, sequence_length, hidden_size)
        v = v.view(self.batch_size, self.sequence_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, num_heads, sequence_length, hidden_size)

        r1 = q @ k.transpose(2,3) / math.sqrt(self.hidden_size) # (batch_size, num_heads, sequence_length, sequence_length)
        if self.mask: 
            r1 += self.mask_mat
        r2 = torch.softmax( r1, dim=3) # (batch_size, num_heads, sequence_length, sequence_length)
        r3 = r2 @ v # (batch_size, num_heads, sequence_length, hidden_size)

        Z = self.W_0(r3.permute((0, 2, 1, 3)).view(self.batch_size, self.sequence_length, self.num_heads * self.hidden_size))  # (batch_size, sequence_length, imbd_size)

        return Z


if __name__ == "__main__":
    pass