import torch 
import torch.nn as nn
from transformer_block import TransformerBlock
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, max_sequence_length, imbd_size, hidden_size, attention_num_heads, ffnn_hidden_dim, mask, num_blocks, vocab_size, dropout, device):
        super().__init__()
        
        self.max_sequence_length = max_sequence_length
        self.device = device

        self.token_embedding = nn.Embedding(vocab_size, imbd_size)
        self.position_embedding = nn.Embedding(vocab_size, imbd_size)

        self.t_blocks = [
            TransformerBlock(imbd_size, 
                             hidden_size, attention_num_heads,
                             ffnn_hidden_dim, mask, dropout, device) for _ in range(num_blocks)
            ]
        self.t_blocks = nn.Sequential(*self.t_blocks)

        self.output = nn.Linear(imbd_size, vocab_size)

    def forward(self, idx): # idx -> (batch_size, sequence_length) # X: indexes of tokens in vocabulary list
        _, sequence_length = idx.shape

        X = self.token_embedding(idx) # (batch_size, sequence_length, imbd_size)

        p = self.position_embedding(torch.arange(sequence_length, device=self.device)).unsqueeze(0) # (1, sequence_length, imbd_size)

        X_t = X + p # (batch_size, sequence_length, imbd_size)

        r1 = self.t_blocks.forward(X_t) # (batch_size, sequence_length, imbd_size)

        output = self.output(r1) # (batch_size, sequence_length, vocab_size)

        return output  # (batch_size, sequence_length, vocab_size)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens): #  idx -> (batch_size, sequence_length)

        for _ in range(max_new_tokens):
            idx_in = idx if idx.shape[-1] <= self.max_sequence_length else idx[:, -self.max_sequence_length:]

            output = self(idx_in) # (batch_size, sequence_length, vocab_size)

            probs = F.softmax(output[:, -1, :], dim=-1) # (batch_size, vocab_size)

            new_idx = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

            idx = torch.cat([idx, new_idx], dim=-1) # (batch_size, sequence_lengt+1)
            
        return idx
            
