import torch
import math
import torch.nn as nn

def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = (torch.range(length) + start_index).float32()
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      ((num_timescales) - 1).float32())
  inv_timescales = min_timescale * torch.exp(
      torch.range(num_timescales).float32() * - log_timescale_increment)
  scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
  signal = torch.concat((torch.sin(scaled_time), torch.cos(scaled_time)), axis=1)
  signal = torch.pad(signal, [[0, 0], [0, torch.mod(channels, 2)]])
  signal = torch.reshape(signal, (1, length, channels))
  return signal


# class MLP(nn.Module):
#     def __init__(self, imbd_size=512, vocab_size=1000, ffnn_num_layers=3, ffnn_hidden_dim=1024):
#         super().__init__()

#         self.imbd_size = imbd_size
#         self.vocab_size = vocab_size
#         self.ffnn_num_layers = ffnn_num_layers
#         self.ffnn_hidden_dim = ffnn_hidden_dim

        
#         self.ffnn = nn.Sequential( 
#                 nn.Linear(self.imbd_size, self.ffnn_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(self.ffnn_hidden_dim, self.ffnn_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(self.ffnn_hidden_dim, self.vocab_size),
#                 nn.Softmax(dim=-1)
#             )

#     def forward(self, X):
#         return self.ffnn(X)