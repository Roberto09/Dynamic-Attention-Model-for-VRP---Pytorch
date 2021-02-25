import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_attention(query, key, value, mask=None):
    """ Function that performs scaled attention given q, k, v and mask.
    q, k, v can have multiple batches and heads, defined across the first dimensions
    and the last 2 dimensions for a given sample of them are in row vector format.
    matmul is brodcasted across batches.
    """
    qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if mask is not None: qk = qk.masked_fill(mask == 1, -1e9)
    qk = F.softmax(qk, dim=-1)
    return torch.matmul(qk, value)

class MultiHeadAttention(nn.Module):
    """ Attention Layer - multi-head scaled dot product attention (for encoder and decoder)
        Observation: This MHA is currently implemented to only support singe-gpu machines

        Args:
            num_heads: number of attention heads which will be computed in parallel
            d_model: embedding size of output AND input features
            * in reality it shouldn't be neccesary that input and ouptut features are the same dimension
              but its the current case for this class.

        Call arguments:
            q: query, shape (..., seq_len_q, depth_q)
            k: key, shape == (..., seq_len_k, depth_k)
            v: value, shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k) or None.

            Since we use scaled-product attention, we assume seq_len_k = seq_len_v

        Returns:
              attention outputs of shape (batch_size, seq_len_q, d_model)
    """
    def __init__(self, n_heads, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.n_heads, self.d_model = n_heads, d_model
        self.head_depth = self.d_model // self.n_heads
        
        assert self.d_model % self.n_heads == 0

        # define weight matrices
        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.w_out = nn.Linear(self.d_model, self.d_model, bias=False)

    def split_heads(self, tensor, batch_size):
        """ Function that splits the heads. This happens in the same tensor since this class doesn't
        support multiple-gpu. Observe inline comments for more details on shapes.
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, head_depth)
        splitted_tensor = tensor.view(batch_size, -1, self.n_heads, self.head_depth)
        return splitted_tensor.transpose(1, 2) # (batch_size, n_heads, seq_len, head_depth)

    def forward(self, query, key, value, mask=None):
        # shape of q: (batch_size, seq_len_q, d_query)
        batch_size = query.shape[0]

        # project query, key and value to d_model dimensional space
        # this is equivalent to projecting them each to a head_depth dimensional space (for every head)
        # but with a single matrix
        Q = self.wq(query) # (batch_size, seq_len_q, d_query) -> (batch_size, seq_len_q, d_model)
        K = self.wk(key) # ... -> (batch_size, seq_len_k, d_model)
        V = self.wv(value) # ... -> (batch_size, seq_len_v, d_model)

        # split individual heads
        Q = self.split_heads(Q, batch_size) # ... -> (batch_size, n_heads, seq_len_q, head_depth)
        K = self.split_heads(K, batch_size) # ... -> (batch_size, n_heads, seq_len_k, head_depth)
        V = self.split_heads(V, batch_size) # ... -> (batch_size, n_heads, seq_len_v, head_depth)
        
        
        # Add dimension to mask so that it can be broadcasted across heads
        # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
            mask = mask.unsqueeze(1)

        # perform attention for each q=(seq_len_q, head_depth), k=(seq_len_k, head_depth), v=(seq_len_v, head_depth)
        attention = scaled_attention(Q, K, V, mask) # (batch_size, n_heads, seq_len_q, head_depth)
        # transpose attention to (batch_size, seq_len_q, n_heads, head_depth)
        attention = attention.transpose(1, 2).contiguous()
        # concatenate results of all heads (batch_size, seq_len_q, self.d_model)
        attention = attention.view(batch_size, -1, self.d_model)

        # project attention to same dimension; observe this is equivalent to summing individual projection
        # as sugested in paper
        output = self.w_out(attention) # (batch_size, seq_len_q, d_model)

        return output