import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from .attention_graph_encoder import GraphAttentionEncoder
from .environment import AgentVRP

def set_decode_type(model, decode_type):
    pass

class AttentionDynamicModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 n_encode_layers=2,
                 n_heads=8,
                 tanh_clipping=10.
                 ):
        pass

    def set_decode_type(self, decode_type):
        pass

    def split_heads(self, tensor, batch_size):
        """Function for computing attention on several heads simultaneously
        Splits tensor to be multi headed.
        """
        # (batch_size, seq_len, output_dim) -> (batch_size, seq_len, n_heads, head_depth)
        splitted_tensor = tensor.view(batch_size, -1, self.n_heads, self.head_depth)
        return splitted_tensor.transpose(1, 2) # (batch_size, n_heads, seq_len, head_depth)

    def _select_node(self, logits):
        """Select next node based on decoding type.
        """
        pass

    def get_step_context(self, state, embeddings):
        """Takes a state and graph embeddings,
           Returns a part [h_N, D] of context vector [h_c, h_N, D],
           that is related to RL Agent last step.
        """
        # index of previous node
        prev_node = state.prev_a  # (batch_size, 1)

        # from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
        cur_embedded_node = embeddings.gather(1, prev_node.view(prev_node.shape[0], -1, 1)
                            .repeat_interleave(embeddings.shape[-1], -1)) # (batch_size, 1, input_dim)

        # add remaining capacity
        step_context = torch.cat([cur_embedded_node, self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]], dim=-1)

        return step_context  # (batch_size, 1, input_dim + 1)

    def decoder_mha(self, Q, K, V, mask=None):
        """ Computes Multi-Head Attention part of decoder
        Args:
            mask: a mask for visited nodes,
                has shape (batch_size, seq_len_q, seq_len_k), seq_len_q = 1 for context vector attention in decoder
            Q: query (context vector for decoder)
                    has shape (batch_size, n_heads, seq_len_q, head_depth) with seq_len_q = 1 for context_vector attention in decoder
            K, V: key, value (projections of nodes embeddings)
                have shape (batch_size, n_heads, seq_len_k, head_depth), (batch_size, n_heads, seq_len_v, head_depth),
                                                                with seq_len_k = seq_len_v = n_nodes for decoder
        """
        
        # Add dimension to mask so that it can be broadcasted across heads
        # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
            mask.unsqueeze(1)

        attention = scaled_attention(Q, K, V, mask) # (batch_size, n_heads, seq_len_q, head_depth)
        # transpose attention to (batch_size, seq_len_q, n_heads, head_depth)
        attention = attention.transpose(1, 2).contiguous()
        # concatenate results of all heads (batch_size, seq_len_q, self.output_dim)
        attention = attention.view(self.batch_size, -1, self.output_dim)

        output = self.w_out(attention)

        return output

    def get_log_p(self, Q, K, mask=None):
        """Single-Head attention sublayer in decoder,
        computes log-probabilities for node selection.

        Args:
            mask: mask for nodes
            Q: query (output of mha layer)
                    has shape (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context attention in decoder
            K: key (projection of node embeddings)
                    has shape  (batch_size, seq_len_k, output_dim), seq_len_k = n_nodes for decoder
        """
        
        compatibility = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        compatibility = torch.tanh(compatibility) * self.tanh_clipping
        if mask is not None: compatibility = compatibility.masked_fill(mask == 1, -1e9)

        log_p = F.softmax(compatibility, dim=-1)  # (batch_size, seq_len_q, seq_len_k)

        return log_p

    def get_log_likelihood(self, _log_p, a):
        pass

    def get_projections(self, embeddings, context_vectors):

        # we compute some projections (common for each policy step) before decoding loop for efficiency
        K = self.wk(embeddings)  # (batch_size, n_nodes, output_dim)
        K_tanh = self.wk_tanh(embeddings)  # (batch_size, n_nodes, output_dim)
        V = self.wv(embeddings)  # (batch_size, n_nodes, output_dim)
        Q_context = self.wq_context(context_vectors[:, None, :])  # (batch_size, 1, output_dim)

        # we dont need to split K_tanh since there is only 1 head; Q will be split in decoding loop
        K = self.split_heads(K, self.batch_size)  # (batch_size, num_heads, n_nodes, head_depth)
        V = self.split_heads(V, self.batch_size)  # (batch_size, num_heads, n_nodes, head_depth)

        return K_tanh, Q_context, K, V

    def forward(self, inputs, return_pi=False):

        embeddings, mean_graph_emb = self.embedder(inputs)

        self.batch_size = embeddings.shape[0]

        outputs = []
        sequences = []

        state = self.problem(inputs)

        K_tanh, Q_context, K, V = self.get_projections(embeddings, mean_graph_emb)

        # Perform decoding steps
        i = 0
        inner_i = 0

        while not state.all_finished():

            if i > 0:
                state.i = torch.zeros(1, dtype=torch.int64)
                att_mask, cur_num_nodes = state.get_att_mask()
                embeddings, context_vectors = self.embedder(inputs, att_mask, cur_num_nodes)
                K_tanh, Q_context, K, V = self.get_projections(embeddings, context_vectors)

            inner_i = 0
            while not state.partial_finished():

                step_context = self.get_step_context(state, embeddings)  # (batch_size, 1, input_dim + 1)
                Q_step_context = self.wq_step_context(step_context)  # (batch_size, 1, output_dim)
                Q = Q_context + Q_step_context

                # split heads for Q
                Q = self.split_heads(Q, self.batch_size)  # (batch_size, num_heads, 1, head_depth)

                # get current mask
                mask = state.get_mask()  # (batch_size, 1, n_nodes) True -> mask, i.e. agent can NOT go

                # compute MHA decoder vectors for current mask
                mha = self.decoder_mha(Q, K, V, mask)  # (batch_size, 1, output_dim)

                # compute probabilities
                log_p = self.get_log_p(mha, K_tanh, mask)  # (batch_size, 1, n_nodes)

                # next step is to select node
                selected = self._select_node(log_p)

                state.step(selected)

                outputs.append(log_p[:, 0, :])
                sequences.append(selected)

                inner_i += 1

            i += 1

        _log_p = torch.stack(outputs, dim=1) # (batch_size, len(outputs), nodes)
        pi = torch.stack(sequences, dim=1).to(torch.float32) # (batch_size, len(outputs))

        cost = self.problem.get_costs(inputs, pi)

        ll = self.get_log_likelihood(_log_p, pi)

        if return_pi:
            return cost, ll, pi

        return cost, ll
