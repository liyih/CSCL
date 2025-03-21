import torch
import torch.nn as nn
import pdb
    
class Self_Interaction_block(nn.Module):
    def __init__(self, num_head, hidden_dim, input_dim, output_dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(input_dim, num_head, dropout=0.0, batch_first=True)
        self.FFN = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
    
    def forward(self, query, query_padding_mask, attn_mask):

        feat_after_self = query + self.dropout1(self.self_attn(query=query, 
                                              key=query, 
                                              value=query,
                                              key_padding_mask=query_padding_mask,
                                              attn_mask=attn_mask)[0])
        feat_after_self = self.norm1(feat_after_self)
        output = feat_after_self + self.dropout2(self.FFN(feat_after_self))
        output = self.norm2(output)
        return output
    
class Self_Interaction(nn.Module):
    def __init__(self, num_head, hidden_dim, input_dim, output_dim, layers=3):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(Self_Interaction_block(num_head, hidden_dim, input_dim, output_dim))

    def forward(self, query, query_padding_mask, query_pos_emb=None, attn_mask=None):
        if query_pos_emb is not None:
            for i in range(len(self.layers)):
                query = self.layers[i](query + query_pos_emb, query_padding_mask, attn_mask)
        else:
            for i in range(len(self.layers)):
                query = self.layers[i](query, query_padding_mask, attn_mask) 
        return query