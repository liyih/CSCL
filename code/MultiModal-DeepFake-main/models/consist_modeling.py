import torch
import torch.nn as nn
import pdb
from .interaction import Self_Interaction
from timm.models.layers import trunc_normal_

class Intra_Modal_Modeling(nn.Module):
    
    def __init__(self, num_head, hidden_dim, input_dim, output_dim, tok_num):
        super().__init__()

        self.correlation_model = Self_Interaction(num_head, hidden_dim, input_dim, output_dim, layers=3)
        self.consist_encoder = nn.Sequential(nn.Linear(output_dim, 256),
                                                  nn.LayerNorm(256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 128),
                                                  nn.LayerNorm(128),
                                                  nn.GELU(),
                                                  nn.Linear(128, 64))
        self.token_number = tok_num
        self.aggregator = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.aggregator_2 = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp_2 = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.num_head = 4

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )
    
    def forward(self, feats, mask, pos_emb, matrix_mask=None):
        
        B, N, C = feats.shape
        feats = self.correlation_model(feats, mask, pos_emb)
        consist_feats = self.consist_encoder(feats)

        norms = torch.norm(consist_feats, p=2, dim=2, keepdim=True)
        normalized_vectors = consist_feats / norms
        similarity_matrix = torch.bmm(normalized_vectors, normalized_vectors.transpose(1, 2))
        similarity_matrix = torch.clamp((similarity_matrix+1)/2, 0, 1)

        if mask.sum() > 0: # for text inputs
            similarity_matrix_unsim = similarity_matrix.clone()
            similarity_matrix_unsim[~matrix_mask] = 2

            similarity_matrix_sim = similarity_matrix.clone()
            similarity_matrix_sim[~matrix_mask] = -1
            diagonal_mask = torch.eye(N, device=feats.device).unsqueeze(0).expand(B, N, N)
            similarity_matrix_sim = similarity_matrix_sim - diagonal_mask

        else: # for image inputs
            similarity_matrix_unsim = similarity_matrix.clone()
            similarity_matrix_sim = similarity_matrix.clone()
            diagonal_mask = torch.eye(N, device=feats.device).unsqueeze(0).expand(B, N, N)
            similarity_matrix_sim = similarity_matrix_sim - diagonal_mask # ignore them self

        unsim_feats_index = torch.topk(similarity_matrix_unsim, self.token_number, dim=-1, largest=False)[1]
        unsim_attn_mask = torch.ones([B, N, N], dtype=torch.bool).to(unsim_feats_index.device)
        batch_indices = torch.arange(B).view(B, 1, 1) # 形状 (B, N, m)
        row_indices = torch.arange(N).view(1, N, 1)   # 形状 (B, N, m)
        unsim_attn_mask[batch_indices, row_indices, unsim_feats_index] = False
        unsim_attn_mask = unsim_attn_mask.repeat(self.num_head ,1,1)

        sim_feats_index = torch.topk(similarity_matrix_sim, self.token_number, dim=-1, largest=True)[1]
        sim_attn_mask = torch.ones([B, N, N], dtype=torch.bool).to(sim_feats_index.device)
        batch_indices = torch.arange(B).view(B, 1, 1) # 形状 (B, N, m)
        row_indices = torch.arange(N).view(1, N, 1)   # 形状 (B, N, m)
        sim_attn_mask[batch_indices, row_indices, sim_feats_index] = False
        sim_attn_mask = sim_attn_mask.repeat(self.num_head ,1,1)
        
        feats = feats + self.aggregator_mlp(self.aggregator(query=feats, 
                                              key=feats, 
                                              value=feats,
                                              attn_mask=sim_attn_mask)[0])
        
        feats = feats + self.aggregator_mlp_2(self.aggregator_2(query=feats, 
                                              key=feats, 
                                              value=feats,
                                              attn_mask=unsim_attn_mask)[0])

        return feats, similarity_matrix, consist_feats
    

class Extra_Modal_Modeling(nn.Module):
    
    def __init__(self, num_head, output_dim, tok_num):
        super().__init__()

        self.feat_encoder = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.cross_encoder = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.token_number = tok_num

        self.consist_encoder_feat = nn.Sequential(nn.Linear(output_dim, 256),
                                                  nn.LayerNorm(256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 128),
                                                  nn.LayerNorm(128),
                                                  nn.GELU(),
                                                  nn.Linear(128, 64))
        
        self.consist_encoder_cross = nn.Sequential(nn.Linear(output_dim, 256),
                                                  nn.LayerNorm(256),
                                                  nn.GELU(),
                                                  nn.Linear(256, 128),
                                                  nn.LayerNorm(128),
                                                  nn.GELU(),
                                                  nn.Linear(128, 64))
        
        self.cls_token_cross = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.aggregator_cross = nn.MultiheadAttention(output_dim, num_head, dropout=0.0, batch_first=True)
        self.norm_layer_cross =nn.LayerNorm(output_dim)

        self.cls_token_feat = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.aggregator_feat = nn.MultiheadAttention(output_dim, num_head, dropout=0.0, batch_first=True)
        self.norm_layer_feat =nn.LayerNorm(output_dim)

        self.aggregator = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp = self.build_mlp(input_dim=output_dim, output_dim=output_dim)
        self.aggregator_2 = nn.MultiheadAttention(output_dim, 4, dropout=0.0, batch_first=True)
        self.aggregator_mlp_2 = self.build_mlp(input_dim=output_dim, output_dim=output_dim)

        trunc_normal_(self.cls_token_cross, std=.02)
        trunc_normal_(self.cls_token_feat, std=.02)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )
    
    def forward(self, feats, gloabl_feature, cross_feat, feats_mask, cross_mask):
        
        bs, _, _ = feats.shape

        feats = self.feat_encoder(feats)
        cross_feat = self.cross_encoder(cross_feat)

        cls_token_cross = self.cls_token_cross.expand(bs, -1, -1)
        feat_aggr_cross = self.aggregator_cross(query=self.norm_layer_cross(cls_token_cross), 
                                            key=self.norm_layer_cross(cross_feat), 
                                            value=self.norm_layer_cross(cross_feat),
                                            key_padding_mask=cross_mask)[0]
        
        feats_consist = self.consist_encoder_feat(feats)
        cross_feats_consist = self.consist_encoder_feat(feat_aggr_cross)

        norms_feat = torch.norm(feats_consist, p=2, dim=2, keepdim=True)
        norms_cross = torch.norm(cross_feats_consist, p=2, dim=2, keepdim=True)
        sim_matrix = torch.bmm(feats_consist/norms_feat, (cross_feats_consist/norms_cross).transpose(1, 2))
        sim_matrix = torch.clamp((sim_matrix+1)/2, 0, 1).squeeze()

        cls_token = self.cls_token_feat.expand(bs, -1, -1)
        global_feats_mask = torch.zeros(feats_mask.shape[0], 1).bool().to(feats_mask.device)
        feat_aggr = self.aggregator_feat(query=self.norm_layer_feat(cls_token), 
                                            key=self.norm_layer_feat(torch.cat([gloabl_feature, feats], dim=1)), 
                                            value=self.norm_layer_feat(torch.cat([gloabl_feature, feats], dim=1)),
                                            key_padding_mask=torch.cat([global_feats_mask,feats_mask],dim=1))[0]
        
        if feats_mask.sum() > 0: # for text inputs
            sim_score = sim_matrix.clone()
            sim_score[feats_mask] = -1

            unsim_score = sim_matrix.clone()
            unsim_score[feats_mask] = 2

        else: # for image inputs
            sim_score = sim_matrix.clone()
            unsim_score = sim_matrix.clone()

        unsim_index = torch.topk(unsim_score, self.token_number, dim=-1, largest=False)[1]
        unsim_patch = feats[torch.arange(feats.shape[0]).unsqueeze(1), unsim_index]

        sim_index = torch.topk(sim_score, self.token_number, dim=-1, largest=True)[1]
        sim_patch = feats[torch.arange(feats.shape[0]).unsqueeze(1), sim_index]

        feat_aggr = feat_aggr + self.aggregator_mlp(self.aggregator(query=feat_aggr, 
                                              key=sim_patch, 
                                              value=sim_patch)[0])
        
        feat_aggr = feat_aggr + self.aggregator_mlp_2(self.aggregator_2(query=feat_aggr, 
                                              key=unsim_patch, 
                                              value=unsim_patch)[0])
        
        return feat_aggr, sim_matrix, feats_consist
