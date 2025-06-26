import torch
import torch.nn as nn
import pdb
from .interaction import Self_Interaction
from timm.models.layers import trunc_normal_

def calculate_patch_labels(images, boxes, fake_text_pos, num_patches=(16, 16)):
    # 获取图片的尺寸
    _, height, width = images.shape[1:4]
    
    # 计算每个 patch 的大小
    patch_height = height // num_patches[0]
    patch_width = width // num_patches[1]

    # 将 boxes 转换为张量
    # boxes = torch.tensor(boxes)  # shape: [N, 4]

    # 计算框的坐标
    box_x1 = (boxes[:, 0] * width).int()
    box_y1 = (boxes[:, 1] * height).int()
    box_w = (boxes[:, 2] * width).int()
    box_h = (boxes[:, 3] * height).int()
    
    # box_x2 = box_x1 + box_w
    # box_y2 = box_y1 + box_h

    box_x2 = box_x1 + 0.5*box_w
    box_y2 = box_y1 + 0.5*box_h

    box_x1 = box_x1 - 0.5*box_w
    box_y1 = box_y1 - 0.5*box_h
    
    # 计算 patch 的坐标
    patch_x1 = torch.arange(0, width, patch_width).view(1, -1).expand(boxes.size(0), -1).to(boxes.device)
    patch_y1 = torch.arange(0, height, patch_height).view(1, -1).expand(boxes.size(0), -1).to(boxes.device)
    patch_x2 = patch_x1 + patch_width
    patch_y2 = patch_y1 + patch_height

    # 计算每个 patch 的面积
    patch_area = patch_width * patch_height

    # 计算相交区域
    inter_x1 = torch.max(patch_x1, box_x1.view(-1, 1))
    inter_y1 = torch.max(patch_y1, box_y1.view(-1, 1))
    inter_x2 = torch.min(patch_x2, box_x2.view(-1, 1))
    inter_y2 = torch.min(patch_y2, box_y2.view(-1, 1))

    # 计算相交区域的面积

    inter_area = torch.max(torch.tensor(0), inter_x2 - inter_x1).unsqueeze(1) * torch.max(torch.tensor(0), inter_y2 - inter_y1).unsqueeze(2)

    # 判断条件：相交面积是否大于 patch 面积的一半
    labels = (inter_area > (patch_area / 2)).int()

    labels_extented = labels.view(images.shape[0], -1, 1)

    consistency_matrix = (labels_extented == labels_extented.transpose(2, 1)).int()
    
    labels_extented_it = labels.view(images.shape[0], 1, -1)
    fake_text_pos_extented = fake_text_pos.view(images.shape[0], -1, 1)

    consistency_matrix_it = ((labels_extented_it + fake_text_pos_extented)<1).int()

    return consistency_matrix, consistency_matrix_it, labels.view(images.shape[0], -1)

def get_sscore_label(img, fake_img_box, fake_text_pos, len_edge=16):
    consistency_matrix, consistency_matrix_it, labels = calculate_patch_labels(img,fake_img_box,fake_text_pos,(len_edge,len_edge))

    patch_score = consistency_matrix.sum(dim=-1)/(len_edge*len_edge)
    img_score = patch_score.sum(dim=-1)/(len_edge*len_edge)

    return consistency_matrix, labels, patch_score, img_score, consistency_matrix_it

def get_sscore_label_text(fake_text_pos):

    fake_text_pos_extend = fake_text_pos.unsqueeze(-1)
    sim_matrix = ((fake_text_pos_extend == fake_text_pos_extend.transpose(2,1))).int()
    matrix_mask = ((fake_text_pos_extend + fake_text_pos_extend.transpose(2,1))>=0)
    for i in range(fake_text_pos.shape[0]):
        sim_matrix[i].fill_diagonal_(1)
    return sim_matrix, matrix_mask

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
