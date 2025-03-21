from functools import partial
import pdb
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import random
import sys
from models import box_ops
from tools.multilabel_metrics import get_multi_label
from consist_modeling import get_sscore_label, get_sscore_label_text
from timm.models.layers import trunc_normal_
from .METER import METERTransformerSS
from torch.nn import CrossEntropyLoss, BCELoss
from .consist_modeling import Intra_Modal_Modeling, Extra_Modal_Modeling
import math
import yaml
def score2posemb1d(pos, num_pos_feats=768, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

    return pos_x

def pos2posemb2d(pos, num_pos_feats=768, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_xy = torch.cat((pos_y, pos_x), dim=-1)
    return pos_xy

def coords_2d(x_size, y_size):
    meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
    batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
    batch_x = (batch_x + 0.5) / x_size
    batch_y = (batch_y + 0.5) / y_size
    coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
    coord_base = coord_base.view(2, -1).transpose(1, 0) # (H*W, 2)
    return coord_base

class CSCL(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 ):
        super().__init__()
       
        config_meter = yaml.load(open('configs/METER.yaml', 'r'), Loader=yaml.Loader) # Multi-modal Encoder
        
        self.args = args      
        embed_dim = config['embed_dim']
        text_width = config_meter['input_text_embed_size'] # text_width = vision_width
        vision_width = config_meter['input_image_embed_size']

        self.fusion_head = self.build_mlp(input_dim=text_width+text_width, output_dim=text_width)

        # creat itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

        # creat multi-cls head
        self.cls_head_img = self.build_mlp(input_dim=text_width, output_dim=2)
        self.cls_head_text = self.build_mlp(input_dim=text_width, output_dim=2)

        # intra_modeling
        self.img_intra_model = Intra_Modal_Modeling(12, 1024, vision_width, vision_width, 16)
        self.text_intra_model = Intra_Modal_Modeling(12, 1024, vision_width, vision_width, 8)

        # extra_modeling
        self.img_extra_model = Extra_Modal_Modeling(12, vision_width, 16)
        self.text_extra_model = Extra_Modal_Modeling(12, vision_width, 8)
        
        self.emb_img_pos = nn.Sequential(
            nn.Linear(text_width*2, text_width),
            nn.LayerNorm(text_width)
        )     

        self.emb_text_pos = nn.Sequential(
            nn.Linear(text_width, text_width),
            nn.LayerNorm(text_width)
        )

        self.apply(self._init_weights)

        # init METER #
        self.meter = METERTransformerSS(config_meter)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes
    
    def get_cos_sim(self, vectors):
        norms = torch.norm(vectors, p=2, dim=2, keepdim=True)
        normalized_vectors = vectors / norms
        similarity_matrix = torch.bmm(normalized_vectors, normalized_vectors.transpose(1, 2))
        sim_score = torch.clamp((similarity_matrix+1)/2, 0, 1)

        # sim_score_max = torch.max(sim_score, dim=-1, keepdim=True)[0]
        # sim_score_min = torch.min(sim_score, dim=-1, keepdim=True)[0]
        # sim_score = (sim_score-sim_score_min)/(sim_score_max-sim_score_min)

        patch_score = sim_score.sum(dim=-1)/vectors.shape[1]
        img_score = patch_score.sum(dim=-1)/vectors.shape[1]

        return sim_score, patch_score, img_score
    
    def forward(self, image, label, text, fake_image_box, fake_text_pos, is_train=True):
        if is_train:
            print(" only support test ", file=sys.stderr)  # 将错误信息输出到标准错误
            sys.exit(1) 

        else:
            
            ##================= multi-label convert ========================## 

            text_atts_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            text_atts_mask_bool = text_atts_mask_clone==0 # 0 = pad token 
            sim_matrix_text_mask = ((text_atts_mask_bool[:,1:].unsqueeze(-1) + text_atts_mask_bool[:,1:].unsqueeze(-1).transpose(2,1))==0)
            ##================= METER ========================## 
            batch={}
            batch["text_ids"] = text.input_ids
            batch["text_masks"] = text.attention_mask

            outputs = self.meter.infer(batch=batch, img=image)
            text_embeds_output = outputs['text_feats']
            image_embeds_output = outputs['image_feats']
            fusion_token = self.fusion_head(outputs['cls_feats'])
 
            ##================= BIC ========================## 
            # forward the positve image-text pair          
            with torch.no_grad():
                bs = image.size(0)          

            logits_real_fake = self.itm_head(fusion_token)   

            ##============ contextual consistancy ===========##
            image_atts = torch.ones(image_embeds_output.size()[:-1],dtype=torch.long).to(image.device)
            image_atts_mask_bool = (image_atts==0)
            patch_pos_emb = self.emb_img_pos(pos2posemb2d(coords_2d(16, 16).to(fusion_token.device).unsqueeze(0).repeat(bs,1,1)))
            img_patch_feat = image_embeds_output[:,1:,:]
            img_patch_feat, img_matrix_pred, _ = self.img_intra_model(img_patch_feat, image_atts_mask_bool[:,1:], patch_pos_emb)
            len_text = text_embeds_output.shape[1]-1
            token_pos_emb = self.emb_text_pos(score2posemb1d(torch.arange(0, len_text, dtype=torch.float).to(text_embeds_output.device).unsqueeze(1).repeat(bs,1,1)))
            text_token_feat = text_embeds_output[:,1:,:]
            text_token_feat, text_matrix_pred, _ = self.text_intra_model(text_token_feat, text_atts_mask_bool[:,1:], token_pos_emb, sim_matrix_text_mask)

            ##============ semantic consistancy ===========##
            agger_feat_img, sim_score_img, _ = self.img_extra_model(img_patch_feat, image_embeds_output[:,0:1,:], text_token_feat, image_atts_mask_bool[:,1:], text_atts_mask_bool[:,1:])
            agger_feat_text, sim_score_text, _ = self.text_extra_model(text_token_feat, text_embeds_output[:,0:1,:], img_patch_feat, text_atts_mask_bool[:,1:], image_atts_mask_bool[:,1:])

            ##================= IMG ========================##
            output_coord = self.bbox_head(agger_feat_img.squeeze(1)).sigmoid()
            logits_multicls_img = self.cls_head_img(agger_feat_img.squeeze(1))
            ##================= TMG ========================##  
            logits_multicls_text = self.cls_head_text(agger_feat_text.squeeze(1))
            ##==============logit merge=====================##
            it_sim_score = torch.clamp(sim_score_text, 0, 1).unsqueeze(-1)
            it_sim_score = (it_sim_score > 0.5).float() # threshold is 0.5
            it_sim_score_convert = 1 - it_sim_score
            it_sim_score = torch.cat([it_sim_score, it_sim_score_convert], dim=-1)
            logits_tok = it_sim_score

            ##================= MLC ========================## 
            logits_multicls = torch.cat([logits_multicls_img, logits_multicls_text], dim=-1)

            ##=================post process=================##
            logits_multicls_mask = ((logits_multicls[:, 0]<0.5)&(logits_multicls[:, 1]<0.5))
            output_coord[logits_multicls_mask] = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(output_coord.device)
            return logits_real_fake, logits_multicls, output_coord, logits_tok

