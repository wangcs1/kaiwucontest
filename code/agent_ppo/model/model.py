#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class EntityAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            make_fc_layer(in_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = make_fc_layer(hidden_dim, 1)

    def forward(self, entity_feat, valid_mask):
        # entity_feat: [B, N, D], valid_mask: [B, N]
        bsz, n_entity, dim = entity_feat.shape
        flat = entity_feat.reshape(bsz * n_entity, dim)
        enc = self.encoder(flat).reshape(bsz, n_entity, -1)

        logits = self.scorer(enc).squeeze(-1)
        logits = logits.masked_fill(valid_mask <= 0.0, -1e9)
        weight = torch.softmax(logits, dim=1)
        weight = weight * valid_mask
        denom = weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weight = weight / denom
        pooled = (enc * weight.unsqueeze(-1)).sum(dim=1)
        return pooled


class SpatialAttentionEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            make_fc_layer(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        feat = self.cnn(x)
        bsz, dim, h, w = feat.shape
        token = feat.reshape(bsz, dim, h * w).transpose(1, 2)
        attn_out, _ = self.attn(token, token, token, need_weights=False)
        token = self.norm(token + attn_out)
        pooled = token.mean(dim=1)
        return self.proj(pooled)


class BranchSelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            make_fc_layer(embed_dim, embed_dim * 2),
            nn.ReLU(),
            make_fc_layer(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, N, D]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_selfattn_ppo"
        self.device = device

        scalar_in = Config.HERO_DIM + Config.GLOBAL_DIM + Config.MAP_STAT_DIM

        self.scalar_encoder = nn.Sequential(
            make_fc_layer(scalar_in, 128),
            nn.ReLU(),
            make_fc_layer(128, 64),
            nn.ReLU(),
        )

        self.map_encoder = SpatialAttentionEncoder(in_channels=1, hidden_dim=64, num_heads=4)

        self.treasure_pool = EntityAttentionPool(Config.TREASURE_UNIT_DIM, 64)
        self.monster_pool = EntityAttentionPool(Config.MONSTER_UNIT_DIM, 64)
        self.buff_pool = EntityAttentionPool(Config.BUFF_UNIT_DIM, 64)
        self.branch_fusion = BranchSelfAttentionFusion(embed_dim=64, num_heads=4)

        fusion_dim = 64 + 64 + 64 + 64 + 64
        self.backbone = nn.Sequential(
            make_fc_layer(fusion_dim, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
        )

        self.actor_head = make_fc_layer(128, Config.ACTION_NUM, gain=0.01)
        self.critic_head = make_fc_layer(128, Config.VALUE_NUM, gain=1.0)

    def forward(self, obs, inference=False):
        hero_end = Config.HERO_DIM
        global_end = hero_end + Config.GLOBAL_DIM
        map_grid_end = global_end + Config.MAP_GRID_DIM
        map_stat_end = map_grid_end + Config.MAP_STAT_DIM
        treasure_end = map_stat_end + Config.TREASURE_DIM
        monster_end = treasure_end + Config.MONSTER_DIM
        buff_end = monster_end + Config.BUFF_DIM

        hero_feat = obs[:, :hero_end]
        global_feat = obs[:, hero_end:global_end]
        map_grid = obs[:, global_end:map_grid_end]
        map_stat = obs[:, map_grid_end:map_stat_end]
        treasure_feat = obs[:, map_stat_end:treasure_end]
        monster_feat = obs[:, treasure_end:monster_end]
        buff_feat = obs[:, monster_end:buff_end]

        scalar_embed = self.scalar_encoder(torch.cat([hero_feat, global_feat, map_stat], dim=1))

        map_size = Config.MAP_VIEW_SIZE
        map_embed = self.map_encoder(map_grid.reshape(-1, 1, map_size, map_size))

        t_feat = treasure_feat.reshape(-1, Config.MAX_TREASURE, Config.TREASURE_UNIT_DIM)
        t_mask = (t_feat[:, :, 0] > 0.5).float()
        t_embed = self.treasure_pool(t_feat, t_mask)

        m_feat = monster_feat.reshape(-1, Config.MAX_MONSTER, Config.MONSTER_UNIT_DIM)
        m_mask = (m_feat[:, :, 0] > 0.5).float()
        m_embed = self.monster_pool(m_feat, m_mask)

        b_feat = buff_feat.reshape(-1, Config.MAX_BUFF, Config.BUFF_UNIT_DIM)
        b_mask = (b_feat[:, :, 0] > 0.5).float()
        b_embed = self.buff_pool(b_feat, b_mask)

        branch_tokens = torch.stack([scalar_embed, map_embed, t_embed, m_embed, b_embed], dim=1)
        fused_tokens = self.branch_fusion(branch_tokens)
        hidden = self.backbone(fused_tokens.reshape(fused_tokens.shape[0], -1))
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
