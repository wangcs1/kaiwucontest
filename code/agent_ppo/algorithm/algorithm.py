#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import torch
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        grad_clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "cum_reward": round(reward.mean().item(), 4),
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list["value_loss"].item(), 4),
                "policy_loss": round(info_list["policy_loss"].item(), 4),
                "entropy_loss": round(info_list["entropy_loss"].item(), 4),
                "grad_clip_norm": round(float(grad_clip_norm), 4),
                "clip_frac": round(info_list["clip_frac"], 4),
                "explained_var": round(info_list["explained_var"], 4),
                "adv_mean": round(info_list["adv_mean"], 4),
                "ret_mean": round(info_list["ret_mean"], 4),
            }
            self.logger.info(
                f"[train] cum_reward:{results['cum_reward']} total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']} clip_frac:{results['clip_frac']} "
                f"explained_var:{results['explained_var']} grad_norm:{results['grad_clip_norm']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        # Masked softmax / 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip) / 策略损失
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        adv_mean = adv.mean()
        adv_std = adv.std().clamp(min=1e-6)
        adv_norm = (adv - adv_mean) / adv_std
        policy_loss1 = -ratio * adv_norm
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv_norm
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()
        clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()

        # Value loss (Clipped) / 价值损失
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # Total loss / 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        target = tdret.detach().view(-1)
        pred = vp.detach().view(-1)
        var_y = torch.var(target)
        if torch.isfinite(var_y) and var_y.item() > 1e-8:
            explained_var = (1.0 - torch.var(target - pred) / var_y).item()
        else:
            explained_var = 0.0

        info_list = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "clip_frac": clip_frac,
            "explained_var": float(explained_var),
            "adv_mean": float(adv.mean().item()),
            "ret_mean": float(tdret.mean().item()),
        }

        return total_loss, info_list

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (suppress illegal actions).

        合法动作掩码下的 softmax（将非法动作概率压为极小值）。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = logits - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(label, dim=1)
