#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import math
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.infer_step = 0
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, remain_info = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        return obs_data, remain_info

    def predict(self, list_obs_data, is_eval=False):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        if not is_eval:
            self.infer_step += 1

        logits, value, prob = self._run_model(feature, legal_action, is_eval=is_eval)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data], is_eval=True)
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action, is_eval=False):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]
        logits_biased = np.array(logits_np, copy=True)

        if Config.RULE_GUIDE_ENABLE and (not is_eval or Config.RULE_GUIDE_ENABLE_EVAL):
            logits_biased = self._apply_rule_guided_logit_bias(logits_biased, feature, legal_action)

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_biased, legal_action_np)

        return logits_biased, value_np, prob

    def _apply_rule_guided_logit_bias(self, logits, feature, legal_action):
        if len(logits) != Config.ACTION_NUM or len(legal_action) != Config.ACTION_NUM:
            return logits

        dirs = np.array(Config.RULE_GUIDE_ACTION_DIRS, dtype=np.float32)
        if dirs.shape != (Config.ACTION_NUM, 2):
            return logits

        legal = np.array(legal_action, dtype=np.float32)
        if np.sum(legal) <= 0.0:
            return logits

        # Anneal guidance over training so PPO can dominate later.
        progress = min(1.0, float(self.infer_step) / max(float(Config.RULE_GUIDE_ANNEAL_STEPS), 1.0))
        anneal = pow(max(0.0, 1.0 - progress), float(Config.RULE_GUIDE_ANNEAL_POWER))

        masked_prob = self._legal_soft_max(logits, legal)
        legal_idx = np.where(legal > 0.5)[0]
        if legal_idx.size == 0:
            return logits

        sorted_prob = np.sort(masked_prob[legal_idx])
        top1 = float(sorted_prob[-1])
        top2 = float(sorted_prob[-2]) if sorted_prob.size > 1 else 0.0
        conf_margin = top1 - top2

        conf_upper = float(Config.RULE_GUIDE_CONF_UPPER)
        conf_lower = float(Config.RULE_GUIDE_CONF_LOWER)
        conf_span = max(conf_upper - conf_lower, 1e-6)
        conf_scale = np.clip((conf_upper - top1) / conf_span, 0.0, 1.0)
        margin_scale = np.clip(1.0 - conf_margin / 0.25, 0.0, 1.0)

        legal_prob = masked_prob[legal_idx]
        entropy = -float(np.sum(legal_prob * np.log(np.clip(legal_prob, 1e-9, 1.0))))
        entropy_max = math.log(float(max(2, legal_idx.size)))
        entropy_norm = entropy / max(entropy_max, 1e-6)
        entropy_scale = np.clip((entropy_norm - float(Config.RULE_GUIDE_ENTROPY_FLOOR)) / 0.5, 0.0, 1.0)

        # Parse compact state from feature vector.
        hero_dim = Config.HERO_DIM
        global_dim = Config.GLOBAL_DIM
        map_grid_dim = Config.MAP_GRID_DIM
        map_stat_dim = Config.MAP_STAT_DIM

        hero_feat = np.asarray(feature[:hero_dim], dtype=np.float32)
        treasure_start = hero_dim + global_dim + map_grid_dim + map_stat_dim
        monster_start = treasure_start + Config.TREASURE_DIM

        nearest_monster = float(np.clip(hero_feat[9], 0.0, 1.0))
        nearest_treasure = float(np.clip(hero_feat[11], 0.0, 1.0))
        danger = float(np.clip(1.0 - nearest_monster, 0.0, 1.0))

        treasure_feat = np.asarray(feature[treasure_start : treasure_start + Config.TREASURE_UNIT_DIM], dtype=np.float32)
        monster_feat = np.asarray(feature[monster_start : monster_start + Config.MONSTER_UNIT_DIM], dtype=np.float32)

        has_treasure = float(treasure_feat[0] > 0.5)
        has_monster = float(monster_feat[0] > 0.5)

        treasure_trigger = float(Config.RULE_GUIDE_TREASURE_TRIGGER)
        danger_trigger = float(Config.RULE_GUIDE_DANGER_TRIGGER)
        treasure_pull = np.clip((1.0 - nearest_treasure - treasure_trigger) / max(1.0 - treasure_trigger, 1e-6), 0.0, 1.0)
        escape_pull = np.clip((danger - danger_trigger) / max(1.0 - danger_trigger, 1e-6), 0.0, 1.0)

        if has_treasure < 0.5:
            treasure_pull = 0.0
        if has_monster < 0.5:
            escape_pull = 0.0

        behavior_strength = max(float(escape_pull), float(treasure_pull))
        guide_scale = anneal * conf_scale * margin_scale * entropy_scale * behavior_strength
        if guide_scale < float(Config.RULE_GUIDE_MIN_SCALE):
            return logits

        # Prefer moving away from nearest monster and toward nearest treasure.
        raw_bias = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        dir_norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.clip(dir_norm, 1e-6, None)

        if has_monster > 0.5:
            m_vec = np.asarray([monster_feat[1], monster_feat[2]], dtype=np.float32)
            m_norm = np.linalg.norm(m_vec)
            if m_norm > 1e-6:
                m_dir = m_vec / m_norm
                escape_score = -np.matmul(dirs, m_dir)
                raw_bias += float(Config.RULE_GUIDE_ESCAPE_WEIGHT) * escape_pull * escape_score

        if has_treasure > 0.5:
            t_vec = np.asarray([treasure_feat[1], treasure_feat[2]], dtype=np.float32)
            t_norm = np.linalg.norm(t_vec)
            if t_norm > 1e-6:
                t_dir = t_vec / t_norm
                chase_score = np.matmul(dirs, t_dir)
                risk_damp = 1.0 - 0.7 * danger
                raw_bias += float(Config.RULE_GUIDE_TREASURE_WEIGHT) * treasure_pull * max(0.0, risk_damp) * chase_score

        max_abs = float(np.max(np.abs(raw_bias)))
        if max_abs < 1e-6:
            return logits

        raw_bias = raw_bias / max_abs
        max_bias = float(Config.RULE_GUIDE_MAX_BIAS)
        bias = np.clip(raw_bias * guide_scale * max_bias, -max_bias, max_bias)
        bias = bias * legal

        legal_sum = float(np.sum(legal))
        if legal_sum > 0:
            bias_mean = float(np.sum(bias) / legal_sum)
            bias = np.where(legal > 0.5, bias - bias_mean, 0.0)

        return logits + bias

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
