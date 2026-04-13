#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（围绕决策重构）
    HERO_DIM = 16
    GLOBAL_DIM = 9
    MAP_GRID_DIM = 25
    MAP_STAT_DIM = 4

    MAX_TREASURE = 10
    TREASURE_UNIT_DIM = 6
    TREASURE_DIM = MAX_TREASURE * TREASURE_UNIT_DIM

    MAX_MONSTER = 2
    MONSTER_UNIT_DIM = 6
    MONSTER_DIM = MAX_MONSTER * MONSTER_UNIT_DIM

    MAX_BUFF = 2
    BUFF_UNIT_DIM = 5
    BUFF_DIM = MAX_BUFF * BUFF_UNIT_DIM

    FEATURES = [
        HERO_DIM,
        GLOBAL_DIM,
        MAP_GRID_DIM,
        MAP_STAT_DIM,
        TREASURE_DIM,
        MONSTER_DIM,
        BUFF_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向
    ACTION_NUM = 8

    # Rule-guided logit bias / 规则引导 logits 偏置
    RULE_GUIDE_ENABLE = True
    RULE_GUIDE_ENABLE_EVAL = False
    RULE_GUIDE_ANNEAL_STEPS = 800000
    RULE_GUIDE_ANNEAL_POWER = 1.2
    RULE_GUIDE_MAX_BIAS = 0.8
    RULE_GUIDE_DANGER_TRIGGER = 0.45
    RULE_GUIDE_TREASURE_TRIGGER = 0.18
    RULE_GUIDE_MIN_SCALE = 0.05
    RULE_GUIDE_CONF_UPPER = 0.72
    RULE_GUIDE_CONF_LOWER = 0.38
    RULE_GUIDE_ENTROPY_FLOOR = 0.45
    RULE_GUIDE_ESCAPE_WEIGHT = 1.0
    RULE_GUIDE_TREASURE_WEIGHT = 0.6
    # Action-direction mapping for 8-way movement.
    # If your environment uses a different action order, adjust this table.
    RULE_GUIDE_ACTION_DIRS = (
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (0.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
    )

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # Phase-aware reward controls / 前后期切换奖励权重
    STAGE_SWITCH_WINDOW = 120.0
    PRE_RESOURCE_WEIGHT = 1.0
    POST_RESOURCE_WEIGHT = 0.35
    PRE_SURVIVAL_WEIGHT = 0.9
    POST_SURVIVAL_WEIGHT = 1.7

    # Fine-grained shaping controls / 细粒度奖励塑形参数
    TREASURE_APPROACH_GAIN = 0.065
    CLOSE_TREASURE_RADIUS = 10.0
    CLOSE_TREASURE_BONUS = 0.06
    CLOSE_TREASURE_PULL_GAIN = 0.07
    BUFF_APPROACH_GAIN = 0.06
    BUFF_PICK_GAIN = 0.075
    SPEED_BUFF_NEAR_BONUS = 0.08

    ESCAPE_EXPAND_GAIN = 0.055
    SAFE_IDLE_PENALTY = 0.03
    FLASH_MISS_PENALTY = 0.045

    # Train / val split & curriculum / 训练验证拆分与课程学习
    TRAIN_MAPS = [1, 2, 3, 4, 5, 6, 7, 8]
    VAL_MAPS = [9, 10]
    VAL_INTERVAL = 10
