#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import copy
import os
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.train_outcome = {"terminated": 0, "completed": 0, "abnormal": 0, "total": 0}
        self.val_outcome = {"terminated": 0, "completed": 0, "abnormal": 0, "total": 0}

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            train_conf = self._build_episode_conf(self.episode_cnt + 1, is_val=False)
            env_obs = self.env.reset(train_conf)

            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            ep_stats = self._init_episode_stats(self.episode_cnt, is_val=False)

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32).reshape(1)
                ep_stats["reward"] += float(reward[0])
                ep_stats["steps"] = step
                self._update_phase_stats(ep_stats, _remain_info)

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    in_speedup = int(_remain_info.get("is_speedup", 0))

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                        ep_stats["post_terminated"] = 1.0 if in_speedup else 0.0
                    else:
                        final_reward[0] = 10.0
                        result_str = "WIN"

                    if in_speedup:
                        ep_stats["post_terminal"] += float(final_reward[0])
                    else:
                        ep_stats["pre_terminal"] += float(final_reward[0])

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{ep_stats['reward']:.3f}"
                    )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    env_info = env_obs["observation"].get("env_info", {})
                    self._finalize_episode_stats(
                        ep_stats=ep_stats,
                        env_info=env_info,
                        remain_info=_remain_info,
                        terminated=terminated,
                        truncated=truncated,
                        is_val=False,
                    )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 30 and self.monitor:
                        self.monitor.put_data({os.getpid(): self._to_monitor_data(ep_stats, is_val=False)})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector

                    if self.episode_cnt % Config.VAL_INTERVAL == 0:
                        self._run_validation()
                    break

                obs_data = _obs_data
                remain_info = _remain_info

    def _build_episode_conf(self, episode_cnt, is_val=False):
        conf = copy.deepcopy(self.usr_conf)
        env_conf = conf.get("env_conf", {})

        if episode_cnt <= 150:
            stage = {
                "treasure_count": (9, 10),
                "buff_count": (2, 2),
                "monster_interval": (220, 300),
                "monster_speedup": (360, 460),
                "max_step": 2000,
            }
        elif episode_cnt <= 500:
            stage = {
                "treasure_count": (8, 10),
                "buff_count": (1, 2),
                "monster_interval": (160, 280),
                "monster_speedup": (240, 420),
                "max_step": 2000,
            }
        elif episode_cnt <= 900:
            stage = {
                "treasure_count": (7, 10),
                "buff_count": (1, 2),
                "monster_interval": (120, 220),
                "monster_speedup": (180, 320),
                "max_step": 2000,
            }
        else:
            stage = {
                "treasure_count": (6, 10),
                "buff_count": (0, 2),
                "monster_interval": (120, 320),
                "monster_speedup": (140, 420),
                "max_step": 2000,
            }

        env_conf["map"] = Config.VAL_MAPS if is_val else Config.TRAIN_MAPS
        env_conf["map_random"] = True
        env_conf["treasure_count"] = int(np.random.randint(stage["treasure_count"][0], stage["treasure_count"][1] + 1))
        env_conf["buff_count"] = int(np.random.randint(stage["buff_count"][0], stage["buff_count"][1] + 1))
        env_conf["monster_interval"] = int(
            np.random.randint(stage["monster_interval"][0], stage["monster_interval"][1] + 1)
        )
        env_conf["monster_speedup"] = int(
            np.random.randint(stage["monster_speedup"][0], stage["monster_speedup"][1] + 1)
        )
        env_conf["max_step"] = int(stage["max_step"])
        conf["env_conf"] = env_conf
        return conf

    def _init_episode_stats(self, episode_cnt, is_val):
        return {
            "episode": float(episode_cnt),
            "reward": 0.0,
            "steps": 0.0,
            "pre_steps": 0.0,
            "post_steps": 0.0,
            "pre_total_r": 0.0,
            "post_total_r": 0.0,
            "pre_shaped_r": 0.0,
            "post_shaped_r": 0.0,
            "pre_step_gain": 0.0,
            "post_step_gain": 0.0,
            "pre_trea_gain": 0.0,
            "post_trea_gain": 0.0,
            "pre_total_gain": 0.0,
            "post_total_gain": 0.0,
            "pre_terminal": 0.0,
            "post_terminal": 0.0,
            "post_terminated": 0.0,
            "total_score": 0.0,
            "step_score": 0.0,
            "treasure_score": 0.0,
            "treasures": 0.0,
            "speedup_reached": 0.0,
            "final_danger": 0.0,
            "final_trea_dist": 0.0,
            "flash_count": 0.0,
            "last_flash_used": 0.0,
            "last_flash_ready": 0.0,
            "last_flash_legal": 0.0,
            "final_visible_tre": 0.0,
            "phase_focus_sum": 0.0,
            "pre_safety_sum": 0.0,
            "post_safety_sum": 0.0,
            "pre_encircle_sum": 0.0,
            "post_encircle_sum": 0.0,
            "invalid_move_cnt": 0.0,
            "flash_escape_cnt": 0.0,
            "flash_abuse_cnt": 0.0,
        }

    def _update_phase_stats(self, ep_stats, remain_info):
        is_speedup = int(remain_info.get("is_speedup", 0))
        shaped = float(remain_info.get("shaped_reward", 0.0))
        step_gain = float(remain_info.get("step_gain", 0.0))
        trea_gain = float(remain_info.get("trea_gain", 0.0))
        total_gain = float(remain_info.get("total_gain", 0.0))

        ep_stats["speedup_reached"] = max(ep_stats["speedup_reached"], float(is_speedup))
        ep_stats["phase_focus_sum"] += float(remain_info.get("phase_focus", 0.0))
        ep_stats["invalid_move_cnt"] += float(remain_info.get("invalid_move", 0.0))
        ep_stats["flash_escape_cnt"] += float(remain_info.get("flash_escape", 0.0))
        ep_stats["flash_abuse_cnt"] += float(remain_info.get("flash_abuse", 0.0))
        if is_speedup:
            ep_stats["post_steps"] += 1.0
            ep_stats["post_total_r"] += shaped
            ep_stats["post_shaped_r"] += shaped
            ep_stats["post_step_gain"] += step_gain
            ep_stats["post_trea_gain"] += trea_gain
            ep_stats["post_total_gain"] += total_gain
            ep_stats["post_safety_sum"] += float(remain_info.get("safety_margin", 0.0))
            ep_stats["post_encircle_sum"] += float(remain_info.get("encircle_risk", 0.0))
        else:
            ep_stats["pre_steps"] += 1.0
            ep_stats["pre_total_r"] += shaped
            ep_stats["pre_shaped_r"] += shaped
            ep_stats["pre_step_gain"] += step_gain
            ep_stats["pre_trea_gain"] += trea_gain
            ep_stats["pre_total_gain"] += total_gain
            ep_stats["pre_safety_sum"] += float(remain_info.get("safety_margin", 0.0))
            ep_stats["pre_encircle_sum"] += float(remain_info.get("encircle_risk", 0.0))

    def _finalize_episode_stats(self, ep_stats, env_info, remain_info, terminated, truncated, is_val=False):
        ep_stats["reward"] += ep_stats["pre_terminal"] + ep_stats["post_terminal"]
        ep_stats["total_score"] = float(env_info.get("total_score", env_info.get("totalScore", 0.0)))
        ep_stats["step_score"] = float(env_info.get("step_score", env_info.get("stepScore", 0.0)))
        ep_stats["treasure_score"] = float(env_info.get("treasure_score", env_info.get("treasureScore", 0.0)))
        ep_stats["treasures"] = float(env_info.get("treasure_count", env_info.get("treasures", 0.0)))
        ep_stats["final_danger"] = float(remain_info.get("danger", 0.0))
        ep_stats["final_trea_dist"] = float(remain_info.get("nearest_treasure_dist_norm", 0.0))
        ep_stats["flash_count"] = float(remain_info.get("flash_count", 0.0))
        ep_stats["last_flash_used"] = float(remain_info.get("last_flash_used", 0.0))
        ep_stats["last_flash_ready"] = float(remain_info.get("last_flash_ready", 0.0))
        ep_stats["last_flash_legal"] = float(remain_info.get("last_flash_legal", 0.0))
        ep_stats["final_visible_tre"] = float(remain_info.get("visible_treasure_ratio", 0.0))

        outcome = self.val_outcome if is_val else self.train_outcome
        outcome["total"] += 1
        if terminated:
            outcome["terminated"] += 1
        elif truncated:
            outcome["abnormal"] += 1
        else:
            outcome["completed"] += 1

        total = max(outcome["total"], 1)
        ep_stats["terminated_rate"] = float(outcome["terminated"]) / total
        ep_stats["completed_rate"] = float(outcome["completed"]) / total
        ep_stats["abnormal_trunc"] = float(outcome["abnormal"]) / total

        total_steps = max(ep_stats["steps"], 1.0)
        ep_stats["phase_focus_mean"] = ep_stats["phase_focus_sum"] / total_steps
        ep_stats["invalid_move_rate"] = ep_stats["invalid_move_cnt"] / total_steps
        ep_stats["flash_escape_rate"] = ep_stats["flash_escape_cnt"] / total_steps
        ep_stats["flash_abuse_rate"] = ep_stats["flash_abuse_cnt"] / total_steps
        ep_stats["pre_safety"] = ep_stats["pre_safety_sum"] / max(ep_stats["pre_steps"], 1.0)
        ep_stats["post_safety"] = ep_stats["post_safety_sum"] / max(ep_stats["post_steps"], 1.0)
        ep_stats["pre_encircle"] = ep_stats["pre_encircle_sum"] / max(ep_stats["pre_steps"], 1.0)
        ep_stats["post_encircle"] = ep_stats["post_encircle_sum"] / max(ep_stats["post_steps"], 1.0)

    def _to_monitor_data(self, ep_stats, is_val=False):
        prefix = "val_" if is_val else "train_"
        data = {
            f"{prefix}reward": round(ep_stats["reward"], 4),
            f"{prefix}total_score": round(ep_stats["total_score"], 4),
            f"{prefix}step_score": round(ep_stats["step_score"], 4),
            f"{prefix}treasure_score": round(ep_stats["treasure_score"], 4),
            f"{prefix}treasures": round(ep_stats["treasures"], 4),
            f"{prefix}steps": round(ep_stats["steps"], 4),
            f"{prefix}speedup_reached": round(ep_stats["speedup_reached"], 4),
            f"{prefix}pre_steps": round(ep_stats["pre_steps"], 4),
            f"{prefix}post_steps": round(ep_stats["post_steps"], 4),
            f"{prefix}pre_total_r": round(ep_stats["pre_total_r"], 4),
            f"{prefix}post_total_r": round(ep_stats["post_total_r"], 4),
            f"{prefix}pre_shaped_r": round(ep_stats["pre_shaped_r"], 4),
            f"{prefix}post_shaped_r": round(ep_stats["post_shaped_r"], 4),
            f"{prefix}pre_step_gain": round(ep_stats["pre_step_gain"], 4),
            f"{prefix}post_step_gain": round(ep_stats["post_step_gain"], 4),
            f"{prefix}pre_trea_gain": round(ep_stats["pre_trea_gain"], 4),
            f"{prefix}post_trea_gain": round(ep_stats["post_trea_gain"], 4),
            f"{prefix}pre_total_gain": round(ep_stats["pre_total_gain"], 4),
            f"{prefix}post_total_gain": round(ep_stats["post_total_gain"], 4),
            f"{prefix}pre_terminal": round(ep_stats["pre_terminal"], 4),
            f"{prefix}post_terminal": round(ep_stats["post_terminal"], 4),
            f"{prefix}post_terminated": round(ep_stats["post_terminated"], 4),
            f"{prefix}terminated_rate": round(ep_stats["terminated_rate"], 4),
            f"{prefix}completed_rate": round(ep_stats["completed_rate"], 4),
            f"{prefix}abnormal_trunc": round(ep_stats["abnormal_trunc"], 4),
            f"{prefix}final_danger": round(ep_stats["final_danger"], 4),
            f"{prefix}final_trea_dist": round(ep_stats["final_trea_dist"], 4),
            f"{prefix}flash_count": round(ep_stats["flash_count"], 4),
            f"{prefix}last_flash_used": round(ep_stats["last_flash_used"], 4),
            f"{prefix}last_flash_ready": round(ep_stats["last_flash_ready"], 4),
            f"{prefix}last_flash_legal": round(ep_stats["last_flash_legal"], 4),
            f"{prefix}final_visible_tre": round(ep_stats["final_visible_tre"], 4),
            f"{prefix}phase_focus_mean": round(ep_stats["phase_focus_mean"], 4),
            f"{prefix}pre_safety": round(ep_stats["pre_safety"], 4),
            f"{prefix}post_safety": round(ep_stats["post_safety"], 4),
            f"{prefix}pre_encircle": round(ep_stats["pre_encircle"], 4),
            f"{prefix}post_encircle": round(ep_stats["post_encircle"], 4),
            f"{prefix}invalid_move_rate": round(ep_stats["invalid_move_rate"], 4),
            f"{prefix}flash_escape_rate": round(ep_stats["flash_escape_rate"], 4),
            f"{prefix}flash_abuse_rate": round(ep_stats["flash_abuse_rate"], 4),
            "episode_cnt": round(ep_stats["episode"], 4),
        }
        return data

    def _run_validation(self):
        val_conf = self._build_episode_conf(self.episode_cnt, is_val=True)
        env_obs = self.env.reset(val_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return

        self.agent.reset(env_obs)
        self.agent.load_model(id="latest")
        obs_data, remain_info = self.agent.observation_process(env_obs)

        done = False
        step = 0
        ep_stats = self._init_episode_stats(self.episode_cnt, is_val=True)

        while not done:
            act = self.agent.exploit(env_obs)
            env_reward, env_obs = self.env.step(act)

            if handle_disaster_recovery(env_obs, self.logger):
                return

            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            done = terminated or truncated
            step += 1

            obs_data, remain_info = self.agent.observation_process(env_obs)
            reward = float(np.array(remain_info.get("reward", [0.0]), dtype=np.float32).reshape(1)[0])
            ep_stats["reward"] += reward
            ep_stats["steps"] = step
            self._update_phase_stats(ep_stats, remain_info)

            if done:
                final_reward = -10.0 if terminated else 10.0
                if int(remain_info.get("is_speedup", 0)):
                    ep_stats["post_terminal"] += final_reward
                    if terminated:
                        ep_stats["post_terminated"] = 1.0
                else:
                    ep_stats["pre_terminal"] += final_reward

                env_info = env_obs["observation"].get("env_info", {})
                self._finalize_episode_stats(
                    ep_stats=ep_stats,
                    env_info=env_info,
                    remain_info=remain_info,
                    terminated=terminated,
                    truncated=truncated,
                    is_val=True,
                )

                if self.monitor:
                    self.monitor.put_data({os.getpid(): self._to_monitor_data(ep_stats, is_val=True)})

                self.logger.info(
                    f"[VAL] episode:{self.episode_cnt} reward:{ep_stats['reward']:.3f} "
                    f"score:{ep_stats['total_score']:.1f} post_terminated:{ep_stats['post_terminated']}"
                )
