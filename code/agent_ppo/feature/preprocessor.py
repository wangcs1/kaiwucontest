#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import math
import numpy as np

from agent_ppo.conf.conf import Config

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 300.0


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    span = v_max - v_min
    return (v - v_min) / span if span > 1e-6 else 0.0


def _pick(data, keys, default=0.0):
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data:
            return data[key]
    return default


def _pos(entity):
    p = entity.get("pos", {}) if isinstance(entity, dict) else {}
    return float(p.get("x", 0.0)), float(p.get("z", 0.0))


def _dist(ax, az, bx, bz):
    return math.sqrt((ax - bx) * (ax - bx) + (az - bz) * (az - bz))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 2000
        self.monster_speedup_step = 9999
        self.last_pos = (0.0, 0.0)
        self.last_flash_cd = 0.0
        self.last_nearest_monster_dist = MAP_SIZE * 1.41
        self.last_nearest_treasure_dist = MAP_SIZE * 1.41
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_total_score = 0.0
        self.last_buff_count = 0.0
        self.visit_counter = {}
        self.flash_count = 0
        self.last_flash_used = 0.0
        self.last_flash_ready = 0.0
        self.last_flash_legal = 1.0

    def feature_process(self, env_obs, last_action):
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])
        legal_act_raw = observation.get("legal_action", [])

        self.step_no = int(observation.get("step_no", self.step_no + 1))
        self.max_step = int(_pick(env_info, ["max_step", "maxStep"], self.max_step))
        self.monster_speedup_step = int(
            _pick(env_info, ["monster_speedup", "monsterSpeedup", "monster_speedup_step"], self.monster_speedup_step)
        )

        hero = frame_state.get("heroes", {})
        hero_x, hero_z = _pos(hero)
        flash_cd = float(_pick(hero, ["flash_cooldown", "talent_cooldown", "flashCooldown"], 0.0))
        buff_remain = float(_pick(hero, ["buff_remaining_time", "buffRemainingTime"], 0.0))
        flash_ready = 1.0 if flash_cd <= 1.0 else 0.0

        legal_action = self._parse_legal_action(legal_act_raw)
        self.last_flash_legal = float(legal_action[-1])

        is_speedup = 1.0 if self.step_no >= self.monster_speedup_step else 0.0
        time_to_speedup = max(0.0, float(self.monster_speedup_step - self.step_no))

        monsters = self._sorted_monsters(frame_state.get("monsters", []), hero_x, hero_z)
        treasures = frame_state.get("treasures", [])
        buffs = frame_state.get("buffs", [])

        nearest_monster_dist = monsters[0]["dist"] if monsters else MAP_SIZE * 1.41
        second_monster_dist = monsters[1]["dist"] if len(monsters) > 1 else MAP_SIZE * 1.41
        nearest_treasure_dist = self._nearest_entity_dist(treasures, hero_x, hero_z)

        map_grid_feat, map_stats_feat = self._extract_map_feature(map_info)
        openness = float(map_stats_feat[0])

        repeated_ratio = self._update_visit_ratio(hero_x, hero_z)

        hero_feat = np.array(
            [
                _norm(hero_x, MAP_SIZE),
                _norm(hero_z, MAP_SIZE),
                _norm(flash_cd, MAX_FLASH_CD),
                flash_ready,
                _norm(buff_remain, MAX_BUFF_DURATION),
                1.0 if buff_remain > 1.0 else 0.0,
                _norm(self.step_no, max(self.max_step, 1)),
                _norm(time_to_speedup, max(self.monster_speedup_step, 1.0)),
                is_speedup,
                _norm(nearest_monster_dist, MAP_SIZE * 1.41),
                _norm(second_monster_dist, MAP_SIZE * 1.41),
                _norm(nearest_treasure_dist, MAP_SIZE * 1.41),
                repeated_ratio,
                openness,
                1.0 if nearest_monster_dist < 16.0 else 0.0,
                1.0 if nearest_treasure_dist < 18.0 else 0.0,
            ],
            dtype=np.float32,
        )

        global_feat = np.array(
            [
                _norm(_pick(env_info, ["step_score", "stepScore"], 0.0), 2000.0),
                _norm(_pick(env_info, ["treasure_score", "treasureScore"], 200.0), 200.0),
                _norm(_pick(env_info, ["total_score", "totalScore"], 2200.0), 2200.0),
                _norm(_pick(env_info, ["treasure_count", "treasures", "treasureCount"], 10.0), 10.0),
                _norm(_pick(env_info, ["buff_count", "buffCount"], 2.0), 2.0),
                _norm(time_to_speedup, max(float(self.max_step), 1.0)),
                is_speedup,
                _norm(self.max_step, 2000.0),
                _norm(_pick(env_info, ["monster_interval", "monsterInterval"], 300.0), 2000.0),
            ],
            dtype=np.float32,
        )

        treasure_feat, visible_treasure_ratio = self._encode_treasures(treasures, monsters, hero_x, hero_z)
        monster_feat = self._encode_monsters(monsters, hero_x, hero_z)
        buff_feat = self._encode_buffs(buffs, hero_x, hero_z)

        feature = np.concatenate(
            [
                hero_feat,
                global_feat,
                map_grid_feat,
                map_stats_feat,
                treasure_feat,
                monster_feat,
                buff_feat,
            ]
        )

        reward, reward_items = self._reward_shaping(
            env_info=env_info,
            hero_pos=(hero_x, hero_z),
            monsters=monsters,
            nearest_monster_dist=nearest_monster_dist,
            second_monster_dist=second_monster_dist,
            nearest_treasure_dist=nearest_treasure_dist,
            time_to_speedup=time_to_speedup,
            is_speedup=is_speedup,
            openness=openness,
            map_stats_feat=map_stats_feat,
            flash_cd=flash_cd,
            flash_ready=flash_ready,
            last_action=last_action,
        )

        self.last_pos = (hero_x, hero_z)
        self.last_nearest_monster_dist = nearest_monster_dist
        self.last_nearest_treasure_dist = nearest_treasure_dist
        self.last_flash_ready = flash_ready
        self.last_flash_cd = flash_cd

        remain_info = {
            "reward": [reward],
            "reward_items": reward_items,
            "is_speedup": int(is_speedup),
            "danger": float(1.0 - _norm(nearest_monster_dist, 32.0)),
            "nearest_treasure_dist_norm": float(_norm(nearest_treasure_dist, MAP_SIZE * 1.41)),
            "visible_treasure_ratio": float(visible_treasure_ratio),
            "flash_count": int(self.flash_count),
            "last_flash_used": float(self.last_flash_used),
            "last_flash_ready": float(self.last_flash_ready),
            "last_flash_legal": float(self.last_flash_legal),
            "step_gain": float(reward_items["step_gain"]),
            "trea_gain": float(reward_items["treasure_gain"]),
            "total_gain": float(reward_items["total_gain"]),
            "shaped_reward": float(reward),
            "phase_focus": float(reward_items["survival_focus"]),
            "safety_margin": float(reward_items["safety_margin"]),
            "encircle_risk": float(reward_items["encircle_risk"]),
            "invalid_move": float(reward_items["invalid_move"]),
            "flash_escape": float(reward_items["flash_escape"]),
            "flash_abuse": float(reward_items["flash_abuse"]),
        }
        return feature, legal_action, remain_info

    def _parse_legal_action(self, legal_act_raw):
        legal_action = [1] * Config.ACTION_NUM
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for i in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                    legal_action[i] = int(legal_act_raw[i])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < Config.ACTION_NUM}
                legal_action = [1 if i in valid_set else 0 for i in range(Config.ACTION_NUM)]
        if sum(legal_action) <= 0:
            legal_action = [1] * Config.ACTION_NUM
        return legal_action

    def _sorted_monsters(self, monsters, hero_x, hero_z):
        encoded = []
        for m in monsters if isinstance(monsters, list) else []:
            mx, mz = _pos(m)
            d = _dist(hero_x, hero_z, mx, mz)
            encoded.append(
                {
                    "raw": m,
                    "x": mx,
                    "z": mz,
                    "dist": d,
                    "in_view": float(_pick(m, ["is_in_view", "isInView"], 1.0)),
                    "speed": float(_pick(m, ["speed"], 1.0)),
                }
            )
        encoded.sort(key=lambda x: x["dist"])
        return encoded

    def _nearest_entity_dist(self, entities, hero_x, hero_z):
        best = MAP_SIZE * 1.41
        for e in entities if isinstance(entities, list) else []:
            ex, ez = _pos(e)
            best = min(best, _dist(hero_x, hero_z, ex, ez))
        return best

    def _extract_map_feature(self, map_info):
        grid = np.zeros(Config.MAP_GRID_DIM, dtype=np.float32)
        stats = np.zeros(Config.MAP_STAT_DIM, dtype=np.float32)
        if not isinstance(map_info, list) or len(map_info) == 0 or not isinstance(map_info[0], list):
            return grid, stats

        h = len(map_info)
        w = len(map_info[0])
        cr = h // 2
        cc = w // 2
        idx = 0
        walkable_cnt = 0
        for r in range(cr - 2, cr + 3):
            for c in range(cc - 2, cc + 3):
                v = 0.0
                if 0 <= r < h and 0 <= c < w:
                    v = 1.0 if map_info[r][c] != 0 else 0.0
                grid[idx] = v
                walkable_cnt += int(v > 0.5)
                idx += 1

        # 4-dir corridor depth
        dir_depth = [
            self._ray_depth(map_info, cr, cc, -1, 0),
            self._ray_depth(map_info, cr, cc, 1, 0),
            self._ray_depth(map_info, cr, cc, 0, -1),
            self._ray_depth(map_info, cr, cc, 0, 1),
        ]
        avg_depth = float(sum(dir_depth)) / 4.0
        stats[0] = float(walkable_cnt) / 25.0
        stats[1] = _norm(avg_depth, 8.0)
        stats[2] = _norm(max(dir_depth), 8.0)
        stats[3] = _norm(min(dir_depth), 8.0)
        return grid, stats

    def _ray_depth(self, map_info, r0, c0, dr, dc, max_len=8):
        h = len(map_info)
        w = len(map_info[0])
        depth = 0
        r, c = r0, c0
        for _ in range(max_len):
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            if map_info[r][c] == 0:
                break
            depth += 1
        return depth

    def _encode_treasures(self, treasures, monsters, hero_x, hero_z):
        feat = np.zeros(Config.TREASURE_DIM, dtype=np.float32)
        treasures = treasures if isinstance(treasures, list) else []
        if len(treasures) == 0:
            return feat, 0.0

        visible_cnt = 0
        for i in range(min(Config.MAX_TREASURE, len(treasures))):
            t = treasures[i]
            tx, tz = _pos(t)
            is_valid = float(_pick(t, ["is_valid", "valid", "alive"], 1.0))
            is_view = float(_pick(t, ["is_in_view", "isInView", "in_view"], 1.0))
            if is_view > 0.5:
                visible_cnt += 1
            dist = _dist(hero_x, hero_z, tx, tz)
            min_md = MAP_SIZE * 1.41
            for m in monsters:
                min_md = min(min_md, _dist(tx, tz, m["x"], m["z"]))
            danger = 1.0 - _norm(min_md, 28.0)
            base = i * Config.TREASURE_UNIT_DIM
            feat[base : base + Config.TREASURE_UNIT_DIM] = np.array(
                [
                    is_valid,
                    np.clip((tx - hero_x) / MAP_SIZE, -1.0, 1.0),
                    np.clip((tz - hero_z) / MAP_SIZE, -1.0, 1.0),
                    _norm(dist, MAP_SIZE * 1.41),
                    is_view,
                    float(np.clip(danger, 0.0, 1.0)),
                ],
                dtype=np.float32,
            )

        return feat, float(visible_cnt) / float(max(len(treasures), 1))

    def _encode_monsters(self, monsters, hero_x, hero_z):
        feat = np.zeros(Config.MONSTER_DIM, dtype=np.float32)
        for i in range(min(Config.MAX_MONSTER, len(monsters))):
            m = monsters[i]
            base = i * Config.MONSTER_UNIT_DIM
            feat[base : base + Config.MONSTER_UNIT_DIM] = np.array(
                [
                    1.0,
                    np.clip((m["x"] - hero_x) / MAP_SIZE, -1.0, 1.0),
                    np.clip((m["z"] - hero_z) / MAP_SIZE, -1.0, 1.0),
                    _norm(m["dist"], MAP_SIZE * 1.41),
                    _norm(m["speed"], MAX_MONSTER_SPEED),
                    m["in_view"],
                ],
                dtype=np.float32,
            )
        return feat

    def _encode_buffs(self, buffs, hero_x, hero_z):
        feat = np.zeros(Config.BUFF_DIM, dtype=np.float32)
        buffs = buffs if isinstance(buffs, list) else []
        for i in range(min(Config.MAX_BUFF, len(buffs))):
            b = buffs[i]
            bx, bz = _pos(b)
            active = float(_pick(b, ["is_valid", "valid", "is_active", "active"], 1.0))
            d = _dist(hero_x, hero_z, bx, bz)
            base = i * Config.BUFF_UNIT_DIM
            feat[base : base + Config.BUFF_UNIT_DIM] = np.array(
                [
                    active,
                    np.clip((bx - hero_x) / MAP_SIZE, -1.0, 1.0),
                    np.clip((bz - hero_z) / MAP_SIZE, -1.0, 1.0),
                    _norm(d, MAP_SIZE * 1.41),
                    1.0 - _norm(d, 30.0),
                ],
                dtype=np.float32,
            )
        return feat

    def _update_visit_ratio(self, hero_x, hero_z):
        gx = int(hero_x // 4.0)
        gz = int(hero_z // 4.0)
        key = (gx, gz)
        cnt = self.visit_counter.get(key, 0) + 1
        self.visit_counter[key] = cnt
        return float(min(cnt, 10)) / 10.0

    def _reward_shaping(
        self,
        env_info,
        hero_pos,
        monsters,
        nearest_monster_dist,
        second_monster_dist,
        nearest_treasure_dist,
        time_to_speedup,
        is_speedup,
        openness,
        map_stats_feat,
        flash_cd,
        flash_ready,
        last_action,
    ):
        step_score = float(_pick(env_info, ["step_score", "stepScore"], 0.0))
        treasure_score = float(_pick(env_info, ["treasure_score", "treasureScore"], 0.0))
        total_score = float(_pick(env_info, ["total_score", "totalScore"], 0.0))
        buff_count = float(_pick(env_info, ["buff_count", "buffCount"], 0.0))

        step_gain = max(0.0, step_score - self.last_step_score)
        treasure_gain = max(0.0, treasure_score - self.last_treasure_score)
        total_gain = max(0.0, total_score - self.last_total_score)
        buff_gain = max(0.0, buff_count - self.last_buff_count)

        switch_window = max(Config.STAGE_SWITCH_WINDOW, 1.0)
        survival_focus = 1.0 if is_speedup > 0.5 else np.clip((switch_window - time_to_speedup) / switch_window, 0.0, 1.0)
        resource_weight = Config.PRE_RESOURCE_WEIGHT * (1.0 - survival_focus) + Config.POST_RESOURCE_WEIGHT * survival_focus
        survival_weight = Config.PRE_SURVIVAL_WEIGHT * (1.0 - survival_focus) + Config.POST_SURVIVAL_WEIGHT * survival_focus
        if is_speedup > 0.5:
            resource_weight = Config.POST_RESOURCE_WEIGHT
            survival_weight = Config.POST_SURVIVAL_WEIGHT

        danger = 1.0 - _norm(nearest_monster_dist, 30.0)
        safety_margin = 0.6 * _norm(nearest_monster_dist, 28.0) + 0.4 * _norm(second_monster_dist, 32.0)

        monster_dist_delta = nearest_monster_dist - self.last_nearest_monster_dist
        monster_dist_reward = survival_weight * 0.07 * np.clip(monster_dist_delta / 4.0, -1.0, 1.0)

        treasure_dist_delta = self.last_nearest_treasure_dist - nearest_treasure_dist
        treasure_approach_reward = resource_weight * 0.04 * np.clip(treasure_dist_delta / 3.0, -1.0, 1.0) * (1.0 - 0.7 * danger)

        survive_reward = 0.008 + 0.012 * survival_focus + (0.01 if is_speedup > 0.5 else 0.0)

        speedup_buffer_reward = 0.0
        if is_speedup < 0.5 and time_to_speedup <= 140.0:
            safe_space = 0.5 * _norm(nearest_monster_dist, 24.0) + 0.5 * _norm(second_monster_dist, 28.0)
            speedup_buffer_reward = (0.03 + 0.03 * survival_focus) * safe_space

        corridor_reward = (0.02 + 0.03 * survival_focus) * (
            float(map_stats_feat[1]) + float(map_stats_feat[2]) - 1.0 + 0.6 * max(0.0, openness - 0.45)
        )
        deadend_penalty = -(0.03 + 0.03 * survival_focus) * max(0.0, 0.22 - openness)
        deadend_penalty += -(0.02 + 0.02 * survival_focus) * max(0.0, 0.2 - float(map_stats_feat[3]))

        danger_penalty = -(0.07 + 0.08 * survival_focus) * max(0.0, 0.42 - _norm(nearest_monster_dist, 26.0))
        second_monster_penalty = -(0.04 + 0.06 * survival_focus) * max(0.0, 0.45 - _norm(second_monster_dist, 30.0))

        encircle_risk = self._encircle_risk(monsters, hero_pos)
        encircle_penalty = -(0.06 + 0.08 * survival_focus) * encircle_risk

        move_dist = _dist(hero_pos[0], hero_pos[1], self.last_pos[0], self.last_pos[1])
        invalid_move = 1.0 if (last_action >= 0 and move_dist < 0.25) else 0.0
        invalid_move_penalty = -(0.015 + 0.045 * danger) * invalid_move

        gx = int(hero_pos[0] // 4.0)
        gz = int(hero_pos[1] // 4.0)
        repeat_cnt = self.visit_counter.get((gx, gz), 1)
        repeat_penalty = -(0.008 + 0.004 * survival_focus) * min(6.0, max(0.0, repeat_cnt - 2.0))

        flash_used = 1.0 if flash_cd > self.last_flash_cd + 5.0 and self.last_flash_ready > 0.5 else 0.0
        if flash_used > 0.5:
            self.flash_count += 1
        flash_escape_reward = 0.0
        flash_abuse_penalty = 0.0
        if flash_used > 0.5:
            danger_drop = (1.0 - _norm(self.last_nearest_monster_dist, 30.0)) - danger
            if nearest_monster_dist - self.last_nearest_monster_dist > 3.0 or danger_drop > 0.12 or openness > 0.45:
                flash_escape_reward = 0.1 + 0.08 * survival_focus
            else:
                flash_abuse_penalty = -(0.06 + 0.05 * survival_focus)
        self.last_flash_used = flash_used

        resource_reward = resource_weight * (0.018 * step_gain + 0.1 * treasure_gain + 0.055 * buff_gain)

        survival_reward = (
            survive_reward
            + monster_dist_reward
            + speedup_buffer_reward
            + corridor_reward
            + deadend_penalty
            + danger_penalty
            + second_monster_penalty
            + encircle_penalty
        )

        reward = (
            resource_reward
            + treasure_approach_reward
            + survival_reward
            + invalid_move_penalty
            + repeat_penalty
            + flash_escape_reward
            + flash_abuse_penalty
        )
        reward = float(np.clip(reward, -2.0, 2.0))

        self.last_step_score = step_score
        self.last_treasure_score = treasure_score
        self.last_total_score = total_score
        self.last_buff_count = buff_count

        reward_items = {
            "step_gain": step_gain,
            "treasure_gain": treasure_gain,
            "total_gain": total_gain,
            "flash_used": flash_used,
            "survival_focus": survival_focus,
            "safety_margin": safety_margin,
            "encircle_risk": encircle_risk,
            "invalid_move": invalid_move,
            "flash_escape": 1.0 if flash_escape_reward > 0.0 else 0.0,
            "flash_abuse": 1.0 if flash_abuse_penalty < 0.0 else 0.0,
        }
        return reward, reward_items

    def _encircle_risk(self, monsters, hero_pos):
        if not isinstance(monsters, list) or len(monsters) < 2:
            return 0.0
        m1 = monsters[0]
        m2 = monsters[1]
        v1x, v1z = m1["x"] - hero_pos[0], m1["z"] - hero_pos[1]
        v2x, v2z = m2["x"] - hero_pos[0], m2["z"] - hero_pos[1]
        n1 = max(1e-6, math.sqrt(v1x * v1x + v1z * v1z))
        n2 = max(1e-6, math.sqrt(v2x * v2x + v2z * v2z))
        cos_theta = np.clip((v1x * v2x + v1z * v2z) / (n1 * n2), -1.0, 1.0)
        angle_risk = np.clip((-cos_theta - 0.1) / 0.9, 0.0, 1.0)
        dist_risk = 0.5 * (1.0 - _norm(m1["dist"], 32.0)) + 0.5 * (1.0 - _norm(m2["dist"], 34.0))
        return float(np.clip(angle_risk * dist_risk, 0.0, 1.0))
