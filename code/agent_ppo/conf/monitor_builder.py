#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_metric_panel(monitor, title_cn, title_en, metric_name):
    return (
        monitor.add_panel(name=title_cn, name_en=title_en, type="line")
        .add_metric(metrics_name=metric_name, expr=f"avg({metric_name}{{}})")
        .end_panel()
    )


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    monitor.title("峡谷追猎")

    monitor.add_group(group_name="算法指标", group_name_en="algorithm")
    for cn, en, metric in [
        ("批次奖励", "cum_reward", "cum_reward"),
        ("总损失", "total_loss", "total_loss"),
        ("价值损失", "value_loss", "value_loss"),
        ("策略损失", "policy_loss", "policy_loss"),
        ("熵", "entropy_loss", "entropy_loss"),
        ("梯度范数", "grad_clip_norm", "grad_clip_norm"),
        ("裁剪比例", "clip_frac", "clip_frac"),
        ("解释方差", "explained_var", "explained_var"),
        ("优势均值", "adv_mean", "adv_mean"),
        ("回报均值", "ret_mean", "ret_mean"),
    ]:
        monitor = _add_metric_panel(monitor, cn, en, metric)
    monitor.end_group()

    train_panel_name_map = {
        "train_speedup_reached": "进入加速",
        "train_post_total_gain": "后期总分增量",
        "train_post_terminated": "后期阵亡",
        "train_terminated_rate": "阵亡率",
        "train_phase_focus_mean": "后期关注度",
        "train_pre_safety": "前期安全边际",
        "train_post_safety": "后期安全边际",
        "train_pre_encircle": "前期包夹风险",
        "train_post_encircle": "后期包夹风险",
        "train_invalid_move_rate": "无效移动率",
        "train_flash_escape_rate": "闪现脱险率",
        "train_flash_abuse_rate": "闪现滥用率",
        "train_final_trea_dist": "终局宝箱距离",
        "train_last_flash_used": "末步闪现使用",
        "train_last_flash_ready": "末步闪现可用",
        "train_last_flash_legal": "闪现合法比例",
        "train_final_visible_tre": "终局可见宝箱比",
    }

    val_panel_name_map = {
        "val_phase_focus_mean": "后期关注度",
        "val_pre_safety": "前期安全边际",
        "val_post_safety": "后期安全边际",
        "val_pre_encircle": "前期包夹风险",
        "val_post_encircle": "后期包夹风险",
        "val_invalid_move_rate": "无效移动率",
        "val_flash_escape_rate": "闪现脱险率",
        "val_flash_abuse_rate": "闪现滥用率",
        "val_final_visible_tre": "终局可见宝箱比",
    }

    monitor.add_group(group_name="训练指标", group_name_en="train")
    for metric in [
        "train_reward",
        "train_total_score",
        "train_step_score",
        "train_treasure_score",
        "train_treasures",
        "train_steps",
        "train_speedup_reached",
        "train_pre_steps",
        "train_post_steps",
        "train_pre_total_r",
        "train_post_total_r",
        "train_pre_shaped_r",
        "train_post_shaped_r",
        "train_pre_step_gain",
        "train_post_step_gain",
        "train_pre_trea_gain",
        "train_post_trea_gain",
        "train_pre_total_gain",
        "train_post_total_gain",
        "train_pre_terminal",
        "train_post_terminal",
        "train_post_terminated",
        "train_terminated_rate",
        "train_completed_rate",
        "train_abnormal_trunc",
        "train_final_danger",
        "train_final_trea_dist",
        "train_flash_count",
        "train_last_flash_used",
        "train_last_flash_ready",
        "train_last_flash_legal",
        "train_final_visible_tre",
        "train_phase_focus_mean",
        "train_pre_safety",
        "train_post_safety",
        "train_pre_encircle",
        "train_post_encircle",
        "train_invalid_move_rate",
        "train_flash_escape_rate",
        "train_flash_abuse_rate",
    ]:
        panel_name = train_panel_name_map.get(metric, metric)
        monitor = _add_metric_panel(monitor, panel_name, metric, metric)
    monitor.end_group()

    monitor.add_group(group_name="验证指标", group_name_en="val")
    for metric in [
        "val_reward",
        "val_total_score",
        "val_step_score",
        "val_treasure_score",
        "val_treasures",
        "val_steps",
        "val_speedup_reached",
        "val_pre_steps",
        "val_post_steps",
        "val_pre_total_r",
        "val_post_total_r",
        "val_pre_shaped_r",
        "val_post_shaped_r",
        "val_pre_step_gain",
        "val_post_step_gain",
        "val_pre_trea_gain",
        "val_post_trea_gain",
        "val_pre_total_gain",
        "val_post_total_gain",
        "val_pre_terminal",
        "val_post_terminal",
        "val_post_terminated",
        "val_terminated_rate",
        "val_completed_rate",
        "val_abnormal_trunc",
        "val_final_danger",
        "val_final_trea_dist",
        "val_flash_count",
        "val_last_flash_used",
        "val_last_flash_ready",
        "val_last_flash_legal",
        "val_final_visible_tre",
        "val_phase_focus_mean",
        "val_pre_safety",
        "val_post_safety",
        "val_pre_encircle",
        "val_post_encircle",
        "val_invalid_move_rate",
        "val_flash_escape_rate",
        "val_flash_abuse_rate",
    ]:
        panel_name = val_panel_name_map.get(metric, metric)
        monitor = _add_metric_panel(monitor, panel_name, metric, metric)
    config_dict = monitor.end_group().build()
    return config_dict
