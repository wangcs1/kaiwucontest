# Gorge Chase PPO 当前特征工程说明

本文基于当前代码实现整理：
- 预处理主逻辑在 code/agent_ppo/feature/preprocessor.py
- 维度配置在 code/agent_ppo/conf/conf.py

## 1. 总体目标

每一步环境交互时，Preprocessor 会把原始观测 env_obs 转成三部分输出：
- feature：给策略网络的连续特征向量
- legal_action：动作合法性掩码
- remain_info：训练日志与辅助指标（包含 reward 及分项）

核心入口函数：Preprocessor.feature_process(env_obs, last_action)

## 2. 特征总维度与分块

特征按 7 个分块拼接，维度由 Config 定义：

- HERO_DIM = 16
- GLOBAL_DIM = 9
- MAP_GRID_DIM = 25
- MAP_STAT_DIM = 4
- TREASURE_DIM = MAX_TREASURE(10) * TREASURE_UNIT_DIM(6) = 60
- MONSTER_DIM = MAX_MONSTER(2) * MONSTER_UNIT_DIM(6) = 12
- BUFF_DIM = MAX_BUFF(2) * BUFF_UNIT_DIM(5) = 10

总维度：

$$
16 + 9 + 25 + 4 + 60 + 12 + 10 = 136
$$

即 Config.FEATURE_LEN = 136。

## 3. 输入解析流程

feature_process 每步会从 env_obs["observation"] 解析：
- frame_state：动态实体（英雄、怪、宝箱、buff）
- env_info：全局计分与回合信息
- map_info：局部地图网格
- legal_action：动作合法性
- step_no：当前步数

并维护若干跨步状态（last_*），例如：
- 上一步角色位置、怪物最近距离
- 上一步闪现 CD 与闪现是否可用
- 上一步分数（step/treasure/total）
- 访问栅格计数 visit_counter

这些状态既用于特征（如 repeated_ratio），也用于奖励塑形（如距离差分）。

## 4. 通用工具函数

- _norm(v, v_max, v_min=0)：把值裁剪后线性归一化到 [0,1]
- _pick(data, keys, default)：兼容不同字段命名（蛇形/驼峰）
- _pos(entity)：读取实体坐标 (x, z)
- _dist(ax, az, bx, bz)：2D 欧氏距离

设计特点：
- 对字段缺失和类型异常有默认值兜底
- 统一归一化尺度，减小训练时数值跨度

## 5. 分块特征详解

### 5.1 Hero 特征（16 维）

顺序如下：
1. hero_x 归一化（按 MAP_SIZE=128）
2. hero_z 归一化
3. flash_cd 归一化（MAX_FLASH_CD=2000）
4. flash_ready（cd <= 1）
5. buff_remaining_time 归一化（MAX_BUFF_DURATION=300）
6. 是否有 buff（buff_remain > 1）
7. step_no / max_step
8. 距离怪物加速阶段剩余步数归一化
9. 是否已进入加速阶段
10. 最近怪距离归一化（按地图对角近似 128*1.41）
11. 次近怪距离归一化
12. 最近宝箱距离归一化
13. repeated_ratio（当前位置网格重复访问比例）
14. openness（局部可通行比例）
15. 最近怪是否过近（<16）
16. 最近宝箱是否较近（<18）

### 5.2 Global 特征（9 维）

来自 env_info：
1. step_score 归一化
2. treasure_score 归一化
3. total_score 归一化
4. treasure_count 归一化
5. buff_count 归一化
6. time_to_speedup / max_step
7. is_speedup
8. max_step 归一化
9. monster_interval 归一化

### 5.3 Map Grid 特征（25 维）

取 map_info 中心点周围 5x5 邻域：
- 可通行记 1，不可通行记 0
- 共 25 维

这部分提供局部拓扑形状，帮助策略判断是否死角、是否有通道。

### 5.4 Map Stat 特征（4 维）

从 5x5 与射线深度提取：
1. openness：5x5 中可通行占比
2. 四方向平均通道深度（归一化到 8）
3. 四方向最大通道深度（归一化到 8）
4. 四方向最小通道深度（归一化到 8）

其中 _ray_depth 从中心向上下左右探测，遇边界或障碍停止。

### 5.5 Treasure 特征（60 维）

最多编码 10 个宝箱，每个 6 维：
1. is_valid
2. 相对 x 偏移（(tx-hero_x)/MAP_SIZE，裁剪到 [-1,1]）
3. 相对 z 偏移
4. 英雄到宝箱距离归一化
5. is_in_view
6. danger：宝箱附近怪物威胁度（最近怪越近 danger 越高）

另外返回 visible_treasure_ratio（可见宝箱比例），写入 remain_info，不进入 feature 主向量。

### 5.6 Monster 特征（12 维）

最多编码 2 只怪，每只 6 维：
1. 存在标记（固定 1）
2. 相对 x 偏移
3. 相对 z 偏移
4. 与英雄距离归一化
5. speed 归一化（MAX_MONSTER_SPEED=5）
6. in_view

在编码前会按与英雄距离升序排序（_sorted_monsters），所以第 1 只始终是最近怪。

### 5.7 Buff 特征（10 维）

最多编码 2 个 buff，每个 5 维：
1. active / valid
2. 相对 x 偏移
3. 相对 z 偏移
4. 距离归一化
5. 距离反向特征（1 - norm(dist, 30)）

此外还会单独计算最近“速度 buff”距离：
- _is_speed_buff 通过 type/tag/name/description 文本匹配 speed/acceler/加速
- _nearest_speed_buff_dist 仅考虑 active 的速度 buff

## 6. 动作合法性处理

_parse_legal_action 支持两种输入格式：
- 布尔列表：按位表示每个动作是否合法
- 动作 id 列表：把出现过的 id 标记为合法

若解析后全不合法，则回退为“全部合法”（防止策略层面崩溃）。

当前动作空间：Config.ACTION_NUM = 8。

## 7. 最终输出结构

feature_process 返回：
- feature: np.ndarray(float32), shape=(136,)
- legal_action: list[int], len=8
- remain_info: dict，包含 reward、风险指标、闪现统计等

remain_info 里的 reward 来自 _reward_shaping（属于奖励塑形，不是观测特征本体）。

## 8. 与奖励塑形的耦合点（简述）

虽然你问的是特征工程，但当前实现里特征与 reward 有明显共享中间量：
- 最近怪距离、次近怪距离
- 最近宝箱/buff/速度buff距离
- openness 与 map_stats
- repeated_ratio 对应的 visit_counter
- flash 相关状态（ready/legal/used）

这使得“感知”与“学习信号”方向一致：
- 特征告诉模型当前局势
- reward 用同一批局势量去塑形行为偏好

## 9. 当前实现优点与注意点

优点：
- 结构化分块清晰，维度固定，兼容 MLP 输入
- 对字段命名差异有容错（_pick）
- 同时覆盖几何关系（相对坐标、距离）与拓扑语义（map grid/stat）
- 通过 last_* 实现时间差分信息（例如距离变化、分数增量）

注意点：
- 预处理与 reward 在同一类中，职责上较耦合
- 频繁小数组构造与 concat 会带来一定运行时开销
- MAP_SIZE*1.41 多处硬编码，后续可集中成常量提升可维护性

## 10. 一句话总结

当前特征工程是“局部地图 + 实体相对几何 + 阶段状态 + 历史差分”的固定维度拼接方案，重点服务于生存/避怪与资源获取的平衡决策。

## 11. 你的实现思路（为什么这样做）

这一版实现本质上是在做一件事：
把原始游戏状态压缩成“可学习、稳定、对决策有直接帮助”的表征，同时尽量降低训练不稳定性。

### 11.1 为什么采用“分块拼接”而不是一股脑喂原始数据

- 不同信息源语义差异很大：
	角色状态、全局进度、地图拓扑、实体关系本质不同，分块后网络更容易学到结构化规律。
- 维度固定且顺序稳定：
	对 MLP 非常友好，避免因实体数量波动导致输入结构变化。
- 便于迭代：
	可以单独增强某一块（比如 map_stat 或 buff）而不破坏整体接口。

### 11.2 为什么大量使用归一化和裁剪

- 统一数值尺度，避免某些大值（如 step、cd、距离）主导梯度。
- 裁剪到合理区间后，极端噪声不会对训练造成过强冲击。
- 对 PPO 这类方法来说，输入稳定通常直接影响策略更新稳定性。

### 11.3 为什么强调“相对位置”和“最近目标”

- 追猎场景决策核心是局部几何关系，不是绝对坐标本身。
- 相对坐标对地图平移更鲁棒，泛化更好。
- 最近怪/次近怪的排序编码，能把最关键威胁放在固定槽位，减少网络学习难度。

### 11.4 为什么加入 map_grid + map_stat 两层地图表示

- map_grid(5x5)提供细粒度障碍形状，可感知“眼前能不能走”。
- map_stat（开阔度与四向深度）提供抽象结构信号，可感知“是不是死角/长廊”。
- 两层结合后，策略既能看见局部细节，也能快速判断地形风险。

### 11.5 为什么保留历史状态（last_*）与访问计数

- 单帧观测无法表达趋势；而很多决策依赖“正在变好还是变坏”。
- 用距离差分、分数增量能直接表达“靠近资源/远离危险”的动态。
- visit_counter 抑制原地绕圈，鼓励探索和有效位移。

### 11.6 为什么把速度 buff 单独识别出来

- 在怪物加速阶段前后，机动能力价值突增。
- 单独建模 speed buff 距离，可以让策略在关键窗口更主动地抢节奏资源。
- 这属于“与任务机制强相关”的先验特征，通常比纯黑盒学习收敛更快。

### 11.7 为什么 legal_action 做了兜底

- 环境字段可能有格式差异（布尔掩码/动作列表）或偶发异常。
- 统一解析并在全 0 时回退全 1，可避免策略推理阶段出现无合法动作而崩溃。
- 这是工程稳定性优先的设计，不影响正常样本。

### 11.8 为什么特征与奖励共用关键中间量

- 特征告诉策略“当前发生了什么”，奖励告诉策略“什么是更好的方向”。
- 当两者围绕同一批状态量（危险度、开阔度、距离变化）时，学习信号一致性更高。
- 这样可以降低“特征表达方向”和“奖励驱动方向”相互打架的概率。

### 11.9 这套设计的核心取舍

- 不是追求最复杂表示，而是追求：
	有效信息密度高、可解释、可调参、收敛稳定。
- 牺牲了一部分端到端自动抽特征能力，换来更强的可控性和迭代效率。

### 11.10 可以用来对外说明的一句话

你的实现思路可以概括为：
“用结构化先验把追猎任务拆成几何风险、资源机会、阶段节奏三类信号，并通过归一化与固定维度编码，换取更稳定、更可控的 PPO 学习过程。”