# Kuavo S46：AME 任务 与 Leju Attention 任务的配置对比

## 0. 对比对象

| | AME (本文仓库) | Attention (Leju-IsaacLab) |
| --- | --- | --- |
| 环境配置文件 | `ame_locomotion/tasks/.../s46/velocity_env_cfg_s46.py`（继承 G1 的 `velocity_env_cfg_29dof.py`） | `ext_template/tasks/locomotion/velocity/config/s42/attention_env_cfg.py` |
| 网络/训练配置 | `agents/kuavo_s46_ame_rsl_rl_ppo_cfg.py` → `G1AMEPPORunnerCfg`，类名 `ActorCriticEncoder` | `config/s42/agents/rsl_rl_ppo_cfg.py` → `KuavoAttentionRoughPPORunnerCfg`，类名 `EncActorCritic` |
| 机器人 | Kuavo S46（26 关节） | Kuavo S46（26 关节） |
| 训练场景 | `KuavoS46RoughEnvCfg`（num_envs=2048，ROUGH 地形） | `KuavoAttentionRoughEnvCfg`（num_envs=4096，ATTEN 地形） |
| 命令类型 | `UniformVelocityCommandCfg` | `UniformSteppingVelocityCommandCfg` |

> 说明：AME 的 S46 配置只做“名称映射”，reward/event 的**函数、参数、权重全部继承 G1 基础配置**；本文以 S46 实际生效值（`G1RoughEnvCfg.__post_init__` 默认分支，即 `FINETUNE=False`）为准。Attention 配置为 s42 独立完整定义。

---

## 1. Reward 对比

### 1.1 逐项对比表

| Reward 项 | AME S46 | Attention | 备注 |
| --- | --- | --- | --- |
| `track_lin_vel_xy_exp` | **2.0**，std=0.25 | **5.0**，std=√0.25=0.5 | 同为 `track_lin_vel_xy_yaw_frame_exp`，权重差 2.5 倍 |
| `track_ang_vel_z_exp` | **3.0**，std=0.25 | **3.0**，std=√0.25=0.5 | 同为 `track_ang_vel_z_world_exp`，std 不同 |
| `ang_vel_xy_l2` | -0.05 | -0.05 | 相同 |
| `dof_vel_l2` | **-0.001**（`joint_vel_l2`） | **-2.0e-3**（`dof_vel_l2`） | 同名不同函数名，量级相同 |
| `dof_acc_l2` | -1.25e-7 | -2.5e-7 | 同为 `joint_acc_l2` |
| `dof_power_l2` | 无 | **-2.0e-5** | Attention 独有（功耗惩罚） |
| `dof_torques_l2` | **-1.5e-7**，全部关节 | **-1.0e-5**，仅 `leg_[lr][1-5]_joint`、`zarm_.*` | 权重差约 2 个数量级；且 Attention 将踝关节单独处理 |
| `dof_torques_ankle_l2` | 无 | **-1.0e-5**，仅 `leg_[lr]6_joint` | Attention 独有（踝关节单独扭矩惩罚） |
| `dof_torques_limits` | **-0.01**（`applied_torque_limits`） | 无 | AME 独有 |
| `action_rate_l2` | **-0.01** | **-0.005** | 同为 `action_rate_l2` |
| `action_smoothness_l2` | 无 | **-0.01** | Attention 独有（动作二阶平滑） |
| `undesired_contacts` | **-1.0**，threshold=1；排除 `leg_[lr][56]_link` | **-1.0**，threshold=1；限定 `leg_[lr][1-5]_link`、`base_link`、`zarm_.*` | 语义相同、写法不同 |
| `dof_pos_limits` | **-1.0** | **-10.0** | 同为 `joint_pos_limits`，权重差 10 倍 |
| `termination_penalty` | **-200**（`is_terminated`） | **-200**（`is_terminated`） | 相同 |
| `feet_air_time` | **0.25**，threshold=0.6 | **1.0**，threshold=0.5，`use_stance_mask=True` | 权重差 4 倍；Attention 有 stance mask |
| `feet_air_time_variance` | **-0.7** | 无 | AME 独有 |
| `feet_slide` | -0.1 | -0.1 | 相同 |
| `feet_stumble` | **-2.0** | **-1.0** | 权重不同 |
| `feet_too_near` | **-1.0**，threshold=0.2 | 无 | AME 独有 |
| `feet_contact_without_cmd` | 无 | **+0.4** | Attention 独有（无指令时鼓励触地） |
| `no_feet_contact` | 无 | **-0.1** | Attention 独有 |
| `contact_force` | 无 | **-0.001**，threshold=900，violation_max=300 | Attention 独有（足端冲击力惩罚） |
| `stand_still_without_cmd` | 无 | **-1.0** | Attention 独有（无指令禁止站立停滞） |
| `illegal_dof_barrier` | 无 | **-0.1**，`leg_[lr][56]_joint` | Attention 独有（软限位屏障） |
| `track_default_arm_pos` | 无 | **+1.0**，alpha=5.0，`zarm_{l,r}{2,3,5,6,7}_joint` | Attention 独有（约束手臂回到默认位） |
| `joint_coordination` | **-0.2**（`joint_coordination_rel`，`leg_l3↔zarm_r1`、`leg_r3↔zarm_l1`） | 无 | AME 独有（跨侧腿臂协调） |
| `joint_deviation_hip` | -0.1，`leg_[lr][12]_joint` | -0.1，`leg_[lr][1,2]_joint` | 相同 |
| `joint_deviation_arms` | **-0.3**，`zarm_[lr][1-7]_joint` | **-0.5**，`zarm_.*` | 权重不同 |
| `joint_deviation_waists` | 无（S46 无腰部，显式置 None） | 无 | — |
| `flat_orientation_l2` | **-2.0** | **-3.0** | 权重不同 |

### 1.2 Reward 设计差异总结

- **共同基础项**：速度跟踪、角速度跟踪、`ang_vel_xy_l2`、`action_rate_l2`、`undesired_contacts`、`dof_pos_limits`、终止惩罚、`feet_air_time`、`feet_slide`、`feet_stumble`、`flat_orientation_l2`、髋/臂偏离。
- **AME 独有的“风格”项**：`feet_air_time_variance`（双腿腾空时间一致性）、`feet_too_near`（双脚过近）、`joint_coordination`（跨侧腿臂协同）、`joint_deviation_waists`（腰部偏离）、`dof_torques_limits`（输出扭矩限制）。这些项重点在**协调步态与自然肢体动作**。
- **Attention 独有的“触地/状态”项**：`feet_contact_without_cmd`、`no_feet_contact`、`stand_still_without_cmd`（均配合 `use_stance_mask` 或命令判断）、`contact_force`（足端冲击）、`dof_power_l2`（能耗）、`action_smoothness_l2`（动作平滑）、`track_default_arm_pos`（手臂归位）、`illegal_dof_barrier`（软限位）、`dof_torques_ankle_l2`（踝关节独立扭矩惩罚）。这些项重点在**能耗、冲击、触地状态约束**。
- **权重倾向**：Attention 对速度跟踪（5.0 vs 2.0）、腾空（1.0 vs 0.25）、位置限位（-10 vs -1）、扭矩（-1e-5 vs -1.5e-7）等核心指标显著更激进；AME 更依赖协调/方差类软约束。

---

## 2. Event 对比

### 2.1 逐项对比表

| Event 项 | mode | AME S46（`FINETUNE=False` 默认生效） | Attention | 备注 |
| --- | --- | --- | --- | --- |
| `physics_material` | startup | 启用；static/dynamic friction **(0.3, 1.0)**，restitution **(0, 0.1)**，num_buckets=64 | 启用；static friction **(0.2, 1.0)**，dynamic **(0.1, 0.9)**，restitution **(0, 0.5)**，num_buckets=64，`make_consistent=True` | Attention 摩擦下限更低、restitution 上限更高 |
| `add_base_mass` | startup | **禁用**（默认分支置 None） | 启用；base_link 质量 add (−2, 2) | AME 默认关闭 |
| `scale_link_mass` | startup | 无 | 启用；leg/zarm 各连杆质量 scale (0.8, 1.2) | Attention 独有 |
| `base_com` / `randomize_rigid_body_com` | startup | **禁用** | 启用；COM x/y/z 各 ±0.1 | AME 默认关闭；Attention 用 `randomize_base_body_com` |
| `scale_actuator_gains` | startup | 无 | 启用；全部关节刚度/阻尼 scale (0.8, 1.2) | Attention 独有 |
| `scale_joint_parameters` | startup | 无 | 启用；摩擦 scale (1.0, 1.0)，armature scale (0.5, 1.5) | Attention 独有 |
| `base_external_force_torque` | reset | 启用但 force/torque 范围全 0（等价空操作） | 无（interval 版被注释） | — |
| `reset_base` | reset | pose x/y ±0.5、yaw ±3.14，velocity **全 0** | pose x/y **±0.7**、yaw ±3.14，velocity **各 ±0.3** | Attention reset 带速度随机 |
| `reset_robot_joints` | reset | position_range **(1.0, 1.0)**（固定默认位）、velocity ±1.0 | position_range **(0.5, 1.5)**、velocity (0, 0) | AME 复位到默认位更确定性 |
| `push_robot` | interval | **禁用**（默认分支置 None；仅 FINETUNE 模式启用，interval 5–10s，xy ±0.5 m/s） | 无 | AME 默认关闭推扰 |

### 2.2 Event 设计差异总结

- **域随机化丰富度**：Attention 一次性启用了 **6 类 startup 随机化**（材料、基座质量加料、连杆质量缩放、COM、执行器增益、关节参数），并配合带速度随机化的 reset；AME 默认（`FINETUNE=False`）**只保留材料随机化和根/关节状态 reset**，`add_base_mass`、`base_com`、`push_robot` 全部关闭，域随机化明显更克制（更强的扰动放在 FINETUNE 阶段）。
- **reset 幅度**：Attention 的 reset 姿态范围（±0.7）与速度随机（±0.3）比 AME（±0.5 / 0）更大。
- **关节复位方式**：AME 将关节位置复位到默认位（position_range=1.0）并随机速度 ±1.0；Attention 关节位置在 0.5–1.5 倍间随机、速度归零。
- **触发模式**：两者均无 interval 扰动（AME 的 push 默认关闭、Attention 无 push）。

---

## 3. 网络结构对比

### 3.1 配置项对比

| 项目 | AME `ActorCriticEncoder` | Attention `EncActorCritic` |
| --- | --- | --- |
| 网络类名 | `ActorCriticEncoder`（`rsl_rl/modules/actor_critic_encoder.py`） | `EncActorCritic`（rsl_rl 内实现，仓库中由 exporter 镜像） |
| 地形/感知编码 | CNN 下采样 + **MultiheadAttention**（query=proprio 嵌入，key/value=CNN 局部特征） | 专用 `encoder(map_scan, proprio)` 模块，`embedding_dim=64`，可输出 attention 权重 |
| map 输入形式 | 拼在 **policy/critic 扁平观测末尾**，2079=33×21×3（0.05m grid），内部 reshape 为 (W,L,3) 再 permute | 独立 **`perception` obs group**，结构化 `[B, L, W, 3]`（17×11，0.1m grid） |
| CNN 结构 | Conv2d(3→16, k5, s2) + BN + ReLU → Conv2d(16→64, k3, s1) + BN + ReLU（下采样后 187 个 token） | 仓库内未见 CNN 定义，map 直接进 attention encoder |
| Attention | `nn.MultiheadAttention(embed_dim=64, num_heads=16, batch_first=True)`；`attach_global=False`（默认） | `embedding_dim=64`；`output_attention` 供可视化；`load_mask=15`（1 策略 + 2 评论 + 4 编码器 + 8 归一化，用于 checkpoint 部分加载） |
| 归一化 | `actor_obs_normalization=False`、`critic_obs_normalization=False` | `actor_obs_normalization=True`、`critic_obs_normalization=True`，且 `empirical_normalization=True` |
| 噪声 std | `noise_std_type` 默认 `scalar` | `noise_std_type='log'`（保证 std>0） |
| obs_groups | 默认解析：`policy=["policy"]`、`critic=["critic"]` | 显式：`policy=["command","policy"]`、`critic=["command","privileged"]`、`perception=["perception"]` |
| 历史维度 | 无 history（单步扁平向量） | `history_length=1`、`flatten_history_dim=False`（保留 `[B,H,D,...]`） |
| Actor MLP | 输入 = 64 + 87 = **151** → `[512,256,128]` → 26 | 基于 embedding+proprio → `[512,256,128]` → 26 |
| Critic MLP | 输入 = 64 + 90 = **154** → `[512,256,128]` → 1 | `[512,256,128]` → 1 |
| 激活函数 | `elu` | `elu` |
| init_noise_std | 1.0 | 1.0 |

> 输入维度说明：AME 的 proprio 维数按迁移文档为 policy 87 / critic 90（不含 2079 维地图），因此 Actor/Critic MLP 输入为 64+87=151 / 64+90=154。ActorCriticEncoder 会从实际观测长度动态推算 proprio 维数。

### 3.2 网络结构差异总结

- **观测组织方式不同**：AME 把高度图作为扁平向量塞进 policy/critic 观测末尾，网络内部自行切分、reshape；Attention 用 `perception` 独立观测组传递结构化 `[B,L,W,3]` 地图，通过 `obs_groups` 映射到 actor/critic。
- **地图分辨率不同**：AME 0.05m / 33×21 / 2079 维；Attention 0.1m / 17×11。
- **编码器结构不同**：AME 先 CNN 下采样得到局部 token，再以 proprio 嵌入作为 query、CNN 特征作为 key/value 做单层 MHA；Attention 的 `EncActorCritic.encoder` 直接在 map 与 proprio 之间做注意力编码（`embedding_dim=64`），具体层数在 rsl_rl 内部实现（本仓库未包含源码，只有 exporter 镜像与 `load_mask`/`output_attention` 配置）。
- **观测归一化策略不同**：AME 完全关闭 obs normalization；Attention 开启 actor/critic 经验归一化并保留 `empirical_normalization`。
- **动作噪声参数化不同**：AME 用 `scalar`（`init_noise_std` 张量），Attention 用 `log`（`log_std` 参数），防止 std 变负。

### 3.3 Runner / 算法参数对比

| 项目 | AME | Attention |
| --- | --- | --- |
| `num_steps_per_env` | 24 | 24 |
| `max_iterations` | 10000 | 40000 |
| `save_interval` | 100 | 50 |
| `entropy_coef` | 0.008 | 0.005 |
| 其余算法参数 | clip 0.2、5 epochs、4 mini-batches、lr 1e-3、adaptive、γ 0.99、λ 0.95、desired_kl 0.01、max_grad_norm 1.0 | 与 AME 一致 |
| `experiment_name` | `kuavo_s46_ame` | `Kuavo/s42/atten` |

---

## 4. 其他相关差异（观测 / 命令 / 场景 / 终止）

| 项目 | AME S46 | Attention |
| --- | --- | --- |
| Policy 观测内容 | ang_vel、gravity、cmd、joint_pos、joint_vel、action、height_scan（含 map） | 单独 `command` 组 + `base_lin_vel`、ang_vel、gravity、joint_pos、joint_vel、action |
| Policy 观测噪声 | ang_vel ±0.2、gravity ±0.05、joint_pos ±0.01、joint_vel ±2.0；`enable_corruption=True` | base_lin_vel ±0.1、ang_vel ±0.2、gravity ±0.05、joint_pos ±0.05、joint_vel ±1.5 |
| 命令类型 | `UniformVelocityCommandCfg`，重采样 10s，lin_x (0,1.5)、lin_y (0,0)、ang_z (−1,1) | `UniformSteppingVelocityCommandCfg`，重采样 5s，lin_x (−1,1)、lin_y (−0.5,0.5)、ang_z (−1,1)，`rel_standing_envs=0.1`、`rel_stepping_envs=0.5` |
| 动作 | `JointPositionActionCfg` scale=0.25，默认位偏移 | 同（scale=0.25） |
| 终止条件 | `time_out` + `base_contact`（illegal_contact，`base_link`/`leg_[lr][1-4]_link`/`zarm_[lr][1-4]_link`） | `time_out` + `base_contact`（仅 `base_link`）+ `dof_pos_illegal`（`actuators_names="motor"`） |
| 地形 | `ROUGH_TERRAINS_CFG`，`max_init_terrain_level=5`，摩擦 combine="multiply"（1.0） | `ATTEN_ROUGH_TERRAINS_CFG`，`max_init_terrain_level=0`，摩擦 combine="average"（0.4） |
| num_envs | 2048 | 4096 |

---

## 5. 结论

1. **Reward**：两者共用速度跟踪、基础惩罚与足部项，但 AME 侧重“步态协调”（`feet_air_time_variance`、`joint_coordination`、`feet_too_near`），Attention 侧重“触地/能耗/冲击/平滑”（`contact_force`、`dof_power_l2`、`action_smoothness_l2`、`track_default_arm_pos` 等），且 Attention 在速度跟踪、限位、扭矩等核心项上权重普遍更大。
2. **Event**：Attention 使用完整 6 类 startup 域随机化 + 大范围/带速度 reset；AME 默认关闭质量/COM/推扰随机化（仅 FINETUNE 阶段开启），随机化强度显著更弱。
3. **网络结构**：二者都是“注意力地形编码 + MLP head”的架构，但 AME 用 **CNN 下采样 + 单层 MHA**（map 在扁平观测尾部，0.05m grid），且**关闭观测归一化**、`scalar` 噪声、单步扁平输入；Attention 用**独立 perception 组 + `EncActorCritic.encoder`**（0.1m grid、结构化 `[B,L,W,3]`），并**开启经验归一化**、`log` 噪声、`[B,H,D]` 输入。两者 MLP 头均为 `[512,256,128]` + elu，Runner 仅 `entropy_coef`、`max_iterations` 等少量差异。
