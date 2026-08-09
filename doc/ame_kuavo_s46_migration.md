# AME 从 Unitree G1 迁移到 Kuavo S46

## 1. 迁移目标与约束

Kuavo S46 使用 `kuavo/assets/Robots/Kuavo/biped_s46.usd`，包含 26 个可驱动关节：双腿各 6 个、双臂各 7 个。原 AME G1 任务包含 29 个关节，二者的主要结构差异是 S46 没有 G1 的 3 个腰部关节。

本次迁移遵循以下原则：

- observation 的项目、顺序、scale、noise 和地形高程图配置与 G1 一致；关节相关维度随 29DoF 变为 26DoF。
- reward 的函数、参数和 scale 与 G1 一致，只替换依赖机器人命名的 joint/link selector。
- termination、event、command、terrain 和 curriculum 的行为保持一致，只替换基座和接触刚体名称。
- S46 不存在腰部关节，因此 `joint_deviation_waists` 在 S46 中显式禁用；这是唯一无法一一映射的 reward。
- 不修改 USD 中的几何体、碰撞体或关节，只在 Isaac Lab 配置中适配现有命名。

## 2. 关节对应关系

### 2.1 腿部

| G1 语义关节 | Kuavo S46 joint | Kuavo S46 child link | 用途 |
| --- | --- | --- | --- |
| `{left,right}_hip_roll_joint` | `leg_{l,r}1_joint` | `leg_{l,r}1_link` | hip deviation、非法接触 |
| `{left,right}_hip_yaw_joint` | `leg_{l,r}2_joint` | `leg_{l,r}2_link` | hip deviation、非法接触 |
| `{left,right}_hip_pitch_joint` | `leg_{l,r}3_joint` | `leg_{l,r}3_link` | 跨侧腿臂协调、非法接触 |
| `{left,right}_knee_joint` | `leg_{l,r}4_joint` | `leg_{l,r}4_link` | 非法接触 |
| `{left,right}_ankle_pitch_joint` | `leg_{l,r}5_joint` | `leg_{l,r}5_link` | 从 undesired contact 中排除 |
| `{left,right}_ankle_roll_joint` | `leg_{l,r}6_joint` | `leg_{l,r}6_link` | 足部 contact/reward |

USD 中关节 1–6 的旋转轴依次对应 roll、yaw、pitch、pitch、pitch、roll，因此上述映射依据实际 USD 关节轴和运动链确定，而非仅依赖编号猜测。

### 2.2 手臂

| G1 语义关节 | Kuavo S46 joint | Kuavo S46 child link | 用途 |
| --- | --- | --- | --- |
| shoulder pitch | `zarm_{l,r}1_joint` | `zarm_{l,r}1_link` | 腿臂协调、arm deviation、非法接触 |
| shoulder roll | `zarm_{l,r}2_joint` | `zarm_{l,r}2_link` | arm deviation、非法接触 |
| shoulder yaw | `zarm_{l,r}3_joint` | `zarm_{l,r}3_link` | arm deviation、非法接触 |
| elbow | `zarm_{l,r}4_joint` | `zarm_{l,r}4_link` | arm deviation、非法接触 |
| wrist yaw | `zarm_{l,r}5_joint` | `zarm_{l,r}5_link` | arm deviation |
| wrist roll | `zarm_{l,r}6_joint` | `zarm_{l,r}6_link` | arm deviation |
| wrist pitch | `zarm_{l,r}7_joint` | `zarm_{l,r}7_link` | arm deviation |

### 2.3 基座与腰部

| G1 link/joint | Kuavo S46 对应项 | 处理方式 |
| --- | --- | --- |
| `pelvis`、`waist_*_link`、`torso_link` | `base_link` | 高程扫描、相机、质量/质心随机化、外力和 base contact 均使用 `base_link` |
| `waist_yaw_joint`、`waist_roll_joint`、`waist_pitch_joint` | 无 | `joint_deviation_waists` 设为 `None` |

## 3. 碰撞体与接触名称

Isaac Lab 的 `ContactSensorCfg` 和 `SceneEntityCfg.body_names` 匹配的是刚体名称，例如 `leg_l6_link`，不是 USD 下的 `_geom_*` 碰撞 prim 名称。S46 USD 中存在碰撞几何的刚体如下：

| 刚体 | USD collision prim |
| --- | --- |
| `base_link` | `_geom_1`、`_geom_2` |
| `leg_l1_link` | `_geom_4` |
| `leg_l3_link` | `_geom_7`、`_geom_8` |
| `leg_l4_link` | `_geom_10` |
| `leg_l6_link` | `_geom_13`、`_geom_14`、`_geom_15` |
| `leg_r1_link` | `_geom_17` |
| `leg_r3_link` | `_geom_20`、`_geom_21` |
| `leg_r4_link` | `_geom_23` |
| `leg_r6_link` | `_geom_26`、`_geom_27`、`_geom_28` |
| `zarm_l1_link`、`zarm_l2_link`、`zarm_l3_link` | `_geom_30`、`_geom_32`、`_geom_34` |
| `zarm_l6_link`、`zarm_l7_link` | `_geom_38`、`_geom_40` |
| `zarm_r1_link`、`zarm_r2_link`、`zarm_r3_link`、`zarm_r4_link` | `_geom_42`、`_geom_44`、`_geom_46`、`_geom_48` |
| `zarm_r6_link`、`zarm_r7_link` | `_geom_51`、`_geom_53` |

具体迁移规则：

- 足部 reward/contact：`.*_ankle_roll_link` → `leg_[lr]6_link`。
- undesired contact：G1 排除所有 ankle link；S46 对应排除 `leg_[lr][56]_link`。当前 USD 中实际足部碰撞位于 `leg_[lr]6_link`，同时排除 5/6 可保持与 G1 selector 的语义一致。
- 非法接触终止：`torso/pelvis/waist` → `base_link`，hip/knee → `leg_[lr][1-4]_link`，shoulder/elbow → `zarm_[lr][1-4]_link`。

## 4. Reward 迁移

不依赖具体名称的 reward 原样保留：

- `termination_penalty`
- `track_lin_vel_xy_exp`
- `track_ang_vel_z_exp`
- `ang_vel_xy_l2`
- `dof_torques_l2`
- `dof_acc_l2`
- `dof_vel_l2`
- `dof_pos_limits`
- `dof_torques_limits`
- `action_rate_l2`
- `flat_orientation_l2`

需要修改 selector 的 reward：

| Reward | G1 selector | S46 selector |
| --- | --- | --- |
| `undesired_contacts` | 非 ankle 刚体 | 除 `leg_[lr][56]_link` 外的刚体 |
| `feet_air_time` | `.*_ankle_roll_link` | `leg_[lr]6_link` |
| `feet_air_time_variance` | `.*_ankle_roll_link` | `leg_[lr]6_link` |
| `feet_slide` | `.*_ankle_roll_link` | `leg_[lr]6_link` |
| `feet_stumble` | `.*_ankle_roll_link` | `leg_[lr]6_link` |
| `feet_too_near` | `.*_ankle_roll_link` | `leg_[lr]6_link` |
| `joint_coordination` | left hip pitch ↔ right shoulder pitch；反侧同理 | `leg_l3_joint` ↔ `zarm_r1_joint`；`leg_r3_joint` ↔ `zarm_l1_joint` |
| `joint_deviation_hip` | hip yaw/roll | `leg_[lr][12]_joint` |
| `joint_deviation_arms` | shoulder/elbow/wrist | `zarm_[lr][1-7]_joint` |
| `joint_deviation_waists` | `waist.*` | 禁用（S46 无对应关节） |

除腰部项禁用外，基础训练与 `FINETUNE` 两阶段中的所有 reward scale 均沿用 G1 配置。

## 5. Termination、Event 与 Observation 迁移

### Termination

- `time_out` 不变。
- `base_contact` 的阈值保持 `1.0`，body selector 改为 `base_link`、`leg_[lr][1-4]_link`、`zarm_[lr][1-4]_link`。

### Event

- 全刚体材料随机化、关节 reset 和 push event 不变。
- `add_base_mass`、`base_com`、`base_external_force_torque` 的 body 从 `torso_link` 改为 `base_link`。
- reset pose、velocity、随机化范围及 FINETUNE 开关行为不变。

### Observation 与动作维度

Observation term 的顺序和参数保持不变：

1. base angular velocity
2. projected gravity
3. velocity command
4. relative joint position
5. relative joint velocity
6. last action
7. height scan

Critic 在最前面额外包含 base linear velocity，与 G1 一致。高度图仍为 `33 × 21 × 3 = 2079` 维。

| 接口 | G1 | Kuavo S46 |
| --- | ---: | ---: |
| action | 29 | 26 |
| policy proprioception | 96 | 87 |
| policy 总 observation | 2175 | 2166 |
| critic 总 observation | 2178 | 2169 |

AME 编码器会从实际 observation 动态计算 proprioception 维度，因此无需修改网络结构。不过 G1 checkpoint 的输入层和动作头维度均与 S46 不同，不能直接加载，S46 需要重新训练。

## 6. 新任务名称

- 训练：`AME-Kuavo-S46-v0`
- 播放：`AME-Kuavo-S46-Play-v0`

两者使用独立的 `kuavo_s46_ame` 实验目录，不修改现有 G1 任务及其运行脚本。
