# AME 从 Unitree G1 迁移到 Kuavo S54

## 1. 模型与自由度

S54 资产入口为 `kuavo/assets/biped_s54/urdf/biped_s54.urdf`。Isaac Lab 通过 `UrdfFileCfg` 在运行时转换为 USD，并合并头部、相机、雷达和末端执行器的 fixed joint。

转换后的 articulation 有 27 个可控关节：双腿 12DoF、腰部 yaw 1DoF、双臂 14DoF。与 G1 29DoF 相比，S54 仅缺少 waist roll/pitch；与 S46 26DoF 相比，S54 多出 `waist_yaw_joint`。

## 2. Joint 与 link 映射

### 腿部

| G1 语义 | S54 joint | S54 child link |
| --- | --- | --- |
| hip roll | `leg_[lr]1_joint` | `leg_[lr]1_link` |
| hip yaw | `leg_[lr]2_joint` | `leg_[lr]2_link` |
| hip pitch | `leg_[lr]3_joint` | `leg_[lr]3_link` |
| knee | `leg_[lr]4_joint` | `leg_[lr]4_link` |
| ankle pitch | `leg_[lr]5_joint` | `leg_[lr]5_link` |
| ankle roll / foot | `leg_[lr]6_joint` | `leg_[lr]6_link` |

### 手臂与腰部

| G1 语义 | S54 joint | S54 child link |
| --- | --- | --- |
| shoulder pitch/roll/yaw | `zarm_[lr][1-3]_joint` | `zarm_[lr][1-3]_link` |
| elbow | `zarm_[lr]4_joint` | `zarm_[lr]4_link` |
| wrist yaw/roll/pitch | `zarm_[lr][5-7]_joint` | `zarm_[lr][5-7]_link` |
| waist yaw | `waist_yaw_joint` | `waist_yaw_link` |
| waist roll/pitch | 无 | 无 |
| pelvis/root | 无关节 | `base_link` |
| torso/upper body | `waist_yaw_joint` | `waist_yaw_link` |

跨侧协调 reward 使用 `leg_l3_joint ↔ zarm_r1_joint` 和 `leg_r3_joint ↔ zarm_l1_joint`。hip deviation 使用腿关节 1/2，arm deviation 使用全部 14 个 `zarm` 关节，waist deviation 只使用 `waist_yaw_joint`。

## 3. 碰撞体与接触映射

`SceneEntityCfg.body_names` 匹配转换后的刚体/link 名称，不匹配 URDF collision 几何名称。

| link | URDF collision 数量 | AME 用途 |
| --- | ---: | --- |
| `base_link` | 2 | pelvis/base 非法接触 |
| `waist_yaw_link` | 2 | torso/waist 非法接触 |
| `leg_[lr]3_link` | 每侧 2 | hip 非法接触 |
| `leg_[lr]4_link` | 每侧 1 | knee 非法接触 |
| `leg_[lr]6_link` | 每侧 13 | 足部 air-time、slide、stumble、distance |
| `zarm_[lr]1_link` | 每侧 1 | shoulder 非法接触 |
| `zarm_[lr]3_link` | 每侧 1 | shoulder 非法接触 |
| `zarm_[lr]4_link` | 每侧 1 | elbow 非法接触 |
| `zarm_[lr]7_link` | 每侧 2 | 手部碰撞；不作为跌倒终止 |

具体 selector：

- 足部 reward：`leg_[lr]6_link`。
- undesired contact：除 `leg_[lr][56]_link` 外的所有刚体。
- base contact termination：`base_link`、`waist_yaw_link`、`leg_[lr][1-4]_link`、`zarm_[lr][1-4]_link`。
- 高程扫描器和播放相机挂载到 `waist_yaw_link`。

## 4. Actuator 配置

全部 27 个可动关节使用 Isaac Lab 原生 `DelayedPDActuatorCfg`，命令延迟范围为 0–4 个 physics step。stiffness、damping、effort limit 和 armature 来自 S54 资产目录中的现有名义参数；velocity limit 来自 URDF。

| 关节组 | stiffness | damping |
| --- | ---: | ---: |
| leg 1/2 | 60 | 6 |
| leg 3 | 80 | 6 |
| leg 4 | 95 | 6 |
| leg 5/6 | 55 | 7.5 |
| waist yaw | 40 | 4 |
| arm 1–4 | 20 | 3 |
| arm 5–7 | 15 | 3 |

URDF importer 的内建 drive 被设为 `target_type="none"`，避免与显式 delayed PD 重复施加控制力。

S54 每只脚包含 13 个 collision shape。为避免 2048 个并行环境且开启 self-collision 时触发 PhysX `collisionStackSize buffer overflow`，S54 训练和播放配置将 `sim.physx.gpu_collision_stack_size` 从默认 `2**26` 提高至 `2**28` bytes。该设置只作用于 S54，不改变 G1/S46。

## 5. MDP 迁移

- Reward 函数、参数和基础/FINETUNE scale 与 G1 保持一致。
- Observation 项目、顺序、scale、noise 和 `33 × 21 × 3` 高程图保持一致。
- Termination 阈值、event 随机化范围、command、terrain 和 curriculum 保持一致。
- `add_base_mass`、`base_com` 和 `base_external_force_torque` 从 G1 的 `torso_link` 映射至 S54 的 `waist_yaw_link`。
- `joint_deviation_waists` 保留原 scale，但只约束现有的 `waist_yaw_joint`。

| 接口 | G1 | S54 |
| --- | ---: | ---: |
| action | 29 | 27 |
| policy proprioception | 96 | 90 |
| policy 总 observation | 2175 | 2169 |
| critic 总 observation | 2178 | 2172 |

AME 编码器会根据实际 observation 动态创建 proprioception embedding，但 G1/S46 checkpoint 的输入层和动作头维度不同，不能直接加载到 S54。

## 6. 任务名称

- 训练：`AME-Kuavo-S54-v0`
- 播放：`AME-Kuavo-S54-Play-v0`
- RSL-RL experiment：`kuavo_s54_ame`
