"""Isaac Lab articulation configuration for Kuavo S54."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import ISAAC_ASSET_DIR


KUAVO_S54_URDF = f"{ISAAC_ASSET_DIR}/biped_s54/urdf/biped_s54.urdf"


KUAVO_S54_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=KUAVO_S54_URDF,
        fix_base=False,
        root_link_name="base_link",
        merge_fixed_joints=True,
        self_collision=True,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.925),
        joint_pos={
            "leg_[lr]1_joint": 0.0,
            "leg_[lr]2_joint": 0.0,
            "leg_[lr]3_joint": -0.4,
            "leg_[lr]4_joint": 0.69,
            "leg_[lr]5_joint": -0.33,
            "leg_[lr]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "motor": DelayedPDActuatorCfg(
            joint_names_expr=[
                "leg_.*_joint",
                "waist_yaw_joint",
                "zarm_.*_joint",
            ],
            effort_limit={
                "leg_[lr]1_joint": 101.6,
                "leg_[lr]2_joint": 56.8,
                "leg_[lr]3_joint": 105.6,
                "leg_[lr]4_joint": 224.0,
                "leg_[lr]5_joint": 91.6,
                "leg_[lr]6_joint": 57.0,
                "waist_yaw_joint": 81.6,
                "zarm_[lr]1_joint": 52.8,
                "zarm_[lr]2_joint": 60.0,
                "zarm_[lr]3_joint": 45.6,
                "zarm_[lr]4_joint": 60.0,
                "zarm_[lr][5-7]_joint": 11.0,
            },
            velocity_limit={
                "leg_[lr]1_joint": 10.4,
                "leg_[lr]2_joint": 8.7,
                "leg_[lr]3_joint": 12.7,
                "leg_[lr]4_joint": 10.4,
                "leg_[lr][56]_joint": 17.8,
                "waist_yaw_joint": 8.7,
                "zarm_[lr]1_joint": 18.8,
                "zarm_[lr]2_joint": 8.0,
                "zarm_[lr]3_joint": 7.5,
                "zarm_[lr]4_joint": 8.0,
                "zarm_[lr][5-7]_joint": 17.5,
            },
            stiffness={
                "leg_[lr][12]_joint": 60.0,
                "leg_[lr]3_joint": 80.0,
                "leg_[lr]4_joint": 95.0,
                "leg_[lr][56]_joint": 55.0,
                "waist_yaw_joint": 40.0,
                "zarm_[lr][1-4]_joint": 20.0,
                "zarm_[lr][5-7]_joint": 15.0,
            },
            damping={
                "leg_[lr][1-4]_joint": 6.0,
                "leg_[lr][56]_joint": 7.5,
                "waist_yaw_joint": 4.0,
                "zarm_[lr][1-7]_joint": 3.0,
            },
            armature={
                "leg_[lr]1_joint": 0.05,
                "leg_[lr][23]_joint": 0.025,
                "leg_[lr][4-6]_joint": 0.05,
                "waist_yaw_joint": 0.025,
                "zarm_[lr]1_joint": 0.025,
                "zarm_[lr][2-4]_joint": 0.02,
                "zarm_[lr][5-7]_joint": 0.01,
            },
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
    },
)
