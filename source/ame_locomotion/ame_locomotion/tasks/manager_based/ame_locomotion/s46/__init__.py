import gymnasium as gym

from ame_locomotion.tasks.manager_based.ame_locomotion import agents


gym.register(
    id="AME-Kuavo-S46-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s46:KuavoS46RoughEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.kuavo_s46_ame_rsl_rl_ppo_cfg:KuavoS46AMEPPORunnerCfg"},
)

gym.register(
    id="AME-Kuavo-S46-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s46:KuavoS46RoughEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.kuavo_s46_ame_rsl_rl_ppo_cfg:KuavoS46AMEPPORunnerCfg"},
)

gym.register(
    id="Baseline-Kuavo-S46-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s46:KuavoS46RoughEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.kuavo_s46_ame_rsl_rl_ppo_cfg:KuavoS46BaselinePPORunnerCfg"},
)

gym.register(
    id="Baseline-Kuavo-S46-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s46:KuavoS46RoughEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.kuavo_s46_ame_rsl_rl_ppo_cfg:KuavoS46BaselinePPORunnerCfg"},
)
