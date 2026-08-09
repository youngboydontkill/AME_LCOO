"""RSL-RL runner configuration for the Kuavo S46 AME task."""

from isaaclab.utils import configclass

from .ame_rsl_rl_ppo_cfg import G1AMEPPORunnerCfg


@configclass
class KuavoS46AMEPPORunnerCfg(G1AMEPPORunnerCfg):
    """Reuse AME PPO settings under an S46-specific experiment name."""

    experiment_name = "kuavo_s46_ame"
