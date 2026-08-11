"""RSL-RL runner configuration for the Kuavo S46 AME task."""

from isaaclab.utils import configclass

from .ame_rsl_rl_ppo_cfg import G1AMEPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class KuavoS46AMEPPORunnerCfg(G1AMEPPORunnerCfg):
    """Reuse AME PPO settings under an S46-specific experiment name."""

    experiment_name = "kuavo_s46_ame"

@configclass
class KuavoS46BaselinePPORunnerCfg(G1AMEPPORunnerCfg):
    """Match AME PPO settings while removing attention from the policy."""

    experiment_name = "kuavo_s46_baseline"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCNNMLP",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )