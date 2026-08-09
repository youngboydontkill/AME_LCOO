from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticCNNMLP(nn.Module):
    is_recurrent = False

    def __init__(self, obs, obs_groups, num_actions, actor_obs_normalization=False,
                 critic_obs_normalization=False, actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128], activation="elu",
                 init_noise_std=1.0, noise_std_type="scalar",
                 map_scan_dim=(33, 21, 3), cnn_output_dim=64, cnn_downsample=True, **kwargs):
        super().__init__()
        self.obs_groups, self.map_scan_dim = obs_groups, map_scan_dim
        self.L, self.W, self.coord_dim = map_scan_dim
        self.cnn_output_dim, self.cnn_downsample = cnn_output_dim, cnn_downsample
        actor_obs = self._obs_dim(obs, obs_groups["policy"])
        critic_obs = self._obs_dim(obs, obs_groups["critic"])
        map_size = self.L * self.W * self.coord_dim
        self.actor_proprio_dim, self.critic_proprio_dim = actor_obs - map_size, critic_obs - map_size
        if min(self.actor_proprio_dim, self.critic_proprio_dim) <= 0:
            raise ValueError("Observation dimensions do not contain a valid proprioceptive prefix")
        self.cnn_rows, self.cnn_cols = ((self.W + 1) // 2, (self.L + 1) // 2) if cnn_downsample else (self.W, self.L)
        first = nn.Conv2d(self.coord_dim, 16, 5, padding=2, stride=2 if cnn_downsample else 1)
        second = nn.Conv2d(16, cnn_output_dim, 3 if cnn_downsample else 5, padding=1 if cnn_downsample else 2)
        self.map_cnn = nn.Sequential(first, nn.ReLU(), nn.BatchNorm2d(16), second, nn.ReLU(), nn.BatchNorm2d(cnn_output_dim))
        # Global average pooling keeps the terrain feature width equal to cnn_output_dim (AME mha_dim=64).
        map_features = cnn_output_dim
        self.actor = MLP(map_features + self.actor_proprio_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(map_features + self.critic_proprio_dim, 1, critic_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(self.actor_proprio_dim) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(self.critic_proprio_dim) if critic_obs_normalization else nn.Identity()
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")
        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _obs_dim(obs, groups):
        return sum(obs[group].shape[-1] for group in groups)

    def _encode(self, obs):
        size = self.L * self.W * self.coord_dim
        scan = obs[:, -size:].reshape(-1, self.W, self.L, self.coord_dim).permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(scan).mean(dim=(2, 3))
        return torch.cat((obs[:, :-size], cnn_features), dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self): return self.distribution.mean
    @property
    def action_std(self): return self.distribution.stddev
    @property
    def entropy(self): return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(self._encode(obs))
        std = (self.std if self.noise_std_type == "scalar" else torch.exp(self.log_std)).expand_as(mean)
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs): return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
    def get_critic_obs(self, obs): return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)
    def act(self, obs, **kwargs):
        self.update_distribution(self.actor_obs_normalizer(self.get_actor_obs(obs)))
        return self.distribution.sample()
    def act_inference(self, obs):
        return self.actor(self._encode(self.actor_obs_normalizer(self.get_actor_obs(obs))))
    def evaluate(self, obs, **kwargs):
        return self.critic(self._encode(self.critic_obs_normalizer(self.get_critic_obs(obs))))
    def get_actions_log_prob(self, actions): return self.distribution.log_prob(actions).sum(dim=-1)
    def update_normalization(self, obs):
        if self.actor_obs_normalization: self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization: self.critic_obs_normalizer.update(self.get_critic_obs(obs))
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
