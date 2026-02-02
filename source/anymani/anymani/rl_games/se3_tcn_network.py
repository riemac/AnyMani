"""TCN-based encoder network for SE(3)+tactile policies (AnyRotate).

This implements an rl_games custom network builder that:
- reads a 50-step history of (body_twists, last_action, tactile) from the flattened observation;
- encodes the temporal window with a causal Temporal Convolutional Network (TCN) into a latent vector h_t;
- concatenates {Prop_t, tau_t, h_t, c_t} and outputs (mu, logstd, value) for PPO.

Notes:
- The observation layout is assumed to match the policy group in
  AnyRotate/source/leaphand/leaphand/tasks/manager_based/leaphand/inhand_se3_tactile_env_cfg.py:
  [body_twists_hist, goal_pose, last_action_hist, fingertip_contact_binary_hist].
- This network is non-RNN from rl_games' perspective (is_rnn=False).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder


def _act(name: str) -> Callable[[], nn.Module]:
    name = (name or "elu").lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name == "silu" or name == "swish":
        return nn.SiLU
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported activation: {name}")


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[..., :-self.chomp_size]


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        act = _act(activation)

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            act(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            _Chomp1d(padding),
            act(),
            nn.Dropout(dropout),
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.final_act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(y + res)


class _TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_size: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for level, c_out in enumerate(channels):
            dilation = 2**level
            layers.append(
                _TemporalBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class _ObsLayout:
    seq_len: int
    twist_dim: int
    act_dim: int
    tactile_dim: int
    cmd_dim: int

    @property
    def prop_dim(self) -> int:
        return self.twist_dim + self.act_dim

    @property
    def seq_feat_dim(self) -> int:
        return self.twist_dim + self.act_dim + self.tactile_dim

    @property
    def body_twists_hist_dim(self) -> int:
        return self.seq_len * self.twist_dim

    @property
    def last_action_hist_dim(self) -> int:
        return self.seq_len * self.act_dim

    @property
    def tactile_hist_dim(self) -> int:
        return self.seq_len * self.tactile_dim

    @property
    def total_obs_dim(self) -> int:
        return self.body_twists_hist_dim + self.cmd_dim + self.last_action_hist_dim + self.tactile_hist_dim


class Se3TcnNet(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)

        self.actions_num = int(kwargs.pop("actions_num"))
        input_shape = kwargs.pop("input_shape")
        self.value_size = int(kwargs.pop("value_size", 1))

        # rl_games passes input_shape as a tuple for flat obs
        if isinstance(input_shape, dict):
            raise ValueError("Se3TcnNet expects a flat obs tensor, got dict input_shape.")
        obs_dim = int(input_shape[0])

        tcn_cfg = params.get("tcn", {})
        mlp_cfg = params.get("mlp", {})

        layout = _ObsLayout(
            seq_len=int(tcn_cfg.get("seq_len", 50)),
            twist_dim=int(tcn_cfg.get("twist_dim", 24)),
            act_dim=int(tcn_cfg.get("act_dim", 24)),
            tactile_dim=int(tcn_cfg.get("tactile_dim", 4)),
            cmd_dim=int(tcn_cfg.get("cmd_dim", 7)),
        )
        self.layout = layout
        if obs_dim != layout.total_obs_dim:
            raise ValueError(f"Obs dim mismatch for Se3TcnNet: got {obs_dim}, expected {layout.total_obs_dim}.")

        channels = list(tcn_cfg.get("channels", [128, 128, 128, 128, 128]))
        kernel_size = int(tcn_cfg.get("kernel_size", 3))
        dropout = float(tcn_cfg.get("dropout", 0.1))
        activation = str(tcn_cfg.get("activation", "elu"))
        latent_dim = int(tcn_cfg.get("latent_dim", channels[-1]))

        self.tcn = _TemporalConvNet(
            in_channels=layout.seq_feat_dim,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation,
        )
        tcn_out_dim = channels[-1]
        self.h_proj = nn.Identity() if latent_dim == tcn_out_dim else nn.Linear(tcn_out_dim, latent_dim)

        # Final feature: [Prop_t, tau_t, h_t, c_t]
        final_in = layout.prop_dim + layout.tactile_dim + latent_dim + layout.cmd_dim

        units = list(mlp_cfg.get("units", [512, 256]))
        mlp_act = str(mlp_cfg.get("activation", "elu"))
        act_cls = _act(mlp_act)

        def _mlp(out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            c_in = final_in
            for u in units:
                layers.append(nn.Linear(c_in, int(u)))
                layers.append(act_cls())
                c_in = int(u)
            layers.append(nn.Linear(c_in, out_dim))
            return nn.Sequential(*layers)

        self.actor = _mlp(self.actions_num)
        self.critic = _mlp(self.value_size)

        # logstd is a trainable parameter for continuous_a2c_logstd
        self.logstd = nn.Parameter(torch.zeros(self.actions_num, dtype=torch.float32), requires_grad=True)

        self._aux_loss_map = None

    def is_rnn(self):
        return False

    def get_aux_loss(self):
        return self._aux_loss_map

    def get_value_layer(self):
        return self.critic[-1]

    def forward(self, input_dict):
        obs = input_dict["obs"]  # (B, D)
        B = obs.shape[0]
        L = self.layout

        i0 = 0
        i1 = i0 + L.body_twists_hist_dim
        i2 = i1 + L.cmd_dim
        i3 = i2 + L.last_action_hist_dim
        i4 = i3 + L.tactile_hist_dim

        body_twists_hist = obs[:, i0:i1].reshape(B, L.seq_len, L.twist_dim)
        cmd = obs[:, i1:i2]
        last_action_hist = obs[:, i2:i3].reshape(B, L.seq_len, L.act_dim)
        tactile_hist = obs[:, i3:i4].reshape(B, L.seq_len, L.tactile_dim)

        # sequence encoder input: (B, C, T)
        seq = torch.cat([body_twists_hist, last_action_hist, tactile_hist], dim=-1)
        seq = seq.transpose(1, 2).contiguous()
        y = self.tcn(seq)  # (B, C, T)
        h_last = y[:, :, -1]
        h_t = self.h_proj(h_last)

        prop_t = torch.cat([body_twists_hist[:, -1, :], last_action_hist[:, -1, :]], dim=-1)
        tau_t = tactile_hist[:, -1, :]

        feat = torch.cat([prop_t, tau_t, h_t, cmd], dim=-1)

        mu = self.actor(feat)
        value = self.critic(feat)
        logstd = self.logstd.unsqueeze(0).expand_as(mu)
        return mu, logstd, value, None


class Se3TcnNetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return Se3TcnNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


# Register network builder for rl_games Runner.
model_builder.register_network("se3_tcn", Se3TcnNetBuilder)
