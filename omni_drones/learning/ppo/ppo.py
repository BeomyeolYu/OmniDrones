# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo"

    lr_a: float = 5e-4 #3e-4  #(default: 5e-4)
    lr_c: float = 5e-4 #2e-4  #(default: 5e-4)

    train_every: int = 2048  #(default: 32)
    ppo_epochs: int = 4  #(default: 4)
    num_minibatches: int = 16  #(default: 16)
    entropy_coef: float = 0.001  #(default: 0.001)
    clip_param: float = 0.1  #(default: 0.1)

    gamma: float = 0.99  #(default: 0.99)
    GAE_lambda: float = 0.9  #(default: 0.95)
    grad_max_norm: float = 5  #(default: 5)
    
    # network size
    actor_hidden_dim: int = 16  #(default: [256, 256, 256])
    critic_hidden_dim: int = 256  #(default: [256, 256, 256])

    lam_T: float = 0.1 #(default: 0.4)
    lam_S: float = 0.05 #(default: 0.3)
    lam_M: float = 0.2 #(default: 0.6)

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        # loc = torch.tanh(self.actor_mean(features)) # TODO: Try tanh to ensure the output is [-1, 1]
        '''
        print("loc:",loc)
            tensor([[[ 0.9240, -0.9794,  0.8404, -0.8992]]], device='cuda:0')
            tensor([[[ 0.9201, -0.9782,  0.8411, -0.9013]]], device='cuda:0')
            tensor([[[ 0.9162, -0.9746,  0.8244, -0.9049]]], device='cuda:0')
        '''
        scale = torch.exp(self.actor_std).expand_as(loc) # TODO: Try * 0. to make it deterministic
        return loc, scale


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Nominal value of quadrotor parameters:
        self.m_nominal = 2.15 # mass of quad, [kg]
        self.d_nominal = 0.23 # arm length, [m]
        self.J = torch.diag(torch.tensor([0.022, 0.022, 0.035], device=self.device)) # inertia matrix of quad, [kg m2]
        self.c_tf_nominal = 0.0135 # torque-to-thrust coefficients
        self.c_tw_nominal = 2.2 # thrust-to-weight coefficients
        self.g = 9.81 # standard gravity

        # Force and Moment:
        self.f = self.m_nominal * self.g # magnitude of total thrust to overcome  
                                 # gravity and mass (No air resistance), [N]
        self.hover_force = self.m_nominal * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.5 # minimum thrust of each motor, [N]
        self.max_force = self.c_tw_nominal * self.hover_force # maximum thrust of each motor, [N]
        self.avrg_act = (self.min_force+self.max_force)/2.0 
        self.scale_act = self.max_force-self.avrg_act # actor scaling
        self.max_action = 1.

        self.entropy_coef = self.cfg.entropy_coef
        self.clip_param = self.cfg.clip_param
        self.gae = GAE(self.cfg.gamma, self.cfg.GAE_lambda)
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        '''
        print("self.n_agents:", self.n_agents)         print("self.action_dim:", self.action_dim)
        self.n_agents: 1        self.action_dim: 4
        '''

        fake_input = observation_spec.zero()

        if self.cfg.priv_actor:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            actor_module = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)),
                    ["feature"], ["loc", "scale"]
                )
            )
        else:
            """ default
            actor_module=TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                [("agents", "observation")], ["loc", "scale"]
            )
            """
            actor_module=TensorDictModule(
                nn.Sequential(make_mlp([self.cfg.actor_hidden_dim, self.cfg.actor_hidden_dim]), Actor(self.action_dim)),
                [("agents", "observation")], ["loc", "scale"]
            )
        # TODO: Here is an actor net
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            """ default
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)"
            """
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([self.cfg.critic_hidden_dim, self.cfg.critic_hidden_dim]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)  #Ben: 0.1
                    nn.init.constant_(module.bias, 0.)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr_a)  #(default: Adam)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=self.cfg.lr_c)  #(default: Adam)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):

        # Prepare obs and next_obs for regularization
        obs = tensordict[("agents", "observation")]
        next_obs = tensordict[("next", "agents", "observation")]

        '''
        loc: tensor([[[...]]]),
        scale: tensor([[[...]]]),
        sample: tensor([[[...]]]),      # ≠ tensordict["action"]
        log_prob: tensor([[...]])

        print("act:", self.actor(obs))
        print("next_act:", self.actor(next_obs))
        act: (tensor([[[ 0.0363,  0.0101, -0.0738, -0.0289]],
                      [[ 0.0465, -0.0627, -0.0159, -0.1074]],
                      [[ 0.0363,  0.0101, -0.0739, -0.0287]]], device='cuda:0', grad_fn=<ViewBackward0>), 
            tensor([[[1.0075, 0.9935, 1.0031, 1.0081]],
                    [[1.0075, 0.9935, 1.0031, 1.0081]],
                    [[1.0075, 0.9935, 1.0031, 1.0081]]], device='cuda:0', grad_fn=<ExpandBackward0>), 
            tensor([[[ 0.0363,  0.0101, -0.0738, -0.0289]],
                    [[ 0.0465, -0.0627, -0.0159, -0.1074]],
                    [[ 0.0363,  0.0101, -0.0739, -0.0287]]], device='cuda:0', grad_fn=<ViewBackward0>), 
            tensor([[-3.6879],
                    [-3.6879],
                    [-3.6879]], device='cuda:0', grad_fn=<SumBackward1>))
        
        next_act: (tensor([[[ 0.0363,  0.0102, -0.0738, -0.0287]],
                           [[ 0.0465, -0.0627, -0.0159, -0.1074]],
                           [[ 0.0363,  0.0102, -0.0738, -0.0287]]], device='cuda:0', grad_fn=<ViewBackward0>), 
            tensor([[[1.0075, 0.9935, 1.0031, 1.0081]],
                    [[1.0075, 0.9935, 1.0031, 1.0081]],
                    [[1.0075, 0.9935, 1.0031, 1.0081]]], device='cuda:0', grad_fn=<ExpandBackward0>), 
            tensor([[[ 0.0363,  0.0102, -0.0738, -0.0287]],
                    [[ 0.0465, -0.0627, -0.0159, -0.1074]],
                    [[ 0.0363,  0.0102, -0.0738, -0.0287]]], device='cuda:0', grad_fn=<ViewBackward0>), 
            tensor([[-3.6879],
                    [-3.6879],
                    [-3.6879]], device='cuda:0', grad_fn=<SumBackward1>))
        '''
        '''
        print("tensordict_act:", tensordict[("agents", "action")])
        print("act:", self.actor(obs))
        tensordict_act: tensor([[[ 0.6093, -0.0344,  1.0727,  1.1755]],
                                [[-0.7148, -0.4562, -0.8892, -0.4837]],
                                [[-0.6393,  1.1645,  1.1666,  1.1457]]], device='cuda:0')
        act: (tensor([[[ 0.0505, -0.0742, -0.0090, -0.1106]],
                     [[ 0.0642, -0.0764, -0.0079, -0.1170]],
                     [[ 0.0642, -0.0765, -0.0080, -0.1171]]], device='cuda:0', grad_fn=<ViewBackward0>), 
            tensor([[[1.0079, 0.9928, 1.0037, 1.0091]],
                    [[1.0079, 0.9928, 1.0037, 1.0091]],
                    [[1.0079, 0.9928, 1.0037, 1.0091]]], device='cuda:0', grad_fn=<ExpandBackward0>), 
            tensor([[[ 0.0505, -0.0742, -0.0090, -0.1106]],
                    [[ 0.0642, -0.0764, -0.0079, -0.1170]],
                    [[ 0.0642, -0.0765, -0.0080, -0.1171]]], device='cuda:0', grad_fn=<ViewBackward0>), 
            tensor([[-3.6893],
                    [-3.6893],
                    [-3.6893]], device='cuda:0', grad_fn=<SumBackward1>))
        '''

        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        # Regularizing action policies for smooth and efficient control
        policy_loss = self.policy_regularization(self.actor, policy_loss, obs, next_obs)

        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_max_norm)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_max_norm)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

    # Regularizing action policies for smooth control
    def policy_regularization(self, actor, actor_loss, batch_obs, batch_obs_next):#, env, args):
        # Retrieving a recent set of actions:
        batch_act = actor(batch_obs)[0].clamp(-self.max_action, self.max_action)
        batch_act_next = actor(batch_obs_next)[0].clamp(-self.max_action, self.max_action)
        
        # Temporal Smoothness:
        Loss_T = F.mse_loss(batch_act, batch_act_next)

        # Spatial Smoothness:
        noise_S = (
            torch.normal(mean=0., std=0.05, size=(1, 23))).to(self.device) # mean and standard deviation
        batch_act_perturbed = actor(batch_obs + noise_S)[0].clamp(-self.max_action, self.max_action)  # Perturbed actions
        Loss_S = F.mse_loss(batch_act, batch_act_perturbed)

        # Magnitude Smoothness:
        batch_size = batch_act.shape[0]
        # f_total_hover = np.interp(4.*env.hover_force, 
        #                             [4.*env.min_force, 4.*env.max_force], 
        #                             [-args.max_action, args.max_action]
        #                 ) * torch.ones(batch_size, 1) # normalized into [-1, 1]
        # Linearly map 4 * hover_force from [4 * min_force, 4 * max_force] to [-max_action, max_action]
        
        hover_force_raw = 4.0 * self.hover_force
        min_raw = 4.0 * self.min_force
        max_raw = 4.0 * self.max_force
        # Normalize to [-1, 1]
        f_total_hover = (hover_force_raw - min_raw) / (max_raw - min_raw) * (2 * self.max_action) - self.max_action
        # Expand to batch
        f_total_hover = f_total_hover * torch.ones(batch_size, 1, device=self.device) 

        M_hover = torch.zeros(batch_size, 3).to(self.device)
        nominal_action = torch.cat([f_total_hover, M_hover], 1).to(self.device)
        nominal_action = nominal_action.unsqueeze(1)  # [B, A] -> [B, 1, A]
        Loss_M = F.mse_loss(batch_act, nominal_action)

        # Regularized actor loss for smooth control:
        regularized_actor_loss = actor_loss + self.cfg.lam_T*Loss_T + self.cfg.lam_S*Loss_S + self.cfg.lam_M*Loss_M

        """
        print("batch_act:", batch_act, "batch_act_next:", batch_act_next, "Loss_T:", Loss_T)
        print("batch_act_perturbed:", batch_act_perturbed, "Loss_S:", Loss_S)
        print("nominal_action:", nominal_action, "Loss_M:", Loss_M)
        print("actor_loss:", actor_loss, "regularized_actor_loss:", regularized_actor_loss)

        batch_act: tensor([[[-0.1586,  0.0253, -0.1985,  0.0909]],
                           [[ 0.1877, -0.0702,  0.0297,  0.0247]]], device='cuda:0', grad_fn=<ClampBackward1>) 
        batch_act_next: tensor([[[-0.1590,  0.0269, -0.1996,  0.0906]],
                                [[ 0.1886, -0.0692,  0.0255,  0.0250]]], device='cuda:0',grad_fn=<ClampBackward1>) 
        Loss_T: tensor(2.9602e-06, device='cuda:0', grad_fn=<MseLossBackward0>)
        batch_act_perturbed: tensor([[[-0.1643,  0.0236, -0.1914,  0.0889]],
                                     [[ 0.1845, -0.0716,  0.0263,  0.0248]]], device='cuda:0',grad_fn=<ClampBackward1>) 
        Loss_S: tensor(1.4232e-05, device='cuda:0', grad_fn=<MseLossBackward0>)
        nominal_action: tensor([[-0.1400,  0.0000,  0.0000,  0.0000],
                                [-0.1400,  0.0000,  0.0000,  0.0000]], device='cuda:0') 
        Loss_M: tensor(0.0203, device='cuda:0', grad_fn=<MseLossBackward0>)
        actor_loss: tensor(2.5622, device='cuda:0', grad_fn=<MulBackward0>) 
        regularized_actor_loss: tensor(2.5744, device='cuda:0', grad_fn=<AddBackward0>)
        
        batch_act: tensor([[[-0.1563,  0.0321, -0.2100,  0.0925]],
                           [[-0.1501,  0.0346, -0.2164,  0.0956]]], device='cuda:0',grad_fn=<ClampBackward1>) 
        batch_act_next: tensor([[[-0.1552,  0.0328, -0.2111,  0.0929]],
                                [[-0.1493,  0.0346, -0.2172,  0.0960]]], device='cuda:0',grad_fn=<ClampBackward1>) 
        Loss_T: tensor(5.1949e-07, device='cuda:0', grad_fn=<MseLossBackward0>)
        batch_act_perturbed: tensor([[[-0.1653,  0.0302, -0.2008,  0.0850]],
                                     [[-0.1591,  0.0322, -0.2084,  0.0887]]], device='cuda:0',grad_fn=<ClampBackward1>) 
        Loss_S: tensor(5.3440e-05, device='cuda:0', grad_fn=<MseLossBackward0>)
        nominal_action: tensor([[-0.1400,  0.0000,  0.0000,  0.0000],
                                [-0.1400,  0.0000,  0.0000,  0.0000]], device='cuda:0') 
        Loss_M: tensor(0.0139, device='cuda:0', grad_fn=<MseLossBackward0>)
        actor_loss: tensor(1.0244, device='cuda:0', grad_fn=<MulBackward0>) 
        regularized_actor_loss: tensor(1.0327, device='cuda:0', grad_fn=<AddBackward0>)
        """

        return regularized_actor_loss
    
def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
