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

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni_drones.utils.kit as kit_utils

import copy
import random
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_rotation_matrix, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni.isaac.debug_draw import _debug_draw

from .utils import attach_payload
from..inv_pendulum.utils import create_pendulum

from omni_drones.envs.utils.math_op import ensure_SO3, ensure_S2, state_normalization_payload, \
    norm_ang_btw_two_vectors, IntegralErrorVec3, IntegralError#, quaternion_to_rotation_matrix

class PayloadHover_F450(IsaacEnv):
    r"""
    An intermediate control task where a spherical payload is attached to the drone.
    The goal for the agent is to hover the payload at a target position.

    ## Observation

    - `drone_payload_rpos` (3): The position of the drone relative to the payload's position.
    - `ref_payload_rpos` (3): The reference positions of the
      payload at multiple future time steps. This helps the agent anticipate the desired payload
      trajectory.
    - `drone_state` (16 + `num_rotors`): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `payload_vels` (6): The linear and angular velocities of the payload.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for maintaining the position of the payload around the target position.
    - `up`: Reward for maintaining an upright orientation.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the drone gets too close to the ground, or when
    the payload gets too close to the ground, or when the maximum episode length
    is reached.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `bar_length`            | float | 1.0           | Length of the pendulum's bar.                                                                                                                                                                                                           |
    | `reward_distance_scale` | float | 1.6           | Scales the reward based on `target_payload_rpos`.                                                                                                                                                                                       |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """
    def __init__(self, cfg, headless):
        # self.reward_effort_weight = cfg.task.reward_effort_weight
        # self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        # self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.bar_length = cfg.task.bar_length
        self.payload_radius = cfg.task.payload_radius
        self.payload_mass = cfg.task.payload_mass
        self.drone_scale = cfg.task.drone_scale
        self.scaled_bar_length = self.bar_length / self.drone_scale
        super().__init__(cfg, headless)

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        # create and initialize additional views
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
            # reset_xform_properties=False,
        )
        self.payload.initialize()
        self.bar = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/bar",
        )
        self.bar.initialize()

        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        '''
        print('self.init_joint_pos:', self.init_joint_pos, 'self.init_joint_vels:', self.init_joint_vels)
        self.init_joint_pos:  tensor([[[0., 0., 0., 0., 0., 0.]],[[0., 0., 0., 0., 0., 0.]]], device='cuda:0') 
        self.init_joint_vels: tensor([[[0., 0., 0., 0., 0., 0.]],[[0., 0., 0., 0., 0., 0.]]], device='cuda:0')
        '''

        self.yd = self.payload_target_pos  # desired tracking position command, [m]
        self.yd_dot = torch.tensor([[0., 0., 0.]], device=self.device)  # [m/s]
        self.qd = torch.tensor([[0., 0., -1.]], device=self.device)  # desired link direction, S^2
        self.wd = torch.tensor([[0., 0., 0.]], device=self.device)  # desired link angular velocity, [rad/s]
        self.target_heading = self.b1d = torch.zeros(self.num_envs, 1, 3, device=self.device)  # desired heading direction        
        self.Wd = torch.tensor([[0., 0., 0.]], device=self.device) # desired angular velocity [rad/s]
        
        self.dt = cfg.sim.dt#0.016
        self.sat_sigma = 1.
        self.eIy = IntegralErrorVec3(self.num_envs, self.device)  # Position error integral
        self.eIb1 = IntegralError(self.num_envs, self.device)     # Yaw alignment error integral
        # Normalized integral errors
        self.eIy_norm = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.eIb1_norm = torch.zeros((self.num_envs, 1, 1), device=self.device)

        # limits of states
        self.y_lim = self.x_lim = 1.0 # [m]
        self.y_dot_lim = self.x_dot_lim = 4.0 # [m/s]
        self.w_lim = self.W_lim = 2*torch.pi # [rad/s]
        self.rp_lim = 0.1 # [rad]
        self.eIy_lim = 3.0
        self.eIb1_lim = 3.0

        # initial condition of link direction, S^2
        self.init_q = torch.tensor([[0., 0., -1.]], device=self.device)  

        self.init_pos = self.payload_target_pos - self.bar_length*self.init_q  # goal location of drones
        '''
        print("self.init_pos:", self.init_pos, "self.payload_target_pos:", self.payload_target_pos)
        self.init_pos: tensor([[0., 0., 2.]], device='cuda:0') self.payload_target_pos: tensor([[0., 0., 1.]], device='cuda:0')
        '''

        # initial conditions of "DRONES"; 50% of initial pos error
        self.init_pos_dist = D.Uniform(
            torch.tensor([-self.init_pos[0,0]*0.5, -self.init_pos[0,1]*0.5, self.init_pos[0,2]-(self.x_lim*0.5)], device=self.device),
            torch.tensor([ self.init_pos[0,0]*0.5,  self.init_pos[0,1]*0.5, self.init_pos[0,2]+(self.x_lim*0.5)], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-self.rp_lim, -self.rp_lim, 0.], device=self.device) * torch.pi,
            torch.tensor([ self.rp_lim,  self.rp_lim, 2.], device=self.device) * torch.pi
        )
        self.init_pos_dist_goal = D.Uniform(
            torch.tensor([self.init_pos[0,0]-0., self.init_pos[0,1]-0., self.init_pos[0,2]-0.], device=self.device),
            torch.tensor([self.init_pos[0,0]+0., self.init_pos[0,1]+0., self.init_pos[0,2]+0.], device=self.device)
        )
        self.init_rpy_dist_goal = D.Uniform(
            torch.tensor([-0., -0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        # randomly push the payload by applying a force
        push_force_scale = self.cfg.task.push_force_scale
        self.push_force_dist = D.Normal(
            torch.tensor([0., 0., 0.], device=self.device),
            torch.tensor(push_force_scale, device=self.device)/self.dt
        )
        push_interval = self.cfg.task.push_interval
        self.push_prob = (1 - torch.exp(-self.dt/torch.tensor(push_interval))).to(self.device)

        payload_mass_scale = self.cfg.task.payload_mass_scale
        self.payload_masses = self.payload.get_masses()
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.payload_masses[0], device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.payload_masses[0], device=self.device)
        )
        '''
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(payload_mass_scale[0] * self.drone.MASS_0[0], device=self.device),
            torch.as_tensor(payload_mass_scale[1] * self.drone.MASS_0[0], device=self.device)
        )
        print("drone.MASS_0[0]:",self.drone.MASS_0[0], "payload_mass_dist:",self.payload_mass_dist, "payload_masses:",self.payload_masses)
        drone.MASS_0[0]: tensor([1.7540], device='cuda:0') 
        payload_mass_dist: Uniform(low: tensor([0.3508], device='cuda:0'), high: tensor([0.5262], device='cuda:0')) 
        payload_masses: tensor([[0.3000],[0.3000]], device='cuda:0')
        '''
        self.rot_speed_vis = torch.full(
            (*self.drone.shape, self.drone.rotor_joint_indices.shape[-1]),
            40.0,  # or any desired value
            device=self.device
        )  # rotor spin vis [rad/s] 

        # reward coeff.
        self.reward_crash = -1. # Out of boundary or crashed!
        self.Cy = 6.0
        self.CIy = 0.1
        self.Cy_dot = 0.4
        self.Cq = 0.2
        self.Cw = 0.2
        self.Cb1 = 6.0
        self.CIb1 = 0.1
        self.CW = 0.6
        self.alpha = 0.01
        self.beta = 0.05
        # Compute reward_min for normalization
        self.reward_min = -torch.ceil(torch.tensor(
            self.Cy + self.CIy + self.Cy_dot + self.Cq + self.Cw + self.Cb1 + self.CIb1 + self.CW,
            device=self.device))

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        attach_payload(f"/World/envs/env_0/{self.drone.name}_0", self.bar_length, self.payload_radius, \
                       self.payload_mass, self.drone_scale)

        self.payload_target_pos = torch.tensor([[0., 0., 1.]], device=self.device)  # desired tracking position command, [m]
        target_vis_sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(self.payload_target_pos[0,0], self.payload_target_pos[0,1], self.payload_target_pos[0,2]),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(target_vis_sphere.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target_vis_sphere.prim_path, disable_gravity=True)

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        # obs_dim = drone_state_dim + 9

        # [ey_norm, eIy_norm, ey_dot_norm, eq_norm, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        # obs_dim = 3+3+3 + 1 + 3+9+1+1+3

        # [ey_norm, eIy_norm, ey_dot_norm, eq, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        # obs_dim = 3+3+3 + 2 + 3+9+1+1+3

        # [ey_norm, eIy_norm, ey_dot_norm, eq, q, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        obs_dim = 3+3+3 + 2+3+ 3+9+1+1+3


        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            # "action_smoothness": UnboundedContinuousTensorSpec(1),
            # "motion_smoothness": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        # Spawning at the origin position and at zero angle (w/ random linear and angular velocity).
        rand_chance = random.random()  # generate a random float between 0.0 and 1.0
        if rand_chance < 0.2: # 20% of the training
            pos = self.init_pos_dist_goal.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
            rpy = self.init_rpy_dist_goal.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
        else:
            pos = self.init_pos_dist.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
            rpy = self.init_rpy_dist.sample(env_ids.shape)#.sample((*env_ids.shape, 1))

        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)  #TODO: sample v and W from dist

        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        self.payload_masses[env_ids] = payload_mass

        target_rpy = self.target_rpy_dist.sample(env_ids.shape) #.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)

        # reset integral errors
        self.eIy.set_zero(env_ids)
        self.eIb1.set_zero(env_ids)
        self.eIy_norm[env_ids] = 0.
        self.eIb1_norm[env_ids] = 0.

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        actions = torch.clamp(actions, min = -1., max = 1.)  # clip(-1,1)
        self.spin_rotors_vis()
        self.effort = self.drone.apply_action(actions)
        self._push_payload()

    def _push_payload(self):
        env_mask = (torch.rand(self.num_envs, device=self.device) < self.push_prob).float()  # multiplies by 0 or 1 per env → only apply force to selected ones.
        forces = self.push_force_dist.sample((self.num_envs,))
        forces = (
            forces.clamp_max(self.push_force_dist.scale * 3)
            * self.payload_masses
            * env_mask.unsqueeze(-1)
        )
        self.payload.apply_forces(forces)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        '''
        print(self.drone_state)
        tensor([[[-4.7240e-03,  1.7196e+00,  1.7196e+00,                                      # self.pos
                   1.7155e-01,  6.7446e-01, 5.9057e-01, -4.0854e-01,                          # self.rot
                  -5.9773e-01,  3.9929e+00, -2.8783e+00,8.9692e+00, -1.7815e+01,  1.3727e+01, # self.vel (v, W) in the world frame
                  -3.1348e-02,  6.5645e-01, -7.5372e-01,                                      # self.heading
                  -3.4846e-01, -7.1395e-01, -6.0733e-01,                                      # self.up
                   5.4139e-02, 5.4139e-02,  5.4139e-02,  5.4139e-02                           # self.throttle * 2 - 1
                ]]
        '''
        x = self.drone_state[..., :3]
        self.quaternion = self.drone_state[..., 3:7]
        v = self.drone_state[..., 7:10]
        W_w = self.drone_state[..., 10:13]  # in the world frame
        W = quat_rotate_inverse(self.quaternion, W_w)  # in the body frame
        b1 = self.drone_state[..., 13:16]
        b3 = self.drone_state[..., 16:19]

        # Get rotation matrix
        R = quaternion_to_rotation_matrix(self.quaternion).reshape(self.quaternion.shape[:-1] + (3, 3))
        R = R.reshape(-1, 3, 3)  # Ensure shape is [N, 3, 3]
        # Check orthogonality error
        RtR = torch.matmul(R.transpose(-2, -1), R)
        identity = torch.eye(3, device=R.device).expand_as(RtR)
        error = torch.norm(RtR - identity, dim=(-2, -1)).max()
        # Repair R if needed
        if not torch.allclose(RtR, identity, atol=1e-2):
            print("Correcting R to ensure it is a valid rotation matrix (SO(3))")
            R = ensure_SO3(R)
        # Sanity checks
        assert not torch.isnan(R).any(), "NaNs in R"
        assert not torch.isinf(R).any(), "Infs in R"
        # Flatten R for downstream usage
        R_vec = R.permute(0, 2, 1).reshape(-1, 9)

        # payload states
        self.payload_pos = y = self.get_env_poses(self.payload.get_world_poses())[0]
        self.payload_vels = self.payload.get_velocities()
        y_dot = self.payload_vels[..., :3]
        w = self.payload_vels[..., 3:]   # in the world frame
        '''
        w_w = self.payload_vels[..., 3:]   # in the world frame
        w = quat_rotate_inverse(self.quaternion.squeeze(1), w_w)  # in the body frame
        '''
        self.drone_payload_rpos = q = (self.payload_pos.unsqueeze(1) - self.drone.pos)/self.bar_length
        q = ensure_S2(q) # re-normalization if needed

        state = [y, y_dot, q, w, R_vec, W]
        # print("x:",x,"y:",y)

        #######################################################################################
        # Dynamically compute desired angular velocity (Wd) from b1d and its derivative
        b3 = quat_axis(self.quaternion, axis=2).squeeze(1)  # (B, 3)
        W = self.drone_state[..., 10:13].squeeze(1)  # (B, 3)
        b3_dot = torch.cross(W, b3, dim=-1)  # (B, 3)

        b1d = self.b1d.squeeze(1)  # (B, 3)
        b1d_dot = self.b1d_dot.squeeze(1) if hasattr(self, 'b1d_dot') else torch.zeros_like(b1d)  # (B, 3)

        dot_b1d_b3 = torch.sum(b1d * b3, dim=-1, keepdim=True)
        dot_b1d_dot_b3 = torch.sum(b1d_dot * b3, dim=-1, keepdim=True)
        dot_b1d_b3_dot = torch.sum(b1d * b3_dot, dim=-1, keepdim=True)

        b1c = b1d - dot_b1d_b3 * b3
        b1c_dot = b1d_dot - dot_b1d_dot_b3 * b3 - dot_b1d_b3_dot * b3 - dot_b1d_b3 * b3_dot

        omega_c = torch.cross(b1c, b1c_dot, dim=-1)  # (B, 3)
        omega_c_3 = torch.sum(b3 * omega_c, dim=-1, keepdim=True)  # (B, 1)

        self.Wd = torch.cat([
            torch.zeros_like(omega_c_3),
            torch.zeros_like(omega_c_3),
            omega_c_3
        ], dim=-1).unsqueeze(1)  # (B, 1, 3)
        #######################################################################################

        obs = self.get_norm_error_state(state) #TODO@ben:(self.framework) 
        obs = [o.unsqueeze(1) if o.ndim == 2 else o for o in obs]
        '''
        print("obs:",obs)
        '''
        # error_obs = [ey_norm, eIy_norm, ey_dot_norm, eq, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        # error_obs = [ey_norm, eIy_norm, ey_dot_norm, eq_norm, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        self.error_obs = copy.deepcopy(obs) 

        '''
        obs = [
            self.drone_payload_rpos.flatten(1).unsqueeze(1),
            self.target_payload_rpos.flatten(1).unsqueeze(1),
            self.drone_state[..., 3:],
            self.payload_vels.unsqueeze(1), # 6
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        '''
        obs = torch.cat(obs, dim=-1)

        # self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        # self.smoothness = (
        #     self.drone.get_linear_smoothness()
        #     + self.drone.get_angular_smoothness()
        # )
        # self.stats["motion_smoothness"].lerp_(self.smoothness, (1-self.alpha))

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # error_obs = [ey_norm, eIy_norm, ey_dot_norm, eq, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        # error_obs = [ey_norm, eIy_norm, ey_dot_norm, eq_norm, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        ey_norm = self.error_obs[0]
        eIy_norm = self.error_obs[1]
        ey_dot_norm = self.error_obs[2]
        eq = self.error_obs[3]
        # eq_norm = self.error_obs[3]
        
        # ew_norm = self.error_obs[4]
        # R_vec = self.error_obs[5]
        # eb1_norm = self.error_obs[6]
        # eIb1_norm = self.error_obs[7]
        # eW_norm = self.error_obs[8]

        q = self.error_obs[4]
        ew_norm = self.error_obs[5]
        R_vec = self.error_obs[6]
        eb1_norm = self.error_obs[7]
        eIb1_norm = self.error_obs[8]
        eW_norm = self.error_obs[9]

        # Ensure all rewards have shape (batch_size, 1)
        reward_ey = -self.Cy * (torch.norm(ey_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_eIy = -self.CIy * (torch.norm(eIy_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_ey_dot = -self.Cy_dot * (torch.norm(ey_dot_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_eq = -self.Cq * (torch.norm(eq, dim=-1, keepdim=True)).squeeze(-1)  # (batch_size, 1)
        # reward_eq = -self.Cq * torch.abs(eq_norm).squeeze(-1)  # (batch_size, 1)
        reward_ew = -self.Cw * (torch.norm(ew_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_eb1 = -self.Cb1 * torch.abs(eb1_norm).squeeze(-1)  # (batch_size, 1)
        reward_eIb1 = -self.CIb1 * (torch.abs(eIb1_norm)**2).squeeze(-1)  # (batch_size, 1)
        reward_eW = -self.CW * (torch.norm(eW_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)

        assert reward_ey.shape == reward_eIy.shape == reward_ey_dot.shape == reward_eq.shape == reward_ew.shape \
            == reward_eb1.shape  == reward_eIb1.shape == reward_eW.shape
        reward = reward_ey + reward_eIy + reward_ey_dot + reward_eq + reward_ew + reward_eb1 + reward_eIb1 + reward_eW
        
        # Clamp reward to the interpolation range for stability
        reward = torch.clamp(reward, min=self.reward_min.item(), max=0.0)

        # Linearly scale reward from [reward_min, 0] → [0, 1]
        reward = (reward - self.reward_min) / (0.0 - self.reward_min)  # Equivalent to np.interp()
        '''
        print("reward:", reward)
        reward: tensor([[0.7237],
                        [0.1959],
        '''

        '''
        # pos reward
        reward_pose = 0. #torch.exp(-self.reward_distance_scale * self.target_distance)

        # uprightness
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = 0.5 / (1.0 + torch.square(tiltage))

        # effort
        # reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        # reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        # spin reward
        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))
        
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            # + reward_effort
            # + reward_action_smoothness
        )

        misbehave = (
            (self.drone.pos[..., 2] < 0.1)
            | (self.payload_pos[..., 2] < 0.1).unsqueeze(-1)
        )
        '''
        
        misbehave = (
            (ey_norm.abs() >= 1.0).any(dim=-1) 
            | (ey_dot_norm.abs() >= 1.0).any(dim=-1) 
            | (ew_norm.abs() >= 1.0).any(dim=-1) 
            | (eW_norm.abs() >= 1.0).any(dim=-1) 
            | (self.drone.pos[..., 2] < 0.2)
            | (self.payload_pos[..., 2] < 0.2).unsqueeze(-1)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # Terminal condition (Out of boundary or crashed!)
        reward[misbehave] = self.reward_crash

        self.target_distance = torch.norm(self.target_payload_rpos[:, [0]], dim=-1)
        heading_alignment = torch.abs(eb1_norm).squeeze(-1)*torch.pi

        self.stats["pos_error"].lerp_(self.target_distance, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
    
    def get_norm_error_state(self, state):
        """
        Compute normalized error observations in the ENU frame (Isaac Sim default).
        Includes position, velocity, rotation, yaw alignment, and angular velocity errors.
        All inputs and outputs are assumed to follow ENU convention:
            - X: East
            - Y: North
            - Z: Up

        Returns:
            error_obs: list of tensors:
                [ex_norm, eIx_norm, ev_norm, R_vec, eb1_norm, eIb1_norm, eW_norm]
        """
        # 1. Normalize state vectors: [y, y_dot, q, w, R_vec, W]
        y_norm, y_dot_norm, q, w_norm, R_vec, W_norm = state_normalization_payload(
            state, self.y_lim, self.y_dot_lim, self.w_lim, self.W_lim
        )

        # 2. Normalize desired (goal) states
        yd_norm = self.yd / self.y_lim
        yd_dot_norm = self.yd_dot / self.y_dot_lim
        wd_norm = self.wd / self.w_lim
        Wd_norm = self.Wd / self.W_lim

        # 3. Compute normalized tracking errors
        ey_norm = y_norm - yd_norm
        ey_dot_norm = y_dot_norm - yd_dot_norm
        # eq_norm = norm_ang_btw_two_vectors(self.qd, q) # [rad]  
        eq = torch.cross(self.qd.expand_as(q), q, dim=-1)[..., :2] 
        '''
        print("q:",q, "qd:",self.qd, "eq_norm:",eq_norm)
        q: tensor([[[-0.1189,  0.9692,  0.2155]],[[-0.7928, -0.2633, -0.5496]]], device='cuda:0') 
        qd: tensor([[ 0.,  0., -1.]], device='cuda:0') 
        eq_norm: tensor([[[0.5691]],[[0.3148]]], device='cuda:0')
        '''
        ew_norm = w_norm - wd_norm
        eW_norm = W_norm - Wd_norm

        # 4. Extract body axes from quaternion (ENU world frame)
        b1 = quat_axis(self.quaternion, axis=0)  # forward
        b2 = quat_axis(self.quaternion, axis=1)  # left
        b3 = quat_axis(self.quaternion, axis=2)  # up

        # 5. Compute yaw (heading) alignment error
        dot_b1d_b3 = torch.einsum('bnd,bnd->bn', self.b1d, b3).unsqueeze(-1)  # shape (B, 1, 1)
        b1c = self.b1d - dot_b1d_b3 * b3  # desired heading projected into horizontal plane

        numerator = torch.einsum('bnd,bnd->bn', b1c, b2).unsqueeze(-1)
        denominator = torch.einsum('bnd,bnd->bn', b1c, b1).unsqueeze(-1)
        eb1 = torch.atan2(-numerator, denominator)  # heading angle error
        eb1_norm = eb1 / torch.pi

        # 6. Update and normalize integral position error
        self.target_payload_rpos = self.payload_pos.unsqueeze(1) - self.payload_target_pos 
        '''
        print("self.target_payload_rpos:",self.target_payload_rpos, "ey:",ey_norm * self.y_lim)
        target_payload_rpos: tensor([[[ 0.0000e+00, -2.5534e-10, -3.5254e-01]],[[ 0.0000e+00, -2.5534e-10, -3.5254e-01]]], device='cuda:0') 
        ey: tensor([[ 0.0000e+00, -2.5534e-10, -3.5254e-01],[ 0.0000e+00, -2.5534e-10, -3.5254e-01]], device='cuda:0')
        '''
        ey = ey_norm * self.y_lim
        eIy_integrand = -self.alpha * self.eIy.error + ey.unsqueeze(1)
        self.eIy.integrate(eIy_integrand, self.dt)
        self.eIy_norm = torch.clamp(
            self.eIy.error / self.eIy_lim,
            min=-self.sat_sigma,
            max=self.sat_sigma
        )

        # 7. Update and normalize integral heading error
        eIb1_integrand = -self.beta * self.eIb1.error + eb1
        self.eIb1.integrate(eIb1_integrand, self.dt)
        self.eIb1_norm = torch.clamp(
            self.eIb1.error / self.eIb1_lim,
            min=-self.sat_sigma,
            max=self.sat_sigma
        )

        # 8. Return full error observation
        error_obs = [
            ey_norm,
            self.eIy_norm,
            ey_dot_norm,
            eq, #eq_norm,
            q,
            ew_norm,
            R_vec.unsqueeze(1),  # (B, 1, 9)
            eb1_norm,
            self.eIb1_norm,
            eW_norm
        ]

        return error_obs
    
    def spin_rotors_vis(self):
        # Get joint velocities from the drone
        self.dof_vel = self.drone.get_joint_velocities()

        # Generate random rotational speeds for each rotor
        num_rotors = self.drone.rotor_joint_indices.shape[-1]

        # Alternate CW/CCW spin directions
        spin_directions = torch.tensor(
            [1 if i % 2 == 0 else -1 for i in range(num_rotors)],
            device=self.device
        ).view(1, -1).expand(*self.drone.shape, -1)

        prop_rot = self.rot_speed_vis * spin_directions

        # Apply to the rotor DOFs
        self.dof_vel[..., self.drone.rotor_joint_indices] = prop_rot
        self.drone.set_joint_velocities(self.dof_vel)