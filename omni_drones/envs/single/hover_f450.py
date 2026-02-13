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

import copy
import math
import random
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_rotation_matrix, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

from omni_drones.envs.utils.math_op import hat, vee, deriv_unit_vector, ensure_SO3, state_normalization, \
    IntegralErrorVec3, IntegralError#, quaternion_to_rotation_matrix

def attach_payload(parent_path):
    from omni.isaac.core import objects
    import omni.physx.scripts.utils as script_utils
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, .15]),
        mass=0.0001
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.)


class Hover_F450(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to maintain a stable
    position and heading in mid-air without drifting. This task is designed
    to serve as a sanity check.

    ## Observation

    The observation space consists of the following part:

    - `rpos` (3): The position relative to the target hovering position.
    - `drone_state` (16 + `num_rotors`): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `rheading` (3): The difference between the reference heading and the current heading.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional vector encoding the current
      progress of the episode.
    'observation': torch.Size([1, 1, 30])}

    ## Reward

    - `pos`: Reward computed from the position error to the target position.
    - `heading_alignment`: Reward computed from the alignment of the heading to the target heading.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the drone misbehaves, i.e., it crashes into the ground or flies too far away:

    ```{math}
        d_\text{pos} > 4 \text{ or } x^w_z < 0.2
    ```

    or when the episode reaches the maximum length.

    ## Config

    | Parameter               | Type  | Default   | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the drone and its target.                                                                                                                                                               |
    | `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    | `has_payload`           | bool  | False     | Indicates whether the drone has a payload attached. If set to True, it means that a payload is attached; otherwise, if set to False, no payload is attached.                                                                            |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()

        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.rot_speed_vis = torch.full(
            (*self.drone.shape, self.drone.rotor_joint_indices.shape[-1]),
            40.0,  # or any desired value
            device=self.device
        )  # rotor spin vis [rad/s] 

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())  #TODO: sample v and W from dist

        self.target_pos = self.xd = torch.tensor([[0., 0., 2.]], device=self.device)  # desired tracking position command, [m]
        self.target_heading = self.b1d = torch.zeros(self.num_envs, 1, 3, device=self.device)  # desired heading direction        
        self.vd = torch.tensor([[0., 0., 0.]], device=self.device)  # [m/s]
        self.Wd = torch.tensor([[0., 0., 0.]], device=self.device) # desired angular velocity [rad/s]
        self.alpha = 0.8

        self.dt = cfg.sim.dt#0.016
        self.sat_sigma = 1.
        self.eIx = IntegralErrorVec3(self.num_envs, self.device)  # Position error integral
        self.eIb1 = IntegralError(self.num_envs, self.device)     # Yaw alignment error integral
        # Normalized integral errors
        self.eIx_norm = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.eIb1_norm = torch.zeros((self.num_envs, 1, 1), device=self.device)

        # Limits of states:
        self.x_lim = 1.0 # [m]
        self.v_lim = 4.0 # [m/s]
        self.W_lim = 2*torch.pi # [rad/s]
        self.rp_lim = 0.2 # [rad]
        self.eIx_lim  = 3.0 
        self.eIb1_lim = 3.0 

        self.init_pos_dist = D.Uniform(
            torch.tensor([-self.x_lim*0.5, -self.x_lim*0.5, self.target_pos[0,-1]-(1.*0.5)], device=self.device),
            torch.tensor([ self.x_lim*0.5,  self.x_lim*0.5, self.target_pos[0,-1]+(1.*0.5)], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-self.rp_lim, -self.rp_lim, 0.], device=self.device) * torch.pi,
            torch.tensor([ self.rp_lim,  self.rp_lim, 2.], device=self.device) * torch.pi
        )
        self.init_pos_dist_goal = D.Uniform(
            torch.tensor([self.target_pos[0,0]-0., self.target_pos[0,1]-0., self.target_pos[0,2]-0.], device=self.device),
            torch.tensor([self.target_pos[0,0]+0., self.target_pos[0,1]+0., self.target_pos[0,2]+0.], device=self.device)
        )
        self.init_rpy_dist_goal = D.Uniform(
            torch.tensor([-0., -0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )

        # reward coeff.
        self.reward_crash = -1. # Out of boundary or crashed!
        self.Cx = 6.0
        self.CIx = 0.1
        self.Cv = 0.4
        self.Cb1 = 6.0
        self.CIb1 = 0.1
        self.CW = 0.6
        self.rwd_alpha = 0.01
        self.rwd_beta = 0.05
        # Compute reward_min for normalization
        self.reward_min = -torch.ceil(torch.tensor(
            self.Cx + self.CIx + self.Cv + self.Cb1 + self.CIb1 + self.CW,
            device=self.device 
        ))

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 2.),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(),
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):  # This method to specify the input and output of the environment. It should at least include `observation_spec` and `action_spec`.
        drone_state_dim = self.drone.state_spec.shape[-1]  # drone_state_dim = 23, including pos(3), quat(4), linvel(3), angvel(3), headingvec(3), upvec(3), thr(4=quadrotor)
        '''
        observation_dim = drone_state_dim + 3  # rheading (3): The difference between the reference heading and the current heading.
        '''
        observation_dim = 23
        
        action_dim = 4
        """-------------------------------------------------------------------------------------------------------------
        | Agents  | Observations                    | obs_dim | Actions      | act_dim | Rewards                       |
        | single  | {ex, eIx, ev, R, eb1, eIb1, eW} | 23      | {f_total, M} | 4       | f(ex, eIx, ev, eb1, eIb1, eW) |
        -------------------------------------------------------------------------------------------------------------"""

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((1, action_dim), device=self.device),  #self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
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
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):  # This method to reset sub-environment instances given by `env_ids`.
        self.drone._reset_idx(env_ids, self.training)
        
        # Spawning at the origin position and at zero angle (w/ random linear and angular velocity).
        rand_chance = random.random()  # generate a random float between 0.0 and 1.0
        if rand_chance < 0.2: # 20% of the training
            pos = self.init_pos_dist_goal.sample((*env_ids.shape, 1))
            rpy = self.init_rpy_dist_goal.sample((*env_ids.shape, 1))
        else:
            pos = self.init_pos_dist.sample((*env_ids.shape, 1))
            rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids, joint_indices=joint_indices)

            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

        # print(self.drone.get_joint_positions(True), self.drone.get_joint_velocities())

        # reset integral errors
        self.eIx.set_zero(env_ids)
        self.eIb1.set_zero(env_ids)
        self.eIx_norm[env_ids] = 0.
        self.eIb1_norm[env_ids] = 0.

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):  # This method to apply the agents’ actions.
        actions = tensordict[("agents", "action")]
        actions = torch.clamp(actions, min = -1., max = 1.)  # clip(-1,1)
        self.spin_rotors_vis()
        self.effort = self.drone.apply_action(actions)
        
        '''
        print(self.effort)  # == self.throttle.sum(-1), used in the reward function
            tensor([[2.7447]], device='cuda:0')
            tensor([[2.2201]], device='cuda:0')
            tensor([[2.0617]], device='cuda:0')
            tensor([[2.4442]], device='cuda:0')
            tensor([[2.2107]], device='cuda:0')
        '''
        """
        # Test the geometric controller
        goal_position = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        goal_yaw_angle = torch.tensor([0.0], device=self.device)
        goal_yaw_direction = torch.tensor([torch.cos(goal_yaw_angle), torch.sin(goal_yaw_angle), 0], device=self.device)
        drone_body_position, drone_body_orientation = self.get_drone_body_pos_rot()
        actions = self.geometric_controller(drone_body_position, drone_body_orientation, goal_position, goal_yaw_direction)
        print(actions)
        #self.effort = self.drone.apply_action(actions)  # self.apply_forces_and_torques()
        """

    def _compute_state_and_obs(self):  # This method to compute the state and observation for the transition step.
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

        # self.heading[:] = quat_axis(self.rot, axis=0)
        # self.up[:] = quat_axis(self.rot, axis=2)
        # state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]
        '''
        x = self.drone_state[..., :3]
        self.q = self.drone_state[..., 3:7]
        v = self.drone_state[..., 7:10]
        W_w = self.drone_state[..., 10:13]  # in the world frame
        W = quat_rotate_inverse(self.q, W_w)  # in the body frame
        '''
        print("W_w:", W_w, "W:", W)
        W_w: tensor([[[ 1.8409,  0.8441, -1.2413]]], device='cuda:0') 
        W: tensor([[[ 0.0502, -2.3642, -0.2245]]], device='cuda:0')
        '''
        ###########################################################
        # Normalize quaternion to prevent instability
        self.q = self.q / self.q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Get rotation matrix
        R = quaternion_to_rotation_matrix(self.q).reshape(self.q.shape[:-1] + (3, 3))
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

        ###########################################################
        b1 = self.drone_state[..., 13:16]
        b3 = self.drone_state[..., 16:19]
        # throttle = self.drone_state[..., 19:]
        state = [x, v, R_vec, W]

        """-------------------------------------------------------------------------------------------------------------
        | Agents  | Observations                    | obs_dim | Actions      | act_dim | Rewards                       |
        | single  | {ex, eIx, ev, R, eb1, eIb1, eW} | 23      | {f_total, M} | 4       | f(ex, eIx, ev, eb1, eIb1, eW) |
        -------------------------------------------------------------------------------------------------------------"""
        # relative position and heading
        # self.rpos = self.target_pos - self.drone_state[..., :3]
        # self.rheading = self.target_heading - self.drone_state[..., 13:16]

        ###########################################################
        # --- Dynamically compute desired angular velocity (Wd) from b1d and its derivative ---
        b3 = quat_axis(self.q, axis=2).squeeze(1)  # (B, 3)
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

        ###########################################################

        obs = self.get_norm_error_state(state) #TODO@ben:(self.framework) 
        self.error_obs = copy.deepcopy(obs) # [ex_norm, self.eIx_norm, ev_norm, R_vec.unsqueeze(1), eb1_norm, self.eIb1_norm, eW_norm]

        #obs = [self.rpos, self.drone_state[..., 3:], self.rheading,]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
            """
            print(t.expand(-1, self.time_encoding_dim).unsqueeze(1))  # if self.time_encoding
                tensor([[[0.0780, 0.0780, 0.0780, 0.0780]]], device='cuda:0')
                tensor([[[0.0800, 0.0800, 0.0800, 0.0800]]], device='cuda:0')
                tensor([[[0.,     0.,     0.,     0.]]], device='cuda:0')  # New episode
                tensor([[[0.0020, 0.0020, 0.0020, 0.0020]]], device='cuda:0')
                tensor([[[0.0040, 0.0040, 0.0040, 0.0040]]], device='cuda:0')
                tensor([[[0.0060, 0.0060, 0.0060, 0.0060]]], device='cuda:0')
            """
        obs = torch.cat(obs, dim=-1)

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

    def _compute_reward_and_done(self):  # This method to compute the reward and termination flags for the transition step.
        #self.error_obs = [ex_norm, self.eIx_norm, ev_norm, R_vec.unsqueeze(1), eb1_norm, self.eIb1_norm, eW_norm]
        '''
        print("self.error_obs:", self.error_obs)
        [tensor([[[ 0.3106, -0.2540,  0.4745]],
                 [[-0.9534,  0.1782,  0.0586]],
                 [[-1.0321, -0.9352, -0.2108]]], device='cuda:0'), 
            tensor([[[ 0.0183, -0.0138,  0.0705]],
                    [[-0.0282,  0.0055,  0.0018]],
                    [[-0.0772, -0.1986, -0.0317]]], device='cuda:0'), 
            tensor([[[ 0.2943, -0.2503,  0.0588]],
                    [[ 0.0542, -0.0719, -0.0276]],
                    [[-0.6428,  0.0509, -0.1927]]], device='cuda:0'), 
            tensor([[[-0.6070, -0.7935, -0.0435,  0.7096, -0.5166, -0.4791,  0.3577, -0.3217,  0.8767]],
                    [[-0.1475, -0.9303, -0.3357,  0.9453, -0.0327, -0.3246,  0.2910, -0.3652,  0.8843]],
                    [[ 0.7533,  0.6410, -0.1474, -0.5674,  0.5199, -0.6386, -0.3327, 0.5647,  0.7553]]], device='cuda:0'), 
            tensor([[[ 0.2444]],
                    [[-0.9313]],
                    [[-0.0273]]], device='cuda:0'), 
            tensor([[[ 0.4046]],
                    [[-0.0857]],
                    [[ 0.4008]]], device='cuda:0'), 
            tensor([[[ 0.0551,  0.0566, -0.0741]],
                    [[-0.0319,  0.0186,  0.0090]],
                    [[-0.1220, -0.0514, -0.2298]]], device='cuda:0')]
        '''

        ex_norm = self.error_obs[0]
        eIx_norm = self.error_obs[1]
        ev_norm = self.error_obs[2]
        R_vec = self.error_obs[3]
        eb1_norm = self.error_obs[4]
        eIb1_norm = self.error_obs[5]
        eW_norm = self.error_obs[6]

        # Ensure all rewards have shape (batch_size, 1)
        reward_eX = -self.Cx * (torch.norm(ex_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_eIX = -self.CIx * (torch.norm(eIx_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_eV = -self.Cv * (torch.norm(ev_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)
        reward_eb1 = -self.Cb1 * torch.abs(eb1_norm).squeeze(-1)  # (batch_size, 1)
        reward_eIb1 = -self.CIb1 * (torch.abs(eIb1_norm)**2).squeeze(-1)  # (batch_size, 1)
        reward_eW = -self.CW * (torch.norm(eW_norm, dim=-1, keepdim=True) ** 2).squeeze(-1)  # (batch_size, 1)

        assert reward_eX.shape == reward_eIX.shape == reward_eV.shape == reward_eb1.shape  == reward_eIb1.shape == reward_eW.shape
        reward = reward_eX + reward_eIX + reward_eV + reward_eb1 + reward_eIb1 + reward_eW
        
        # Clamp reward to the interpolation range for stability
        reward = torch.clamp(reward, min=self.reward_min.item(), max=0.0)

        # Linearly scale reward from [reward_min, 0] → [0, 1]
        reward = (reward - self.reward_min) / (0.0 - self.reward_min)  # Equivalent to np.interp()
        '''
        print("reward:", reward)
        reward: tensor([[0.7237],
                        [0.1959],
                        [0.1205]], device='cuda:0')
        '''

        # pose reward
        
        # pos_error = torch.norm(self.rpos, dim=-1)
        # heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)
        # distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)
        # misbehave = (self.drone.pos[..., 2] < 0.2) | (distance > 4) | (ev_norm.abs() >= 1.0).any(dim=-1) | (eW_norm.abs() >= 1.0).any(dim=-1)
        '''
        print("self.rpos:", self.rpos, "self.rheading:", self.rheading, "distance:", distance)
        self.rpos: tensor([[[ 0.7938, -0.8114, -0.3564]],
                            [[ 0.5842,  0.4034,  0.9982]],
                            [[-0.2099,  0.9195,  0.8527]]], device='cuda:0') 
        self.rheading: tensor([[[-0.7553,  1.8000, -0.3045]],
                                [[-1.6507, -0.3297, -0.1090]],
                                [[ 1.4378, -1.2975,  0.3501]]], device='cuda:0') 
        distance: tensor([[2.3062],
                        [2.0847],
                        [2.3431]], device='cuda:0')
        '''

        '''
        reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # pose_reward = torch.exp(-distance * self.reward_distance_scale)
        # uprightness
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # spin reward
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # effort
        # reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        print(reward_pose.shape, reward_up.shape, reward_spin.shape)
        
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            # + reward_effort
            + reward_action_smoothness
        )
        '''

        pos_error = (torch.norm(ex_norm, dim=-1, keepdim=True)).squeeze(-1)*self.x_lim
        heading_alignment = torch.abs(eb1_norm).squeeze(-1)*torch.pi
        '''
        print("pos_error:", pos_error, "heading_alignment:", heading_alignment)
        pos_error: tensor([[0.6214],
                           [0.9716],
                           [1.4086]], device='cuda:0') 
        heading_alignment: tensor([[0.7677],
                                   [2.9259],
                                   [0.0857]], device='cuda:0')

        '''

        misbehave = (
            (ex_norm.abs() >= 1.0).any(dim=-1) 
            | (ev_norm.abs() >= 1.0).any(dim=-1) 
            | (eW_norm.abs() >= 1.0).any(dim=-1) 
            |(self.drone.pos[..., 2] < 0.2)
        )
        '''
        print("misbehave:", misbehave)
        misbehave: tensor([[False],
                           [False],
                           [ True]], device='cuda:0')
        '''

        hasnan = torch.isnan(self.drone_state).any(-1)
        
        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # Terminal condition (Out of boundary or crashed!)
        reward[misbehave] = self.reward_crash
        '''
        print("reward:", reward)
        reward: tensor([[ 0.7237],
                        [ 0.1959],
                        [-1.0000]], device='cuda:0')
        '''

        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        # self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1-self.alpha))
        # self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["return"] += reward
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
        # 1. Normalize state vectors: [x, v, R_vec, W]
        x_norm, v_norm, R_vec, W_norm = state_normalization(
            state, self.x_lim, self.v_lim, self.W_lim
        )

        # 2. Normalize desired (goal) states
        xd_norm = self.xd / self.x_lim
        vd_norm = self.vd / self.v_lim
        Wd_norm = self.Wd / self.W_lim

        # 3. Compute normalized tracking errors
        ex_norm = x_norm - xd_norm
        ev_norm = v_norm - vd_norm
        eW_norm = W_norm - Wd_norm

        # 4. Extract body axes from quaternion (ENU world frame)
        b1 = quat_axis(self.q, axis=0)  # forward
        b2 = quat_axis(self.q, axis=1)  # left
        b3 = quat_axis(self.q, axis=2)  # up

        # 5. Compute yaw (heading) alignment error
        dot_b1d_b3 = torch.einsum('bnd,bnd->bn', self.b1d, b3).unsqueeze(-1)  # shape (B, 1, 1)
        b1c = self.b1d - dot_b1d_b3 * b3  # desired heading projected into horizontal plane

        numerator = torch.einsum('bnd,bnd->bn', b1c, b2).unsqueeze(-1)
        denominator = torch.einsum('bnd,bnd->bn', b1c, b1).unsqueeze(-1)
        eb1 = torch.atan2(-numerator, denominator)  # heading angle error
        eb1_norm = eb1 / torch.pi

        # 6. Update and normalize integral position error
        ex = ex_norm * self.x_lim
        eIx_integrand = -self.rwd_alpha * self.eIx.error + ex
        self.eIx.integrate(eIx_integrand, self.dt)
        self.eIx_norm = torch.clamp(
            self.eIx.error / self.eIx_lim,
            min=-self.sat_sigma,
            max=self.sat_sigma
        )

        # 7. Update and normalize integral heading error
        eIb1_integrand = -self.rwd_beta * self.eIb1.error + eb1
        self.eIb1.integrate(eIb1_integrand, self.dt)
        self.eIb1_norm = torch.clamp(
            self.eIb1.error / self.eIb1_lim,
            min=-self.sat_sigma,
            max=self.sat_sigma
        )

        # 8. Return full error observation
        error_obs = [
            ex_norm,
            self.eIx_norm,
            ev_norm,
            R_vec.unsqueeze(1),  # (B, 1, 9)
            eb1_norm,
            self.eIb1_norm,
            eW_norm
        ]

        return error_obs



        """
        ##################################################################################
        self.g = 9.81 # standard gravity
        # self.dt = 1/60
        # Isaac Sim uses ENU frame, but NED frame is used in FDCL.
        # Note that ENU and the NED here refer to their direction order.
        # ENU: E - axis 1, N - axis 2, U - axis 3
        # NED: N - axis 1, E - axis 2, D - axis 3
        self.theta = 1.
        pi_tensor, cos, sin = torch.tensor(math.pi), torch.cos, torch.sin
        # Transformation matrices
        self.R_fw = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, cos(pi_tensor), -sin(pi_tensor)],
            [0.0, sin(pi_tensor),  cos(pi_tensor)]
        ], device=self.device)  # Transformation from Isaac-world-fixed frame(ENU) to the FDCL-fixed frame(NED)
        self.R_bl = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, cos(pi_tensor), -sin(pi_tensor)],
            [0.0, sin(pi_tensor),  cos(pi_tensor)]
        ], device=self.device)  # Transformation from Isaac-local-body frame(ENU) to the FDCL-body frame(NED)

        #------ UAVs' properties ------#
        self.crazyflie_x0 = torch.tensor([0., 0., 0.1], device=self.device)
        self.scale_uav = 7
        self.m_uav = (0.025 + 4*0.0008)*self.scale_uav
        self.d = 0.0438*self.scale_uav
        self.J = torch.diag(torch.tensor([1.4e-5, 1.4e-5, 2.17e-5], device=self.device) * self.scale_uav)  # inertia matrix
        self.c_tf = 0.0135 # torque-to-thrust coefficients
        self.c_tw = 2.25 # thrust-to-weight coefficients
        self.hover_force = self.m_uav * self.g / 4.0 # thrust magnitude of each motor, [N]
        self.min_force = -0.005 # minimum thrust of each motor, [N]
        self.max_force = self.c_tw * self.hover_force # maximum thrust of each motor, [N]
        self.prop_max_rot = 433.3
        self.motor_assymetry = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        self.motor_assymetry = self.motor_assymetry * 4.0 / torch.sum(self.motor_assymetry)
        self.fM = torch.zeros((4, 1), device=self.device) # Force-moment vector
        # Force-moment conversion matrix
        fM_to_forces = torch.tensor([
            [1.0,  1.0,  1.0,  1.0],
            [self.d*cos(pi_tensor / 4),  self.d*cos(pi_tensor / 4), -self.d * cos(pi_tensor / 4), -self.d * cos(pi_tensor / 4)],
            [self.d*sin(pi_tensor / 4), -self.d*sin(pi_tensor / 4), -self.d * sin(pi_tensor / 4),  self.d * sin(pi_tensor / 4)],
            [-self.c_tf,           self.c_tf,            -self.c_tf,            self.c_tf]
        ], device=self.device)
        self.fM_to_forces_inv = torch.linalg.inv(fM_to_forces)

        #---- PD control gains ----#
        self.kX = 0.3 * torch.diag(torch.tensor([0.4, 0.4, 0.6], device=self.device))  # position gains  
        self.kV = 3.0 * torch.diag(torch.tensor([0.2, 0.2, 0.4], device=self.device))  # velocity gains 
        self.kX[2, 2] = 0.4

        self.kR = 0.15 * torch.diag(torch.tensor([1.5, 1.5, 2.0], device=self.device))  # attitude gains 
        self.kW = 0.15 * torch.diag(torch.tensor([0.8, 0.8, 1.5], device=self.device))  # angular velocity gains 

        #---- Integral control ----#
        # self.use_integral = False
        # self.sat_sigma = 3.5
        # self.eIX = IntegralErrorVec3() # Position integral error
        # self.eIR = IntegralErrorVec3() # Attitude integral error
        # self.eIX.set_zero() # Set all integrals to zero
        # self.eIR.set_zero()
        # #---- I control gains ----#
        # self.kIX = 0.5 * torch.diag(torch.tensor([1.0, 1.0, 1.4]))  # Position integral gains
        # self.kIR = 0.05 * torch.diag(torch.tensor([1.0, 1.0, 0.7]))  # Attitude integral gain

    def get_drone_body_pos_rot(self):
        drone_body_position, drone_body_orientation = self.drone.base_link.get_world_poses(clone=False)
        # print("Drone's position is : " + str(drone_body_position.view(-1)))
        # print("Drone's orientation is : " + str(drone_body_orientation.view(-1)))
        return drone_body_position.view(-1), drone_body_orientation.view(-1)
    
    def geometric_controller(self, uav_position, uav_orientation, goal_position, goal_yaw_direction):
        #---- States ----#
        x_w = uav_position # Pos in Isaac-world-fixed frame(ENU)
        # v_w = self.drone.base_link.get_velocities(clone=False)[:, :3][0] # Vel in Isaac-world-fixed frame(ENU) 
        v_w = self.drone.base_link.get_velocities(clone=False).view(-1)[:3] # Vel in Isaac-world-fixed frame(ENU) 
        R_wl = ensure_SO3(quaternion_to_rotation_matrix(uav_orientation)) # Isaac-local-body frame(ENU) to Isaac-world-fixed frame(ENU)
        # W_l = self.drone.base_link.get_velocities(clone=False)[:, 3:][0] # Ang Vel in Isaac-world-local frame(ENU)
        W_l = self.drone.base_link.get_velocities(clone=False).view(-1)[3:] # Ang Vel in Isaac-world-local frame(ENU)
        W_l = R_wl.T@W_l
        #---- Desired commands ----#
        xd_w = goal_position # Goal Pos in Isaac-world-fixed frame(ENU)
        xd_dot  = torch.zeros(3, device=self.device)
        xd_2dot = torch.zeros(3, device=self.device)
        xd_3dot = torch.zeros(3, device=self.device)
        xd_4dot = torch.zeros(3, device=self.device)

        b1d_l = goal_yaw_direction
        b1d_dot  = torch.zeros(3, device=self.device)
        b1d_2dot = torch.zeros(3, device=self.device)

        #----
        x_f = self.R_fw@x_w # Pos in FDCL-fixed frame(NED)
        v_f = self.R_fw@v_w # Vel in FDCL-fixed frame(NED)
        R_fb = self.R_fw@R_wl@self.R_bl.T # Att in FDCL-body frame(NED)
        # R_fb = self.R_fw@R_wl
        W_b = self.R_bl@W_l # Ang Vel in FDCL-body frame(NED)

        b1d_b = self.R_bl@b1d_l # Heading Cmd in FDCL-body frame(NED)
        xd_f = self.R_fw@xd_w # Pos Cmd in FDCL-fixed frame(NED)
        # print(f"x_f: {x_f}, xd_f: {xd_f}, v_f: {v_f}, R_fb: {R_fb}, W_b: {W_b}")

        e3 = torch.tensor([0., 0., 1.], device=self.device)
        R_T = R_fb.T
        hatW = hat(W_b)
        #---- Position control ----#
        # Translational error functions
        eX = x_f - xd_f     # position tracking errors 
        eV = v_f - xd_dot # velocity tracking errors 
        
        #---- Position integral terms
        # if self.use_integral:
        #     self.eIX.integrate(eX + eV, self.dt) 
        #     self.eIX.error = torch.clamp(self.eIX.error, -self.sat_sigma, self.sat_sigma)
        # else:
        #     self.eIX.set_zero()

        #---- Force 'f' along negative b3-axis ----#
        # This term equals to R_fb.e3
        A = - self.kX@eX \
            - self.kV@eV \
            - self.m_uav*self.g*e3 \
            + self.m_uav*xd_2dot 
        # if self.use_integral:
        #     A -= self.kIX@self.eIX.error

        b3 = R_fb@e3
        b3_dot = R_fb@hatW@e3
        f_total = -A@b3

        #---- Intermediate terms for rotational errors ----#
        ea = self.g*e3 \
            - f_total/self.m_uav*b3 \
            - xd_2dot
        A_dot = - self.kX@eV \
                - self.kV@ea \
                + self.m_uav*xd_3dot  

        f_dot = - A_dot@b3 \
                - A@b3_dot
        eb = - f_dot/self.m_uav*b3 \
                - f_total/self.m_uav*b3_dot \
                - xd_3dot
        A_2dot = - self.kX@ea \
                    - self.kV@eb \
                    + self.m_uav*xd_4dot
        
        b3c, b3c_dot, b3c_2dot = deriv_unit_vector(-A, -A_dot, -A_2dot)

        hat_b1d = hat(b1d_b)
        hat_b1d_dot = hat(b1d_dot)
        hat_b2d_dot = hat(b1d_2dot)

        A2 = -hat_b1d@b3c
        A2_dot = - hat_b1d_dot@b3c - hat_b1d@b3c_dot
        A2_2dot = - hat_b2d_dot@b3c \
                    - 2.0*hat_b1d_dot@b3c_dot \
                    - hat_b1d@b3c_2dot

        b2c, b2c_dot, b2c_2dot = deriv_unit_vector(A2, A2_dot, A2_2dot)

        hat_b2c = hat(b2c)
        hat_b2c_dot = hat(b2c_dot)
        hat_b2c_2dot = hat(b2c_2dot)

        b1c = hat_b2c@b3c
        b1c_dot = hat_b2c_dot@b3c + hat_b2c@b3c_dot
        b1c_2dot = hat_b2c_2dot@b3c \
                    + 2.0*hat_b2c_dot@b3c_dot \
                    + hat_b2c@b3c_2dot

        Rd = torch.vstack((b1c, b2c, b3c)).T
        Rd_dot = torch.vstack((b1c_dot, b2c_dot, b3c_dot)).T
        Rd_2dot = torch.vstack((b1c_2dot, b2c_2dot, b3c_2dot)).T

        Rd_T = Rd.T
        Wd = vee(Rd_T@Rd_dot)

        hat_Wd = hat(Wd)
        Wd_dot = vee(Rd_T@Rd_2dot - hat_Wd@hat_Wd)
        
        #---- Attitude control ----#
        RdtR = Rd_T@R_fb
        eR = 0.5*vee(RdtR - RdtR.T) # attitude error vector
        eW = W_b - R_T@Rd@Wd # angular velocity error vector

        #---- Attitude integral terms ----#
        # if self.use_integral:
        #     self.eIR.integrate(eR + eW, self.dt) 
        #     self.eIR.error = torch.clamp(self.eIR.error, -self.sat_sigma, self.sat_sigma)
        # else:
        #     self.eIR.set_zero()

        M = - self.kR@eR \
            - self.kW@eW \
            + hat(R_T@Rd@Wd)@self.J@R_T@Rd@Wd \
            + self.J@R_T@Rd@Wd_dot
        # if self.use_integral:
        #     M -= self.kIR@self.eIR.error

        #---- Print error terms ----#    
        # print(f"eX: {eX}, eV: {eV}, eR: {eR}, eW: {eW}")

        #---- Compute the thrust of each motor from the total force and moment ----#
        f_total = torch.clamp(f_total, -self.c_tw * self.m_uav * self.g, self.c_tw * self.m_uav * self.g)
        self.fM[0] = f_total
        for i in range(3):
            self.fM[i + 1] = M[i]
        f_motor = (self.fM_to_forces_inv@self.fM).flatten()
        f_motor = torch.clamp(self.fM_to_forces_inv @ self.fM, -self.max_force, self.min_force).flatten()

        self.f_b = torch.tensor([0, 0, f_total], device=self.device)
        self.f_l = self.R_bl.T@self.f_b
        self.f_w = R_wl@self.f_l 
        self.f_w = R_wl@self.f_b

        self.M_b = M
        self.M_l = self.R_bl.T@self.M_b
        self.M_w = R_wl@self.M_l
        # self.M_w = R_wl@self.M_b

        # self.M_b = M
        # R_cb = np.array([
        #     [1.,  0.,  0.],
        #     [0., cos(pi),-sin(pi)],
        #     [0., sin(pi), cos(pi)]
        # ], device=self.device) 
        # self.M_c = R_cb@self.M_b
        # R_lc = np.array([
        #     [cos(pi/4),-sin(pi/4),  0.],
        #     [sin(pi/4), cos(pi/4),  0.],
        #     [0.,  0., 1.]
        # ]) 
        # self.M_l = R_lc@self.M_c
        # self.M_w = R_wl@self.M_l

        # M_w = np.array([0.0, 0.001, 0.0], device=self.device)
        # print(f"f_b: {self.f_b}, f_l: {self.f_l}, f_w: {self.f_w}")
        # print(f"M_b: {self.M_b}, M_c: {self.M_c}, M_l: {self.M_l}, M_w: {self.M_w}")
        """