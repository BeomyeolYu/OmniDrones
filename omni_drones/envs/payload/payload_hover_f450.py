import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni_drones.utils.kit as kit_utils

import os
import gc
import copy
import random
import numpy as np
import torch
import torch.distributions as D
from scipy.spatial.transform import Rotation as R

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_rotation_matrix, quat_rotate_inverse, rotation_matrix_to_quaternion, quat_rotate

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni.isaac.debug_draw import _debug_draw

from .utils import attach_payload

from omni_drones.envs.utils.math_op import ensure_SO3, ensure_S2, state_normalization_payload, \
    norm_ang_btw_two_vectors, IntegralErrorVec3, IntegralError#, quaternion_to_rotation_matrix


class PayloadHover_F450(IsaacEnv):
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.trajectory_collection = cfg.task.trajectory_collection
        self.domain_adaptation = cfg.task.domain_adaptation
        self.fine_tuning = cfg.task.fine_tuning
        self.real_deployment = cfg.task.real_deployment
        
        self.inverted_pendulum = cfg.task.inverted_pendulum
        self.bar_length = cfg.task.bar_length
        self.payload_radius = cfg.task.payload_radius
        self.payload_mass = cfg.task.payload_mass
        self.drone_scale = cfg.task.drone_scale
        self.scaled_bar_length = self.bar_length / self.drone_scale
        self.fictitious_force_gravity_ratio = cfg.task.fictitious_force_gravity_ratio

        self.rwd_k_exp = cfg.task.rwd_gains.rwd_k_exp
        self.Cy = cfg.task.rwd_gains.Cy
        self.Cy_dot = cfg.task.rwd_gains.Cy_dot
        self.Cq = cfg.task.rwd_gains.Cq
        self.Cw = cfg.task.rwd_gains.Cw
        self.Cb1 = cfg.task.rwd_gains.Cb1
        self.CR = cfg.task.rwd_gains.CR
        self.CW = cfg.task.rwd_gains.CW
        self.CIy = cfg.task.rwd_gains.CIy
        self.CIb1 = cfg.task.rwd_gains.CIb1
        self.rwd_alpha = cfg.task.rwd_gains.rwd_alpha
        self.rwd_beta = cfg.task.rwd_gains.rwd_beta
        super().__init__(cfg, headless)

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])
                
        # create and initialize additional views
        self.payload = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/payload",
            reset_xform_properties=False
        )
        self.payload.initialize()
        self.bar = RigidPrimView(
            f"/World/envs/env_*/{self.drone.name}_*/bar",
            reset_xform_properties=False
        )
        self.bar.initialize()

        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        '''
        print('self.init_joint_pos:', self.init_joint_pos, 'self.init_joint_vels:', self.init_joint_vels)
        ['rotor_0_joint', 'rotor_1_joint', 'rotor_2_joint', 'rotor_3_joint', 'D6Joint:0', 'D6Joint:1']
        self.init_joint_pos:  tensor([[[0., 0., 0., 0., 0., 0.]],[[0., 0., 0., 0., 0., 0.]]], device='cuda:0') 
        self.init_joint_vels: tensor([[[0., 0., 0., 0., 0., 0.]],[[0., 0., 0., 0., 0., 0.]]], device='cuda:0')
        '''

        # =================================================================================
        if self.domain_adaptation:
            self.real_states = None
            self.real_actions = None
            self.real_next_states = None
            self.real_dones = None
            self.trajectory_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.eval_trajectory_idx = None
            self.timestep_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.Rd = torch.zeros(self.num_envs, 3, 3, device=self.device)

            # Desired states will now be dynamic, loaded from real data
            self.yd = torch.zeros(self.num_envs, 1, 3, device=self.device)  # = self.payload_target_pos desired tracking position command, [m]
            self.yd_dot = torch.zeros(self.num_envs, 1, 3, device=self.device)  # [m/s]
            self.qd = torch.zeros(self.num_envs, 1, 3, device=self.device)  # desired link direction, S^2
            self.wd = torch.zeros(self.num_envs, 1, 3, device=self.device)  # desired link angular velocity, [rad/s]
            self.b1d = torch.zeros(self.num_envs, 1, 3, device=self.device)  # = self.target_heading desired heading direction       
            self.Wd = torch.zeros(self.num_envs, 1, 3, device=self.device)  # desired angular velocity [rad/s]
        else:
            self.yd = self.payload_target_pos  # desired tracking position command, [m]
            self.yd_dot = torch.tensor([[0., 0., 0.]], device=self.device)  # [m/s]
            # self.qd = torch.tensor([[0., 0., -1.]], device=self.device)  # desired link direction, S^2
            if self.inverted_pendulum:
                self.qd = torch.tensor([[0., 0., 1.]], device=self.device)
            else:
                self.qd = torch.tensor([[0., 0., -1.]], device=self.device)
            self.wd = torch.tensor([[0., 0., 0.]], device=self.device)  # desired link angular velocity, [rad/s]
            self.target_heading = self.b1d = torch.zeros(self.num_envs, 1, 3, device=self.device)  # desired heading direction        
            self.Wd = torch.tensor([[0., 0., 0.]], device=self.device) # desired angular velocity [rad/s]
        # =================================================================================

        self.dt = cfg.sim.dt#0.016
        self.sat_sigma = 1.
        self.eIy = IntegralErrorVec3(self.num_envs, self.device)  # Position error integral
        self.eIb1 = IntegralError(self.num_envs, self.device)     # Yaw alignment error integral
        # Normalized integral errors
        self.eIy_norm = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.eIb1_norm = torch.zeros((self.num_envs, 1, 1), device=self.device)

        # limits of states
        self.y_lim = self.x_lim = 1.0 # [m]
        self.y_dot_lim = self.x_dot_lim = 3.0 #4.0 # [m/s]
        self.w_lim = torch.pi # 2*torch.pi [rad/s]
        self.W_lim = 2*torch.pi # [rad/s]
        self.rp_lim = 0.1 # [rad]
        self.eIy_lim = 2.0 #3.0
        self.eIb1_lim = 2.5 #3.0
        if self.inverted_pendulum:
            self.y_dot_lim = 4.0
            self.w_lim = 2 * torch.pi
            self.rp_lim = 0.05 #0.1 # [rad]
            self.eIy_lim = 3.0

        # initial condition of link direction, S^2
        if self.inverted_pendulum:
            self.init_q = torch.tensor([[0., 0., 1.]], device=self.device)  
        else:
            self.init_q = torch.tensor([[0., 0., -1.]], device=self.device)

        self.init_pos = self.payload_target_pos - self.bar_length*self.init_q  # goal location of drones
        '''
        print("self.init_pos:", self.init_pos, "self.payload_target_pos:", self.payload_target_pos)
        self.init_pos: tensor([[0., 0., 2.]], device='cuda:0') self.payload_target_pos: tensor([[0., 0., 1.]], device='cuda:0')
        '''
        
        # initial conditions of "DRONES"; training mode
        pos_xy_offset, pos_z_offset = self.x_lim*0.3, self.x_lim*0.3  # define the position randomization range
        self.init_pos_dist = D.Uniform(
            torch.tensor([self.init_pos[0,0]-pos_xy_offset, self.init_pos[0,1]-pos_xy_offset, self.init_pos[0,2]-pos_z_offset], device=self.device),
            torch.tensor([self.init_pos[0,0]+pos_xy_offset, self.init_pos[0,1]+pos_xy_offset, self.init_pos[0,2]+pos_z_offset], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-self.rp_lim, -self.rp_lim, 0.], device=self.device) * torch.pi,
            torch.tensor([ self.rp_lim,  self.rp_lim, 2.], device=self.device) * torch.pi
        )
        if self.trajectory_collection:  # for data collection:
            # # EASY: 130000 trajs
            # self.init_rpy_dist = D.Uniform(
            #     torch.tensor([-self.rp_lim, -self.rp_lim, 0.], device=self.device) * torch.pi,
            #     torch.tensor([ self.rp_lim,  self.rp_lim, 1.], device=self.device) * torch.pi
            # )
            ## HARD: 40000 trajs
            self.init_rpy_dist = D.Uniform(
                torch.tensor([-0.15, -0.15, 0.], device=self.device) * torch.pi,
                torch.tensor([ 0.15,  0.15, 2.], device=self.device) * torch.pi
            )
            if self.inverted_pendulum:
                self.init_rpy_dist = D.Uniform(
                    torch.tensor([-0.06, -0.06, 0.], device=self.device) * torch.pi,
                    torch.tensor([ 0.06,  0.06, 2.], device=self.device) * torch.pi
                )

        if self.real_deployment:
            if self.inverted_pendulum:
                # pos_xy_offset, pos_z_offset = self.x_lim*0.4, self.x_lim*0.4  # define the position randomization range
                # self.init_pos_dist = D.Uniform(
                #     torch.tensor([self.init_pos[0,0]-pos_xy_offset, self.init_pos[0,1]-pos_xy_offset, self.init_pos[0,2]-pos_z_offset], device=self.device),
                #     torch.tensor([self.init_pos[0,0]+pos_xy_offset, self.init_pos[0,1]+pos_xy_offset, self.init_pos[0,2]+pos_z_offset], device=self.device)
                # )
                # self.init_rpy_dist = D.Uniform(
                #     torch.tensor([-0.07, -0.07, 0.], device=self.device) * torch.pi,
                #     torch.tensor([ 0.07,  0.07, 2.], device=self.device) * torch.pi
                # )
                pos_xy_offset, pos_z_offset = self.x_lim*0.3, self.x_lim*0.4  # define the position randomization range
                self.init_pos_dist = D.Uniform(
                    torch.tensor([self.init_pos[0,0]-pos_xy_offset, self.init_pos[0,1]-pos_xy_offset, self.init_pos[0,2]-pos_z_offset], device=self.device),
                    torch.tensor([self.init_pos[0,0]+pos_xy_offset, self.init_pos[0,1]+pos_xy_offset, self.init_pos[0,2]+pos_z_offset], device=self.device)
                )
                self.init_rpy_dist = D.Uniform(
                    torch.tensor([-0.07, -0.07, 0.], device=self.device) * torch.pi,
                    torch.tensor([ 0.07,  0.07, 2.], device=self.device) * torch.pi
                )
            else:
                pos_xy_offset, pos_z_offset = self.x_lim*0.2, self.x_lim*0.6  # define the position randomization range
                self.init_pos_dist = D.Uniform(
                    torch.tensor([self.init_pos[0,0]-pos_xy_offset, self.init_pos[0,1]-pos_xy_offset, self.init_pos[0,2]-pos_z_offset], device=self.device),
                    torch.tensor([self.init_pos[0,0]+pos_xy_offset, self.init_pos[0,1]+pos_xy_offset, self.init_pos[0,2]+pos_z_offset], device=self.device)
                )
                self.init_rpy_dist = D.Uniform(
                    torch.tensor([-0.2, -0.2, 0.], device=self.device) * torch.pi,
                    torch.tensor([ 0.2,  0.2, 0.5], device=self.device) * torch.pi
                )

        self.init_pos_dist_goal = D.Uniform(
            torch.tensor([self.init_pos[0,0]-0., self.init_pos[0,1]-0., self.init_pos[0,2]-0.], device=self.device),
            torch.tensor([self.init_pos[0,0]+0., self.init_pos[0,1]+0., self.init_pos[0,2]+0.], device=self.device)
        )
        self.init_rpy_dist_goal = D.Uniform(
            torch.tensor([-0., -0., 0.], device=self.device) * torch.pi,
            torch.tensor([ 0.,  0., 2.], device=self.device) * torch.pi
        )
        # self.target_rpy_dist = D.Uniform(
        #     torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
        #     torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        # )
        # Distribution for target yaw *delta* (e.g., +/- 45 deg)
        delta_yaw_rad = torch.pi / 4.0 # 45 degrees
        self.target_delta_yaw_dist = D.Uniform(
            torch.tensor(-delta_yaw_rad, device=self.device),
            torch.tensor(delta_yaw_rad, device=self.device)
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
        self.reward_min = -torch.ceil(torch.tensor(
            self.Cy + self.CIy + self.Cy_dot + self.Cq + self.Cw + self.Cb1 + self.CIb1 + self.CW,
            device=self.device)) # Compute reward_min for normalization

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

        attach_payload(f"/World/envs/env_0/{self.drone.name}_0", self.inverted_pendulum, self.bar_length, \
                       self.payload_radius, self.payload_mass, self.drone_scale)

        if self.inverted_pendulum:
            self.payload_target_pos = torch.tensor([[0., 0., 3.]], device=self.device)  # desired tracking position command, [m]
        else:
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

    def _reset_idx(self, env_ids: torch.Tensor):
        # print(f"env_ids: {env_ids}")

        if self.domain_adaptation:
            # ==============================================================================
            if self.real_states is None:
                raise ValueError("Real-world data has not been loaded. Call set_real_world_data().")

            self.drone._reset_idx(env_ids)
            
            # Randomly select a trajectory index FOR EACH environment
            self.trajectory_indices[env_ids] = torch.randint(0, self.num_real_trajectories, (len(env_ids),), device=self.device)
            if self.eval_trajectory_idx is not None:
                self.trajectory_indices[env_ids] = self.eval_trajectory_idx
                # print(f"self.eval_trajectory_idx:{self.eval_trajectory_idx}")
            # Get the corresponding t=0 state for each environment
            self.timestep_indices[env_ids] = 0
            state = self.real_states[self.trajectory_indices[env_ids], self.timestep_indices[env_ids]]
            # print(f"state:{state}")
            
            # Apply the t=0 state to the simulator (drone and joints)
            y, y_dot, q, w, R_vec, W_b = self.unpack_state_vector(state)

            # print(f"Initial States (y) in `_reset_idx`: {y}")
            # print(f"Initial States (y_dot) in `_reset_idx`: {y_dot}")
            # print(f"Initial States (q) in `_reset_idx`: {q}")
            # print(f"Initial States (w) in `_reset_idx`: {w}")
            # print(f"Initial States (R_vec) in `_reset_idx`: {R_vec}")
            # print(f"Initial States (W_b) in `_reset_idx`: {W_b}")
            # print("----------------------------------------\n")

            w = w - (w * q).sum(dim=-1, keepdim=True) * q
            R_wb = ensure_SO3(R_vec.view(-1, 3, 3))
            R_bw = R_wb.transpose(1, 2) # body <- world
            quat_wb = rotation_matrix_to_quaternion(R_wb)
            x = y - self.bar_length * q
            x_dot = y_dot - self.bar_length * torch.cross(w, q, dim=-1)
            W_world = quat_rotate(quat_wb, W_b)
            
            self.drone.set_world_poses(x + self.envs_positions[env_ids], quat_wb, env_ids)
            self.drone.set_velocities(torch.cat([x_dot, W_world], dim=-1), env_ids)

            # --- Compute R_bar (bar orientation in world frame) ---
            z_axis = q if self.inverted_pendulum else -q
            z_axis = z_axis / torch.norm(z_axis, dim=-1, keepdim=True)

            x_drone_world = R_wb[:, :, 0]
            proj = x_drone_world - (x_drone_world * z_axis).sum(dim=-1, keepdim=True) * z_axis
            x_axis = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
            y_axis = torch.cross(z_axis, x_axis, dim=-1)
            R_bar = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # [N, 3, 3]
            
            # --- Compute joint_pos: Euler angles from drone to bar ---
            R_rel = torch.einsum("bij,bjk->bik", R_bw, R_bar)  # bar ← drone
            """
            Compute the logarithm map from SO(3) to so(3), returning axis-angle vectors.
            Input: R [B, 3, 3]
            Output: log_R [B, 3]
            """
            cos_theta = ((R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]) - 1) / 2
            theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0)).unsqueeze(-1)  # [B, 1]

            # skew = (R_rel - R_rel.transpose(1, 2)) / (2 * torch.sin(theta) + 1e-6)  # [B, 3, 3]
            skew = (R_rel - R_rel.transpose(1, 2)) / (2 * torch.sin(theta) + 1e-6).unsqueeze(-1) # [B, 3, 3]
            axis = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)  # [B, 3]
            joint_pos = theta * axis * -1  # [B, 3] negative due to PhysX convention
            # The following is incorrect
            # R_rel = torch.einsum("bij,bjk->bik", R_wb.transpose(1, 2), R_bar)  # R_bw @ R_bar
            # euler = R.from_matrix(R_rel.cpu().numpy()).as_euler("xyz", degrees=False)
            # joint_pos = torch.tensor(euler, dtype=torch.float32, device=q.device) *-1
            
            # --- D6 Joint Velocity (`joint_vel`) Calculation ---

            # Step 1: Compute R_dot_bar and R_dot_wb (drone)
            # ω_world is angular velocity in world frame
            w_hat = self.skew_matrix_from_vector(w)
            W_hat = self.skew_matrix_from_vector(W_world)
            ''' Shomehow these show the same y_dot and omega:
            w_parallel = (w * q).sum(dim=-1, keepdim=True) * q
            W_world_parallel = (W_world * q).sum(dim=-1, keepdim=True) * q
            w_swing_world = w - w_parallel
            W_swing_world = W_world - W_world_parallel
            w_hat = skew_matrix_from_vector(w_swing_world)
            W_hat = skew_matrix_from_vector(W_swing_world)
            '''

            '''
            If your angular velocity ω is in the world frame (ω_world):
            The formula is Ṙ = hat(ω_world) @ R (pre-multiply by the skew matrix).

            If your angular velocity ω is in the body frame (ω_body):
            The formula is Ṙ = R @ hat(ω_body) (post-multiply by the skew matrix).
            '''
            R_dot_bar = torch.bmm(w_hat, R_bar)       # 𝑅̇_bar = [ω]_x * R_bar
            R_dot_wb = torch.bmm(W_hat, R_wb)         # _wb = [Ω]_x * R_wb

            # Step 2: Compute R_rel = R_bw * R_bar and Ṙ_rel
            R_rel = torch.bmm(R_bw, R_bar)
            R_dot_bw = -torch.bmm(R_bw, torch.bmm(R_dot_wb, R_bw))  # d/dt(Rᵀ) = -Rᵀ Ṙ Rᵀ
            R_dot_rel = torch.bmm(R_dot_bw, R_bar) + torch.bmm(R_bw, R_dot_bar)

            # Step 3: Compute ω_rel in drone body frame
            omega_rel_hat = torch.bmm(R_rel.transpose(1, 2), R_dot_rel)
            omega_rel_body = self.vee(omega_rel_hat)  # Extract ω from skew-symmetric matrix
            joint_vel = -omega_rel_body 
            
            # ------------------------ Apply to Simulation ------------------------
            # joint_indices = torch.tensor([
            #     self.drone._view._dof_indices["SphericalJoint:0"],
            #     self.drone._view._dof_indices["SphericalJoint:1"],
            #     self.drone._view._dof_indices["SphericalJoint:2"],
            # ], device=self.device)

            joint_indices = torch.tensor([
                self.drone._view._dof_indices["D6Joint:0"],
                self.drone._view._dof_indices["D6Joint:1"],
                self.drone._view._dof_indices["D6Joint:2"],
            ], device=self.device)
            
            # Set the joint positions and velocities
            self.drone._view.set_joint_positions(joint_pos, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(joint_vel, env_indices=env_ids, joint_indices=joint_indices)
            # =====================================================================================
            
            # Apply one simulation step to register changes
            # self.sim.step()
        else:
            self.drone._reset_idx(env_ids)

            # Spawning at the origin position and at zero angle (w/ random linear and angular velocity).
            if self.trajectory_collection is False:
                rand_chance = random.random()  # generate a random float between 0.0 and 1.0
                if rand_chance < 0.8: # 80% of the training
                    pos = self.init_pos_dist.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
                    rpy = self.init_rpy_dist.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
                else: # 20% of the training
                    pos = self.init_pos_dist_goal.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
                    rpy = self.init_rpy_dist_goal.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
            else:
                rand_chance = random.random()  # generate a random float between 0.0 and 1.0
                if rand_chance < 0.99: # 99% of trajectory collection
                    pos = self.init_pos_dist.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
                    rpy = self.init_rpy_dist.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
                else: # 1% of trajectory collection
                    pos = self.init_pos_dist_goal.sample(env_ids.shape)#.sample((*env_ids.shape, 1))
                    rpy = self.init_rpy_dist_goal.sample(env_ids.shape)#.sample((*env_ids.shape, 1))

            if self.real_deployment:
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

            # target_rpy = self.target_rpy_dist.sample(env_ids.shape) #.sample((*env_ids.shape, 1))
            # --- NEW YAW TARGET LOGIC ---
            # Get the initial yaw (shape [N, 1])
            initial_yaw = rpy[..., 2].unsqueeze(-1) 
            
            # Sample a yaw *delta* (shape [N, 1])
            delta_yaw = self.target_delta_yaw_dist.sample(env_ids.shape).to(self.device).unsqueeze(-1)

            # Target yaw = initial + delta. (No need to wrap, atan2 handles it)
            target_yaw = initial_yaw + delta_yaw 
            
            # Create target RPY tensor: [0, 0, target_yaw]
            target_rpy = torch.cat([
                torch.zeros_like(target_yaw), 
                torch.zeros_like(target_yaw), 
                target_yaw
            ], dim=-1)
            
            target_rot = euler_to_quaternion(target_rpy)
            self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)

        # ------------------------ Additional Reset Operations ------------------------
        # Sample random payload masses and apply to the corresponding envs
        payload_mass = self.payload_mass_dist.sample(env_ids.shape)
        self.payload.set_masses(payload_mass, env_ids)
        self.payload_masses[env_ids] = payload_mass
        # print(self.drone.base_link.get_masses().clone())

        self.eIy.set_zero(env_ids)
        self.eIb1.set_zero(env_ids)
        self.eIy_norm[env_ids] = 0.
        self.eIb1_norm[env_ids] = 0.
        self.stats[env_ids] = 0.

    def skew_matrix_from_vector(self, v):
        # Helper functions for D6 Joint Velocity (`joint_vel`) Calculation
        B = v.shape[0]
        O = torch.zeros(B, device=v.device)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        return torch.stack([
            torch.stack([O, -z, y], dim=-1),
            torch.stack([z, O, -x], dim=-1),
            torch.stack([-y, x, O], dim=-1)
        ], dim=1)

    def vee(self, mat):
        # Correct vee operator for skew-symmetric matrix to vector (so(3) -> ℝ³)
        return torch.stack([
            mat[:, 2, 1] - mat[:, 1, 2],
            mat[:, 0, 2] - mat[:, 2, 0],
            mat[:, 1, 0] - mat[:, 0, 1]
        ], dim=-1) * 0.5

    def load_data_from_chunk(self, filepath: str):
        """
        Loads all trajectory data from a single .npz chunk file into memory.
        This function handles the necessary padding and tensor conversion.
        """
        if not os.path.exists(filepath):
            print(f"Warning: Data chunk not found at {filepath}. Skipping.")
            return

        # --- ROBUST FIX: Check if the attribute exists AND is not None ---
        if hasattr(self, 'real_states') and self.real_states is not None:
            # Delete the references to the large tensors
            del self.real_states
            del self.real_actions
            del self.real_next_states
            del self.real_dones
            # Force Python's garbage collector to run
            gc.collect()
            # Force PyTorch to clear any cached memory on the GPU
            torch.cuda.empty_cache()
        # ----------------------------------------------------------------

        print(f"\nLoading new data chunk from: {os.path.basename(filepath)}")
        
        try:
            with np.load(filepath, allow_pickle=True) as data:
                # Directly use the numpy arrays from the file
                states_data = data["states"]
                actions_data = data["actions"]
                next_states_data = data["next_states"]
                dones_data = data["dones"]

            self.num_real_trajectories = len(states_data)
            if self.num_real_trajectories == 0:
                print("Warning: Loaded chunk contains no trajectories.")
                return
            
            # Find max trajectory length *in this specific chunk* for padding
            self.real_trajectory_length = max(len(traj) for traj in states_data)
            
            # Create padded numpy arrays
            padded_states = np.zeros((self.num_real_trajectories, self.real_trajectory_length, states_data[0].shape[1]), dtype=np.float32)
            for i, traj in enumerate(states_data): padded_states[i, :len(traj)] = traj
            
            padded_actions = np.zeros((self.num_real_trajectories, self.real_trajectory_length, actions_data[0].shape[1]), dtype=np.float32)
            for i, traj in enumerate(actions_data): padded_actions[i, :len(traj)] = traj

            padded_next_states = np.zeros_like(padded_states)
            for i, traj in enumerate(next_states_data): padded_next_states[i, :len(traj)] = traj

            padded_dones = np.zeros((self.num_real_trajectories, self.real_trajectory_length), dtype=np.int8)
            for i, traj_dones in enumerate(dones_data):
                if traj_dones.ndim == 0: traj_dones = [traj_dones]
                padded_dones[i, :len(traj_dones)] = traj_dones
            
            # Convert to PyTorch tensors and store them in the environment
            self.real_states = torch.from_numpy(padded_states).float().to(self.device)
            self.real_actions = torch.from_numpy(padded_actions).float().to(self.device)
            self.real_next_states = torch.from_numpy(padded_next_states).float().to(self.device)
            self.real_dones = torch.from_numpy(padded_dones).to(self.device)
            
            print(f"Successfully loaded {self.num_real_trajectories} trajectories into memory.")

        except Exception as e:
            print(f"Error processing chunk {os.path.basename(filepath)}: {e}")

    def set_real_world_data(self, real_data: dict, eval_trajectory_idx: bool):
        self.eval_trajectory_idx = eval_trajectory_idx

        # The 'states' array is now a 1D object array containing other arrays
        states_data = real_data['states']
        
        self.num_real_trajectories = states_data.shape[0]

        # Since trajectories have different lengths, find the maximum length.
        # This will be used for setting episode truncation boundaries.
        if self.num_real_trajectories > 0:
            self.real_trajectory_length = max(len(traj) for traj in states_data)
        else:
            self.real_trajectory_length = 0
        
        # We now need to convert the list of arrays into a tensor for the GPU.
        # This requires padding, which is best handled here.
        padded_states = np.zeros(
            (self.num_real_trajectories, self.real_trajectory_length, states_data[0].shape[1]),
            dtype=np.float32
        )
        for i, traj in enumerate(states_data):
            padded_states[i, :len(traj)] = traj

        # The rest of the data should also be padded for consistency
        padded_next_states = np.zeros_like(padded_states)
        next_states_data = real_data['next_states']
        for i, traj in enumerate(next_states_data):
            padded_next_states[i, :len(traj)] = traj
            
        padded_actions = np.zeros(
            (self.num_real_trajectories, self.real_trajectory_length, real_data['actions'][0].shape[1]),
            dtype=np.float32
        )
        actions_data = real_data['actions']
        for i, traj in enumerate(actions_data):
            padded_actions[i, :len(traj)] = traj

        # Convert the now-padded numpy arrays to tensors
        self.real_states = torch.from_numpy(padded_states).float().to(self.device)
        self.real_next_states = torch.from_numpy(padded_next_states).float().to(self.device)
        self.real_actions = torch.from_numpy(padded_actions).float().to(self.device)
        
        # --- NEW: Load and pad the 'dones' array ---
        dones_data = real_data['dones']
        padded_dones = np.zeros(
            (self.num_real_trajectories, self.real_trajectory_length), # Shape (N, T)
            dtype=np.int8
        )
        for i, traj_dones in enumerate(dones_data):
            # The last element should have a length of 1 if it's just a single int
            if traj_dones.ndim == 0:
                traj_dones = [traj_dones]
            padded_dones[i, :len(traj_dones)] = traj_dones
            
        self.real_dones = torch.from_numpy(padded_dones).to(self.device)
        # --- END NEW ---

        print(f"Loaded {self.num_real_trajectories} trajectories of max length {self.real_trajectory_length}.")

    def update_stats(self):
        # This function is called at the end of _compute_state_and_obs

        ey_norm, eIy_norm, ey_dot_norm, eq, q, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm = [t.squeeze(1) for t in self.error_obs]

        # Use keepdim=True to maintain shape for lerp_
        self.stats["ey"].lerp_(torch.norm(ey_norm * self.y_lim, dim=-1, keepdim=True), (1 - self.alpha))
        # self.stats["ey_norm"].lerp_(torch.norm(ey_norm, dim=-1, keepdim=True), (1 - self.alpha))
        self.stats["eIy"].lerp_(torch.norm(eIy_norm * self.eIy_lim, dim=-1, keepdim=True), (1 - self.alpha))
        # self.stats["eIy_norm"].lerp_(torch.norm(eIy_norm, dim=-1, keepdim=True), (1 - self.alpha))
        self.stats["ey_dot"].lerp_(torch.norm(ey_dot_norm * self.y_dot_lim, dim=-1, keepdim=True), (1 - self.alpha))
        # self.stats["ey_dot_norm"].lerp_(torch.norm(ey_dot_norm, dim=-1, keepdim=True), (1 - self.alpha))
        self.stats["eq"].lerp_(torch.norm(eq, dim=-1, keepdim=True), (1 - self.alpha))
        self.stats["ew"].lerp_(torch.norm(ew_norm * self.w_lim, dim=-1, keepdim=True), (1 - self.alpha))
        # self.stats["ew_norm"].lerp_(torch.norm(ew_norm, dim=-1, keepdim=True), (1 - self.alpha))
        self.stats["eb1"].lerp_(torch.abs(eb1_norm * torch.pi), (1 - self.alpha))
        # self.stats["eb1_norm"].lerp_(torch.abs(eb1_norm), (1 - self.alpha))
        self.stats["eIb1"].lerp_(torch.abs(eIb1_norm * self.eIb1_lim), (1 - self.alpha))
        # self.stats["eIb1_norm"].lerp_(torch.abs(eIb1_norm), (1 - self.alpha))
        self.stats["eW"].lerp_(torch.norm(eW_norm * self.W_lim, dim=-1, keepdim=True), (1 - self.alpha))
        # self.stats["eW_norm"].lerp_(torch.norm(eW_norm, dim=-1, keepdim=True), (1 - self.alpha))

        if self.domain_adaptation:            
            self.stats["eR"].lerp_(torch.norm(self.eR, dim=-1, keepdim=True).squeeze(-1), (1 - self.alpha))

    def unpack_state_vector(self, state_vec, add_agent_dim=False):
        parts = [state_vec[..., i] for i in [[0,1,2],[3,4,5],[6,7,8],[9,10,11],list(range(12,21)),[21,22,23]]]
        if add_agent_dim: return [p.unsqueeze(1) for p in parts]
        return parts

    def _pre_sim_step(self, tensordict: TensorDictBase):
        # Get the local Center of Mass position (e.g., [0.02, 0, 0])
        # This is the 'r_com' vector, with shape [N, 1, 3]
        # We get it from the drone's intrinsics, which are set during randomization
        local_com_pos = self.drone.intrinsics["com"]

        if self.domain_adaptation:
            # --- Step 3: Delta Action Training Logic ---
            delta_action_6d = tensordict[("agents", "action")]
            delta_controls_4d = delta_action_6d[..., :4]
            fictitious_forces_norm_2d = delta_action_6d[..., 4:]

            real_action_4d = self.real_actions[self.trajectory_indices, self.timestep_indices].unsqueeze(1)
            corrected_action_4d = torch.clamp(real_action_4d + delta_controls_4d, min=-1., max=1.)

            # 1. Calculate motor forces and torques (no physics applied yet)
            motor_forces, motor_torques = self.drone.apply_action(corrected_action_4d)
            
            # 2. Scale fictitious forces (broadcasts [N,1,3] * [N,1,1])
            max_fictitious_force = self.drone.masses * 9.81 * self.fictitious_force_gravity_ratio
            scaled_fictitious_forces_2d = fictitious_forces_norm_2d * max_fictitious_force

            # --- NEW: Pad 2D forces to 3D, setting Z=0 ---
            scaled_fictitious_forces_3d = torch.nn.functional.pad(
                scaled_fictitious_forces_2d, (0, 1), 'constant', 0.0
            )
            # ----------------------------------------------
            
            # 3. Combine the motor forces and the fictitious forces.
            total_forces = motor_forces + scaled_fictitious_forces_3d.reshape(motor_forces.shape)

            # 4. Make a SINGLE call to the physics engine
            self.drone.base_link.apply_forces_and_torques_at_pos(
                forces=total_forces.reshape(-1, 3), # Reshape for API
                torques=motor_torques.reshape(-1, 3), # Reshape for API
                is_global=False 
            )
            self.effort = motor_forces.sum(-1)

        elif self.fine_tuning:
            # --- Step 4: Fine-Tuning Logic (Corrected) ---
            action_6d = tensordict[("agents", "action")]
            corrected_controls_4d = action_6d[..., :4] 
            fictitious_forces_norm_2d = action_6d[..., 4:]

            # 1. Calculate motor forces and torques
            motor_forces, motor_torques = self.drone.apply_action(corrected_controls_4d)
            
            # 2. Scale fictitious forces (broadcasts [N,1,3] * [N,1,1])
            max_fictitious_force = self.drone.masses * 9.81 * self.fictitious_force_gravity_ratio
            scaled_fictitious_forces_2d = fictitious_forces_norm_2d * max_fictitious_force

            # --- NEW: Pad 2D forces to 3D, setting Z=0 ---
            scaled_fictitious_forces_3d = torch.nn.functional.pad(
                scaled_fictitious_forces_2d, (0, 1), 'constant', 0.0
            )
            # ----------------------------------------------
            
            # 3. Combine the motor forces and the fictitious forces.
            total_forces = motor_forces + scaled_fictitious_forces_3d.reshape(motor_forces.shape)

            # 4. Make a SINGLE call to the physics engine
            self.drone.base_link.apply_forces_and_torques_at_pos(
                forces=total_forces.reshape(-1, 3), # Reshape for API
                torques=motor_torques.reshape(-1, 3), # Reshape for API
                is_global=False 
            )
            self.effort = motor_forces.sum(-1)
            
        else:
            # --- Standard 4D Action Logic (Original) ---
            actions = tensordict[("agents", "action")]
            actions = torch.clamp(actions, min = -1., max = 1.)
            
            # 1. Calculate forces and torques
            motor_forces, motor_torques = self.drone.apply_action(actions)
            
            # 2. Make a SINGLE call to the physics engine
            self.drone.base_link.apply_forces_and_torques_at_pos(
                forces=motor_forces.reshape(-1, 3),
                torques=motor_torques.reshape(-1, 3),
                is_global=False
            )
            self.effort = motor_forces.sum(-1)

    # def _pre_sim_step(self, tensordict: TensorDictBase):
    #     if self.domain_adaptation:

    #         # --- 7D Delta Action Logic (Corrected) ---
    #         delta_action_7d = tensordict[("agents", "action")]
    #         delta_controls_4d = delta_action_7d[..., :4]
    #         fictitious_forces_norm_3d = delta_action_7d[..., 4:]

    #         real_action_4d = self.real_actions[self.trajectory_indices, self.timestep_indices].unsqueeze(1)
    #         corrected_action_4d = torch.clamp(real_action_4d + delta_controls_4d, min=-1., max=1.)

    #         # 1. Calculate the forces and torques from the motor commands.
    #         motor_forces, motor_torques = self.drone.apply_action(corrected_action_4d)
            
    #         # 2. Scale the fictitious forces.
    #         # Scale fictitious forces relative to a physical quantity the drone must always fight against: gravity. 
    #         # The unmodeled forces like drag are typically a fraction of the gravitational force.
    #         # A value of 0.3 would mean you expect the maximum unmodeled force to be about 30% of the force of gravity. 
    #         # This makes the scaling adaptive to the drone's mass.
    #         max_fictitious_force = self.drone.masses * 9.81 * self.fictitious_force_gravity_ratio  # Scale fictitious forces relative to gravity
    #         scaled_fictitious_forces = fictitious_forces_norm_3d * max_fictitious_force
            
    #         # 3. Combine the motor forces and the fictitious forces.
    #         total_forces = motor_forces + scaled_fictitious_forces.reshape(motor_forces.shape)

    #         # 4. Make a SINGLE call to the physics engine with all forces and torques.
    #         self.drone.base_link.apply_forces_and_torques_at_pos(
    #             forces=total_forces.reshape(-1, 3),
    #             torques=motor_torques.reshape(-1, 3),
    #             is_global=False
    #         )
    #         # We can still return effort for logging if needed
    #         self.effort = motor_forces.sum(-1)

    #     elif self.fine_tuning:
    #         # --- Step 4: Fine-Tuning Logic ---
    #         action_7d = tensordict[("agents", "action")]
    #         corrected_controls_4d = action_7d[..., :4] 
    #         fictitious_forces_norm_3d = action_7d[..., 4:]

    #         motor_forces, motor_torques = self.drone.apply_action(corrected_controls_4d)
            
    #         # --- THE FIX: Remove the incorrect reshape ---
    #         max_fictitious_force = (self.drone.masses * 9.81 * self.fictitious_force_gravity_ratio)
    #         # Broadcasting works correctly here: [N, 1, 3] * [N, 1, 1] -> [N, 1, 3]
    #         scaled_fictitious_forces = fictitious_forces_norm_3d * max_fictitious_force 
    #         # ---------------------------------------------

    #         # The reshape in the next line might also be unnecessary if shapes match,
    #         # but it's less likely to cause errors. Keep it for now.
    #         total_forces = motor_forces + scaled_fictitious_forces.reshape(motor_forces.shape) 

    #         self.drone.base_link.apply_forces_and_torques_at_pos(
    #             forces=total_forces.reshape(-1, 3),
    #             torques=motor_torques.reshape(-1, 3),
    #             is_global=False
    #         )
    #         self.effort = motor_forces.sum(-1)

    #     else:
    #         # --- Standard 4D Action Logic (e.g., for pre-training, fine-tuning) ---
    #         actions = tensordict[("agents", "action")]
    #         actions = torch.clamp(actions, min = -1., max = 1.)
    #         # 1. Calculate the forces and torques.
    #         forces, torques = self.drone.apply_action(actions)
            
    #         # 2. Make a SINGLE call to the physics engine.
    #         self.drone.base_link.apply_forces_and_torques_at_pos(
    #             forces=forces.reshape(-1, 3),
    #             torques=torques.reshape(-1, 3),
    #             is_global=False
    #         )
    #         self.effort = forces.sum(-1)

    # def _pre_sim_step(self, tensordict: TensorDictBase):
    #     if self.domain_adaptation:
    #         # ==============================================================================
    #         # 1. The policy now outputs a 7D delta action.
    #         delta_action_7d = tensordict[("agents", "action")]

    #         # 2. Split the 7D tensor into its two components.
    #         delta_controls_4d = delta_action_7d[..., :4]    # First 4 elements
    #         fictitious_forces_3d = 0.1 * delta_action_7d[..., 4:] # Last 3 elements

    #         # 3. Get the 4D real action from the dataset.
    #         traj_idx = self.trajectory_indices
    #         time_idx = self.timestep_indices
    #         real_action_4d = self.real_actions[traj_idx, time_idx].unsqueeze(1)

    #         # 4. Apply the CONTROL correction (4D + 4D) to the motors.
    #         corrected_action = torch.clamp(real_action_4d + delta_controls_4d, min=-1., max=1.)
    #         self.effort = self.drone.apply_action(corrected_action)

    #         # 5. Apply the FICTITIOUS FORCES directly to the drone's body.
    #         #    The forces are applied in the drone's local frame (is_global=False).
    #         self.drone.base_link.apply_forces_and_torques_at_pos(
    #             forces=fictitious_forces_3d.reshape(-1, 3),
    #             is_global=False
    #         )
    #         # ==============================================================================
    #         '''
    #         # ==============================================================================
    #         delta_action = tensordict[("agents", "action")]

    #         # Get the real action from the dataset for the current timestep
    #         traj_idx = self.trajectory_indices
    #         time_idx = self.timestep_indices
    #         real_action = self.real_actions[traj_idx, time_idx].unsqueeze(1)
    #         # print(f"delta_action: {delta_action}, real_action: {real_action}")

    #         # The policy learns a corrective term
    #         corrected_action = real_action + delta_action
    #         corrected_action = torch.clamp(corrected_action, min=-1., max=1.)  # clip(-1,1)
    #         # print(f"corrected_action: {corrected_action}")

    #         self.effort = self.drone.apply_action(corrected_action)
    #         '''
    #     else:
    #         actions = tensordict[("agents", "action")]
    #         actions = torch.clamp(actions, min = -1., max = 1.)  # clip(-1,1)
    #         # print("actions:", actions)

    #         self.effort = self.drone.apply_action(actions)
    #         # ================================================================================

    #     # self.spin_rotors_vis()
    #     # self._push_payload()

    def _push_payload(self):
        env_mask = (torch.rand(self.num_envs, device=self.device) < self.push_prob).float()  # multiplies by 0 or 1 per env → only apply force to selected ones.
        forces = self.push_force_dist.sample((self.num_envs,))
        forces = (
            forces.clamp_max(self.push_force_dist.scale * 3)
            * self.payload_masses
            * env_mask.unsqueeze(-1)
        )
        self.payload.apply_forces(forces)

    def _set_specs(self):
        '''
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 9
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        '''
        # ================================================================================
        # Original obs_dim is 32
        # sim_obs: [ey, eIy, ey_dot, eq, q, ew, R_vec, eb1, eIb1, eW]
        obs_dim = 3 + 3 + 3 + 3 + 3 + 3 + 9 + 1 + 1 + 3 # = 32
        
        if self.domain_adaptation:
            # New observation adds the 4D real action
            obs_dim += 4 # 32 + 4 = 36
        # ================================================================================

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            }
        }).expand(self.num_envs).to(self.device)

        # Define the new 7D action space
        if self.domain_adaptation or self.fine_tuning:
            action_dim = 6 #7
        else:
            action_dim = self.drone.action_spec.shape[-1]
        self.action_spec = CompositeSpec({
            "agents": {
                "action": UnboundedContinuousTensorSpec((1, action_dim)),
            }
        }).expand(self.num_envs).to(self.device)
        '''
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        '''
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

        # ==============================================================================
        stats_dict = {
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "ey": UnboundedContinuousTensorSpec(1),
            # "ey_norm": UnboundedContinuousTensorSpec(1),
            "eIy": UnboundedContinuousTensorSpec(1),
            # "eIy_norm": UnboundedContinuousTensorSpec(1),
            "ey_dot": UnboundedContinuousTensorSpec(1),
            # "ey_dot_norm": UnboundedContinuousTensorSpec(1),
            "eq": UnboundedContinuousTensorSpec(1),
            "ew": UnboundedContinuousTensorSpec(1),
            # "ew_norm": UnboundedContinuousTensorSpec(1),
            "eb1": UnboundedContinuousTensorSpec(1),
            # "eb1_norm": UnboundedContinuousTensorSpec(1),
            "eIb1": UnboundedContinuousTensorSpec(1),
            # "eIb1_norm": UnboundedContinuousTensorSpec(1),
            "eW": UnboundedContinuousTensorSpec(1),
            # "eW_norm": UnboundedContinuousTensorSpec(1),

            "rwd_ey": UnboundedContinuousTensorSpec(1),
            "rwd_ey_dot": UnboundedContinuousTensorSpec(1),
            "rwd_eq": UnboundedContinuousTensorSpec(1),
            "rwd_ew": UnboundedContinuousTensorSpec(1),
            "rwd_eb1": UnboundedContinuousTensorSpec(1),
            "rwd_eW": UnboundedContinuousTensorSpec(1) 
        }

        if self.domain_adaptation:
            stats_dict.update({
                "eR": UnboundedContinuousTensorSpec(1),
                "rwd_eR": UnboundedContinuousTensorSpec(1)
            })
        else:
            stats_dict.update({
                "rwd_eIy": UnboundedContinuousTensorSpec(1),
                "rwd_eIy": UnboundedContinuousTensorSpec(1),
                "rwd_eb1": UnboundedContinuousTensorSpec(1),
                "rwd_eIb1": UnboundedContinuousTensorSpec(1)
            })
        stats_spec = CompositeSpec(stats_dict).expand(self.num_envs).to(self.device)
        # ==============================================================================

        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

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
        self.W = quat_rotate_inverse(self.quaternion, W_w)  # in the body frame
        b1 = self.drone_state[..., 13:16]
        b3 = self.drone_state[..., 16:19]

        # Quadrotor states
        x = self.get_env_poses(self.drone.get_world_poses())[0]
        x_dot = self.drone.get_velocities()[..., :3].squeeze(1)
        # Get rotation matrix
        R = quaternion_to_rotation_matrix(self.quaternion).reshape(self.quaternion.shape[:-1] + (3, 3))
        R = R.reshape(-1, 3, 3)  # Ensure shape is [N, 3, 3]
        # Check orthogonality error
        RtR = torch.matmul(R.transpose(-2, -1), R)
        identity = torch.eye(3, device=R.device).expand_as(RtR)
        error = torch.norm(RtR - identity, dim=(-2, -1)).max()
        # Repair R if needed
        if not torch.allclose(RtR, identity, atol=1e-2):
            # print("Correcting R to ensure it is a valid rotation matrix (SO(3))")
            R = ensure_SO3(R)
        # Sanity checks
        assert not torch.isnan(R).any(), "NaNs in R"
        assert not torch.isinf(R).any(), "Infs in R"
        # Flatten R for downstream usage
        R_vec = R.permute(0, 2, 1).reshape(-1, 9)

        # Payload states
        self.payload_pos = self.y = self.get_env_poses(self.payload.get_world_poses())[0]
        self.payload_vels = self.payload.get_velocities()
        self.y_dot = self.payload_vels[..., :3]
        self.drone_payload_rpos = q = (self.payload_pos.unsqueeze(1) - self.drone.pos)/self.bar_length
        q = ensure_S2(q).squeeze(1) # re-normalization if needed
        #payload_w = self.payload_vels[..., 3:]   # Wrong: this is payload angular velocity
        #bar_w = self.bar.get_velocities()[..., 3:]  # Somehow, payload_w == bar_w
        w = torch.cross(q, (self.y_dot - x_dot), dim=-1) / self.bar_length  # bar angular velocity (world frame)
        self.w = w - (w * q).sum(dim=-1, keepdim=True) * q  # w_projected

        # state = [y, y_dot, q, w_projected, R_vec, W.squeeze(1)]
        # Re-create the state list with the correct 3D shapes
        state = [t.unsqueeze(1) for t in [self.y, self.y_dot, q, self.w]]
        state.insert(4, R_vec.unsqueeze(1))
        state.insert(5, self.W) # W is already [B, 1, 3] from quat_rotate_inverse

        # print(f"Initial States (y) in `_compute_state_and_obs`: {state[0]}")
        # print(f"Initial States (y_dot) in `_compute_state_and_obs`: {state[1]}")
        # print(f"Initial States (q) in `_compute_state_and_obs`: {state[2]}")
        # print(f"Initial States (w) in `_compute_state_and_obs`: {state[3]}")
        # print(f"Initial States (R_vec) in `_compute_state_and_obs`: {state[4]}")
        # print(f"Initial States (W_b) in `_compute_state_and_obs`: {state[5]}")
        # print("----------------------------------------\n")

        # ==============================================================================
        # Conditionally update desired states and compute eR
        if self.domain_adaptation:
            traj_idx, time_idx = self.trajectory_indices, self.timestep_indices
            
            next_time_idx = torch.clamp(time_idx + 1, max=self.real_trajectory_length - 1)
            real_next_state = self.real_states[traj_idx, next_time_idx] # TODO:self.real_next_states[traj_idx, next_time_idx]
            self.yd, self.yd_dot, self.qd, self.wd, R_vec_d, self.Wd = self.unpack_state_vector(real_next_state, add_agent_dim=True)
            # print(f"yd: {self.yd}, y: {y}")

            self.Rd = R_vec_d.view(-1, 3, 3)#.transpose(1, 2)
            self.b1d = self.Rd[..., :, 0].unsqueeze(1)
            # print(f"Rd: {self.Rd}, R: {R}")
            self.eR = 0.5 * self.vee(torch.bmm(self.Rd.transpose(-1, -2), R) - torch.bmm(R.transpose(-1, -2), self.Rd))
            self.eR = self.eR.unsqueeze(1) # Shape will now be [2048, 1, 3]
        else:
            # Dynamically compute desired angular velocity (Wd) from b1d and its derivative
            b3 = quat_axis(self.quaternion, axis=2).squeeze(1)

            # Use WORLD frame angular velocity to compute the derivative of b3 (world frame)
            W_w = self.drone_state[..., 10:13].squeeze(1) # Get W_world from full state (B, 3) 
            b3_dot = torch.cross(W_w, b3, dim=-1)  # (B, 3)
            #b3_dot = torch.cross(self.W.squeeze(1), b3, dim=-1)  # (B, 3)

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
        # ==============================================================================

        # Original observation calculation
        obs_list = self.get_norm_error_state(state, R)
        obs_list_unsqueeze = [o.unsqueeze(1) if o.ndim == 2 else o for o in obs_list]
        self.error_obs = copy.deepcopy(obs_list_unsqueeze) 
        # print(f"obs_list: {obs_list}, obs_list_unsqueeze: {obs_list_unsqueeze}")
        obs = torch.cat(obs_list_unsqueeze, dim=-1)

        # Conditionally append real action
        if self.domain_adaptation:
            real_action = self.real_actions[self.trajectory_indices, self.timestep_indices].unsqueeze(1)
            obs = torch.cat([obs, real_action], dim=-1)
        
        # Reward func test
        # print(f"y in _compute_state_and_obs: {self.get_env_poses(self.payload.get_world_poses())[0]}")

        self.update_stats()

        # self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        # self.smoothness = (
        #     self.drone.get_linear_smoothness()
        #     + self.drone.get_angular_smoothness()
        # )
        # self.stats["motion_smoothness"].lerp_(self.smoothness, (1-self.alpha))

        if self.trajectory_collection:
            # FIX: Add the full state vector to the tensordict so it can be collected
            full_state_vector = torch.cat([s.squeeze(1) for s in state], dim=-1)

            return TensorDict(
                {
                    "agents": {
                        "observation": obs,
                        "intrinsics": self.drone.intrinsics,
                    },
                    "state": full_state_vector, # ADD THIS LINE
                    "stats": self.stats.clone(),
                },
                self.batch_size,
            )
        else:
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
        # print(f"y in _compute_reward_and_done: {self.get_env_poses(self.payload.get_world_poses())[0]}")

        ey_norm, eIy_norm, ey_dot_norm, eq, q, ew_norm, R_vec, eb1_norm, eIb1_norm, eW_norm = self.error_obs
        # print(f"ey_norm: {ey_norm}, norm(ey_norm): {torch.norm(ey_norm, dim=-1, keepdim=True)}")

        # ==============================================================================
        # --- THE FIX: IGNORE THE UNCONTROLLABLE w_3 COMPONENT ---
        # Create a mask to zero out the 3rd component (the z-axis) of the bar's angular velocity error.
        # ew_mask = torch.tensor([[1., 1., 0.]], device=self.device)
        # ew_norm_masked = ew_norm * ew_mask
        # ew_sq_norm = torch.sum((ew_norm_masked*self.w_lim)**2, dim=-1)
        # ---------------------------------------------------------
        
        # Calculate squared norms for each error term
        ey_sq_norm = torch.sum((ey_norm*self.y_lim)**2, dim=-1)
        ey_dot_sq_norm = torch.sum((ey_dot_norm*self.y_dot_lim)**2, dim=-1)
        eq_sq_norm = torch.sum(eq**2, dim=-1)
        ew_sq_norm = torch.sum((ew_norm*self.w_lim)**2, dim=-1)
        eb1_sq_norm = (eb1_norm*torch.pi).squeeze(-1)**2
        eW_sq_norm = torch.sum((eW_norm*self.W_lim)**2, dim=-1)
        ''' USE NORM ERRORS
        ey_sq_norm = torch.sum(ey_norm**2, dim=-1)
        ey_dot_sq_norm = torch.sum(ey_dot_norm**2, dim=-1)
        eq_sq_norm = torch.sum(eq**2, dim=-1)
        ew_sq_norm = torch.sum(ew_norm**2, dim=-1)
        eb1_sq_norm = eb1_norm.squeeze(-1)**2
        eW_sq_norm = torch.sum(eW_norm**2, dim=-1)
        '''

        if self.domain_adaptation:
            # """
            eR_sq_norm = torch.sum(self.eR**2, dim=-1)

            # The reward is now exp(-k * error) instead of exp(-k * error^2)
            ey_sq_norm = torch.sqrt(ey_sq_norm) 
            ########################################################################
            # 2. New linear penalty on the physical position error
            #    Note: We take the square root to get the linear error magnitude
            # ey_linear_penalty = self.Cy_penalty * torch.sqrt(ey_sq_norm)
            ########################################################################
            # ey_dot_sq_norm = torch.sqrt(ey_dot_sq_norm) 
            eq_sq_norm = torch.sqrt(eq_sq_norm) 
            # ew_sq_norm = torch.sqrt(ew_sq_norm) 
            eb1_sq_norm = torch.sqrt(eb1_sq_norm) 
            eR_sq_norm = torch.sqrt(eR_sq_norm) 
            # eW_sq_norm =torch.sqrt(eW_sq_norm) 

            r_min = 0.0
            r_max = self.Cy + self.Cy_dot + self.Cq + self.Cw + self.Cb1 + self.CR + self.CW

            # --------------------------
            # Negative Exponential
            # --------------------------
            exp_ey_sq_norm = torch.exp(-self.rwd_k_exp * ey_sq_norm)
            ########################################################################
            # exp_ey_sq_norm -= ey_linear_penalty
            # r_min = -self.Cy_penalty  # rough approximation, but sufficient
            ########################################################################
            exp_ey_dot_sq_norm = torch.exp(-self.rwd_k_exp * ey_dot_sq_norm)
            exp_eq_sq_norm = torch.exp(-self.rwd_k_exp * eq_sq_norm)
            exp_ew_sq_norm = torch.exp(-self.rwd_k_exp * ew_sq_norm)
            exp_eb1_sq_norm = torch.exp(-self.rwd_k_exp * eb1_sq_norm)
            exp_eR_sq_norm = torch.exp(-self.rwd_k_exp * eR_sq_norm)
            exp_eW_sq_norm = torch.exp(-self.rwd_k_exp * eW_sq_norm)

            # Stats tracking
            self.stats["rwd_ey"].lerp_(exp_ey_sq_norm, (1-self.alpha))
            self.stats["rwd_ey_dot"].lerp_(exp_ey_dot_sq_norm, (1-self.alpha))
            self.stats["rwd_eq"].lerp_(exp_eq_sq_norm, (1-self.alpha))
            self.stats["rwd_ew"].lerp_(exp_ew_sq_norm, (1-self.alpha))
            self.stats["rwd_eb1"].lerp_(exp_eb1_sq_norm, (1-self.alpha))
            self.stats["rwd_eR"].lerp_(exp_eR_sq_norm, (1-self.alpha))
            self.stats["rwd_eW"].lerp_(exp_eW_sq_norm, (1-self.alpha))

            # Weighted exponential reward
            reward = (
                self.Cy*exp_ey_sq_norm + self.Cy_dot*exp_ey_dot_sq_norm +
                self.Cq*exp_eq_sq_norm + self.Cw*exp_ew_sq_norm + self.Cb1*exp_eb1_sq_norm +
                self.CR*exp_eR_sq_norm + self.CW*exp_eW_sq_norm
            )
            # """

            """
           # 1. Maximum possible reward is 0 (when all errors are zero)
            r_max = 0.0

            # 2. Calculate the theoretical minimum reward based on max possible errors
            coeffs_3d = (self.Cy*self.y_lim) + (self.Cy_dot*self.y_dot_lim) \
                + self.Cq + (self.Cw*self.w_lim) + (self.CW*self.W_lim) + self.CR # Sum of coefficients for 3D error terms (max squared norm = 3.0)
            coeffs_1d = (self.Cb1*torch.pi)  # Sum of coefficients for 1D error terms (max squared norm = 1.0)
            r_min = - (coeffs_3d * 3.0 + coeffs_1d * 1.0)

            eR_sq_norm = torch.sum(self.eR**2, dim=-1)
            
            quad_ey_sq_norm     = -ey_sq_norm     #1. - (ey_sq_norm     / 3.).clamp(0., 1.)
            quad_ey_dot_sq_norm = -ey_dot_sq_norm #1. - (ey_dot_sq_norm / 3.).clamp(0., 1.)
            quad_eq_sq_norm     = -eq_sq_norm     #1. - (eq_sq_norm     / 1.).clamp(0., 1.)
            quad_ew_sq_norm     = -ew_sq_norm     #1. - (ew_sq_norm     / 4.).clamp(0., 1.)
            quad_eb1_sq_norm    = -eb1_sq_norm    #1. - (eb1_sq_norm    / 1.).clamp(0., 1.)
            quad_eR_sq_norm     = -eR_sq_norm
            quad_eW_sq_norm     = -eW_sq_norm     #1. - (eW_sq_norm     / 4.).clamp(0., 1.)

            # Stats tracking
            self.stats["rwd_ey"].lerp_(quad_ey_sq_norm, (1-self.alpha))
            self.stats["rwd_ey_dot"].lerp_(quad_ey_dot_sq_norm, (1-self.alpha))
            self.stats["rwd_eq"].lerp_(quad_eq_sq_norm, (1-self.alpha))
            self.stats["rwd_ew"].lerp_(quad_ew_sq_norm, (1-self.alpha))
            self.stats["rwd_eb1"].lerp_(quad_eb1_sq_norm, (1-self.alpha))
            self.stats["rwd_eR"].lerp_(quad_eR_sq_norm, (1-self.alpha))
            self.stats["rwd_eW"].lerp_(quad_eW_sq_norm, (1-self.alpha))

            reward = (
                self.Cy*quad_ey_sq_norm + self.Cy_dot*quad_ey_dot_sq_norm +
                self.Cq*quad_eq_sq_norm + self.Cw*quad_ew_sq_norm +
                self.Cb1*quad_eb1_sq_norm + self.CW*quad_eW_sq_norm + self.CR*quad_eR_sq_norm
            )
            """
            """
            # 1. Maximum possible reward is 0 (when all errors are zero)
            r_max = 0.0

            # 2. Calculate the theoretical minimum reward based on max possible errors
            coeffs_3d = (self.Cy*self.y_lim) + (self.CIy*self.eIy_lim) + (self.Cy_dot*self.y_dot_lim) \
                + self.Cq + (self.Cw*self.w_lim) + (self.CW*self.W_lim) + self.CR # Sum of coefficients for 3D error terms (max squared norm = 3.0)
            coeffs_1d = (self.Cb1*torch.pi) + (self.CIb1*self.eIb1_lim)  # Sum of coefficients for 1D error terms (max squared norm = 1.0)
            ''' USE NORM ERRORS
            coeffs_3d = self.Cy + self.CIy + self.Cy_dot + self.Cq + self.Cw + self.CW  # Sum of coefficients for 3D error terms (max squared norm = 3.0)
            coeffs_1d = self.Cb1 + self.CIb1  # Sum of coefficients for 1D error terms (max squared norm = 1.0)
            '''
            r_min = - (coeffs_3d * 3.0 + coeffs_1d * 1.0)

            eIy_sq_norm = torch.sum(eIy_norm**2, dim=-1)
            eIb1_sq_norm = eIb1_norm.squeeze(-1)**2
            eR_sq_norm = torch.sum(self.eR**2, dim=-1)
            
            quad_ey_sq_norm     = -ey_sq_norm     #1. - (ey_sq_norm     / 3.).clamp(0., 1.)
            quad_eIy_sq_norm    = -eIy_sq_norm    #1. - (eIy_sq_norm    / 3.).clamp(0., 1.)
            quad_ey_dot_sq_norm = -ey_dot_sq_norm #1. - (ey_dot_sq_norm / 3.).clamp(0., 1.)
            quad_eq_sq_norm     = -eq_sq_norm     #1. - (eq_sq_norm     / 1.).clamp(0., 1.)
            quad_ew_sq_norm     = -ew_sq_norm     #1. - (ew_sq_norm     / 4.).clamp(0., 1.)
            quad_eb1_sq_norm    = -eb1_sq_norm    #1. - (eb1_sq_norm    / 1.).clamp(0., 1.)
            quad_eIb1_sq_norm   = -eIb1_sq_norm   #1. - (eIb1_sq_norm   / 1.).clamp(0., 1.)
            quad_eR_sq_norm     = -eR_sq_norm
            quad_eW_sq_norm     = -eW_sq_norm     #1. - (eW_sq_norm     / 4.).clamp(0., 1.)

            # Stats tracking
            self.stats["rwd_ey"].lerp_(quad_ey_sq_norm, (1-self.alpha))
            self.stats["rwd_ey_dot"].lerp_(quad_ey_dot_sq_norm, (1-self.alpha))
            self.stats["rwd_eq"].lerp_(quad_eq_sq_norm, (1-self.alpha))
            self.stats["rwd_ew"].lerp_(quad_ew_sq_norm, (1-self.alpha))
            self.stats["rwd_eb1"].lerp_(quad_eb1_sq_norm, (1-self.alpha))
            self.stats["rwd_eR"].lerp_(quad_eR_sq_norm, (1-self.alpha))
            self.stats["rwd_eW"].lerp_(quad_eW_sq_norm, (1-self.alpha))

            reward = (
                self.Cy*quad_ey_sq_norm + self.CIy*quad_eIy_sq_norm + self.Cy_dot*quad_ey_dot_sq_norm +
                self.Cq*quad_eq_sq_norm + self.Cw*quad_ew_sq_norm +
                self.Cb1*quad_eb1_sq_norm + self.CIb1*quad_eIb1_sq_norm + self.CW*quad_eW_sq_norm + self.CR*quad_eR_sq_norm
            )
            """
            # ==============================================================================     

            """
            eR_sq_norm = torch.sum(self.eR**2, dim=-1)

            ey_sq_norm = torch.sqrt(ey_sq_norm) 
            eq_sq_norm = torch.sqrt(eq_sq_norm) 
            eb1_sq_norm = torch.sqrt(eb1_sq_norm) 
            eR_sq_norm = torch.sqrt(eR_sq_norm) 

            r_max_exp = self.Cy + self.Cy_dot + self.Cq + self.Cw + self.Cb1 + self.CR + self.CW
            
            exp_ey_sq_norm = torch.exp(-self.rwd_k_exp * ey_sq_norm)
            exp_ey_dot_sq_norm = torch.exp(-self.rwd_k_exp * ey_dot_sq_norm)
            exp_eq_sq_norm = torch.exp(-self.rwd_k_exp * eq_sq_norm)
            exp_ew_sq_norm = torch.exp(-self.rwd_k_exp * ew_sq_norm)
            exp_eb1_sq_norm = torch.exp(-self.rwd_k_exp * eb1_sq_norm)
            exp_eR_sq_norm = torch.exp(-self.rwd_k_exp * eR_sq_norm)
            exp_eW_sq_norm = torch.exp(-self.rwd_k_exp * eW_sq_norm)

            # Stats tracking
            self.stats["rwd_ey"].lerp_(exp_ey_sq_norm, (1-self.alpha))
            self.stats["rwd_ey_dot"].lerp_(exp_ey_dot_sq_norm, (1-self.alpha))
            self.stats["rwd_eq"].lerp_(exp_eq_sq_norm, (1-self.alpha))
            self.stats["rwd_ew"].lerp_(exp_ew_sq_norm, (1-self.alpha))
            self.stats["rwd_eb1"].lerp_(exp_eb1_sq_norm, (1-self.alpha))
            self.stats["rwd_eR"].lerp_(exp_eR_sq_norm, (1-self.alpha))
            self.stats["rwd_eW"].lerp_(exp_eW_sq_norm, (1-self.alpha))

            reward_exp = (
                self.Cy*exp_ey_sq_norm + self.Cy_dot*exp_ey_dot_sq_norm +
                self.Cq*exp_eq_sq_norm + self.Cw*exp_ew_sq_norm + self.Cb1*exp_eb1_sq_norm +
                self.CR*exp_eR_sq_norm + self.CW*exp_eW_sq_norm
            )
            
            # --- NEW: ADD INTEGRAL REWARDS TO DOMAIN ADAPTATION ---
            eIy_sq_norm = torch.sum(eIy_norm**2, dim=-1)
            eIb1_sq_norm = eIb1_norm.squeeze(-1)**2
            
            quad_eIy_sq_norm    = -eIy_sq_norm
            quad_eIb1_sq_norm   = -eIb1_sq_norm

            reward_integral = self.CIy*quad_eIy_sq_norm + self.CIb1*quad_eIb1_sq_norm
            
            # Combine rewards
            reward = reward_exp + reward_integral
            
            # Re-calculate r_max and r_min
            r_max = r_max_exp # Optimal integral reward is 0
            
            # r_min approximation
            r_min_integral = - ( (self.CIy*self.eIy_lim) * 3.0 + (self.CIb1*self.eIb1_lim) * 1.0 )
            r_min = r_min_integral # Min for exp part is 0
            # --- END NEW ---
            """

        else:
            # 1. Maximum possible reward is 0 (when all errors are zero)
            r_max = 0.0

            # 2. Calculate the theoretical minimum reward based on max possible errors
            coeffs_3d = (self.Cy*self.y_lim) + (self.CIy*self.eIy_lim) + (self.Cy_dot*self.y_dot_lim) \
                + self.Cq + (self.Cw*self.w_lim) + (self.CW*self.W_lim)  # Sum of coefficients for 3D error terms (max squared norm = 3.0)
            coeffs_1d = (self.Cb1*torch.pi) + (self.CIb1*self.eIb1_lim)  # Sum of coefficients for 1D error terms (max squared norm = 1.0)
            ''' USE NORM ERRORS
            coeffs_3d = self.Cy + self.CIy + self.Cy_dot + self.Cq + self.Cw + self.CW  # Sum of coefficients for 3D error terms (max squared norm = 3.0)
            coeffs_1d = self.Cb1 + self.CIb1  # Sum of coefficients for 1D error terms (max squared norm = 1.0)
            '''
            r_min = - (coeffs_3d * 3.0 + coeffs_1d * 1.0)

            eIy_sq_norm = torch.sum(eIy_norm**2, dim=-1)
            eIb1_sq_norm = eIb1_norm.squeeze(-1)**2
            
            quad_ey_sq_norm     = -ey_sq_norm     #1. - (ey_sq_norm     / 3.).clamp(0., 1.)
            quad_eIy_sq_norm    = -eIy_sq_norm    #1. - (eIy_sq_norm    / 3.).clamp(0., 1.)
            quad_ey_dot_sq_norm = -ey_dot_sq_norm #1. - (ey_dot_sq_norm / 3.).clamp(0., 1.)
            quad_eq_sq_norm     = -eq_sq_norm     #1. - (eq_sq_norm     / 1.).clamp(0., 1.)
            quad_ew_sq_norm     = -ew_sq_norm     #1. - (ew_sq_norm     / 4.).clamp(0., 1.)
            quad_eb1_sq_norm    = -eb1_sq_norm    #1. - (eb1_sq_norm    / 1.).clamp(0., 1.)
            quad_eIb1_sq_norm   = -eIb1_sq_norm   #1. - (eIb1_sq_norm   / 1.).clamp(0., 1.)
            quad_eW_sq_norm     = -eW_sq_norm     #1. - (eW_sq_norm     / 4.).clamp(0., 1.)

            # Stats tracking
            self.stats["rwd_ey"].lerp_(quad_ey_sq_norm, (1-self.alpha))
            self.stats["rwd_eIy"].lerp_(quad_eIy_sq_norm, (1-self.alpha))
            self.stats["rwd_ey_dot"].lerp_(quad_ey_dot_sq_norm, (1-self.alpha))
            self.stats["rwd_eq"].lerp_(quad_eq_sq_norm, (1-self.alpha))
            self.stats["rwd_ew"].lerp_(quad_ew_sq_norm, (1-self.alpha))
            self.stats["rwd_eb1"].lerp_(quad_eb1_sq_norm, (1-self.alpha))
            self.stats["rwd_eIb1"].lerp_(quad_eIb1_sq_norm, (1-self.alpha))
            self.stats["rwd_eW"].lerp_(quad_eW_sq_norm, (1-self.alpha))

            reward = (
                self.Cy*quad_ey_sq_norm + self.CIy*quad_eIy_sq_norm + self.Cy_dot*quad_ey_dot_sq_norm +
                self.Cq*quad_eq_sq_norm + self.Cw*quad_ew_sq_norm +
                self.Cb1*quad_eb1_sq_norm + self.CIb1*quad_eIb1_sq_norm + self.CW*quad_eW_sq_norm
            )
            # ==============================================================================     
            
        # Normalize weighted sum to [0,1]
        reward = (reward - r_min) / (r_max - r_min)

        # -- Termination Conditions based on PHYSICAL limits --
        # These checks use the unnormalized state variables stored in _compute_state_and_obs
        
        misbehave_margin = 1.5 # 150%
        misbehave = (
            (ey_norm.abs() >= 1.0).any(dim=-1) #(self.y.abs() >= self.y_lim).any(dim=-1).reshape(-1, 1)
            | (self.y_dot.abs() >= (self.y_dot_lim * misbehave_margin)).any(dim=-1).reshape(-1, 1)
            | (self.w.abs() >= (self.w_lim * misbehave_margin)).any(dim=-1).reshape(-1, 1)
            | (self.W.abs() >= (self.W_lim * misbehave_margin)).any(dim=-1).reshape(-1, 1)
            | (self.drone.pos[..., 2] < 0.2).reshape(-1, 1)
            | (self.payload_pos[..., 2] < 0.2).reshape(-1, 1)
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
        '''
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan

        # Terminal condition (Out of boundary or crashed!)
        # TODO: reward[misbehave] = self.reward_crash
        reward[terminated.squeeze(-1)] = self.reward_crash

        # ==============================================================================
        # --- MODIFIED TRUNCATION LOGIC USING SAVED 'dones' ---
        if self.domain_adaptation:
            # Get the 'done' flag from the dataset for the CURRENT step
            # for each environment in the batch.
            done_from_data = self.real_dones[
                self.trajectory_indices, self.timestep_indices
            ].bool()

            # Truncate if the flag from the data is True.
            truncated = done_from_data.unsqueeze(-1)

            # Now, advance the timestep counter for the next step.
            self.timestep_indices += 1
        else:
            truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        # ==============================================================================

        # self.target_distance = torch.norm(self.target_payload_rpos[:, [0]], dim=-1)
        # heading_alignment = torch.abs(eb1_norm).squeeze(-1)*torch.pi

        # self.stats["pos_error"].lerp_(self.target_distance, (1-self.alpha))
        # self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
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

    def get_norm_error_state(self, state, R):    
            
        # 1. Get the UNNORMALIZED 2D simulation states
        state_2d = [s.squeeze(1) for s in state]
        
        y, y_dot, q, w, R_vec, W = state_2d
        
        # 2. Squeeze desired angular velocities
        wd_s = self.wd.squeeze(1)
        Wd_s = self.Wd.squeeze(1)

        # 3. Compute geometric errors for angular states FIRST
        # ew = w + q x (q x wd)
        ew = w + torch.cross(q, torch.cross(q, wd_s, dim=-1), dim=-1)
        
        # eW = W - R.T @ Rd @ Wd
        if self.domain_adaptation:
            eW = W - (R.mT @ self.Rd @ Wd_s.unsqueeze(-1)).squeeze(-1)
        else:
            eW = W - Wd_s
        
        # 4. Now, normalize ALL states and errors
        y_norm, y_dot_norm, _, _, _, _ = state_normalization_payload(
           state_2d, self.y_lim, self.y_dot_lim, self.w_lim, self.W_lim
        )

        ew_norm = ew / self.w_lim
        eW_norm = eW / self.W_lim

        '''
        y_norm, y_dot_norm, _, w_norm, _, W_norm = state_normalization_payload(
           state_2d, self.y_lim, self.y_dot_lim, self.w_lim, self.W_lim
        )
        ######################################################
        wd_norm = self.wd / self.w_lim
        Wd_norm = self.Wd.squeeze(1) / self.W_lim
        
        ew_norm = w_norm - wd_norm
        eW_norm = W_norm - Wd_norm
        ######################################################
        '''

        # 5. Add back the agent dimension for the observation tensor
        y_norm = y_norm.unsqueeze(1)
        y_dot_norm = y_dot_norm.unsqueeze(1)
        q = q.unsqueeze(1) # Unsqueeze q here
        ew_norm = ew_norm.unsqueeze(1)
        eW_norm = eW_norm.unsqueeze(1)

        # 6. Compute final observation errors
        yd_norm = self.yd / self.y_lim
        yd_dot_norm = self.yd_dot / self.y_dot_lim
        ey_norm = y_norm - yd_norm
        ey_dot_norm = y_dot_norm - yd_dot_norm
        eq = torch.cross(self.qd.expand_as(q), q, dim=-1) #eq = torch.cross(self.qd, q, dim=-1)

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
        ey = ey_norm * self.y_lim
        eIy_integrand = -self.rwd_alpha * self.eIy.error + ey#.unsqueeze(1)
        # print(f"ey: {ey}, self.eIy.error: {self.eIy.error}")
        self.eIy.integrate(eIy_integrand, self.dt)
        self.eIy_norm = torch.clamp(
            self.eIy.error / self.eIy_lim,
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
            ey_norm,
            self.eIy_norm,
            ey_dot_norm,
            eq,
            q,
            ew_norm,
            R_vec,#.unsqueeze(1), #R_vec.view(R_vec.shape[0], -1),  # (B, 9) ,  # (B, 1, 9)
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

    def get_state(self):
        # Drone state extraction
        drone_state = self.drone.get_state()
        x = self.drone.pos  # drone position in world frame, [B, 3]
        quaternion = drone_state[..., 3:7]  # quaternion (orientation),[B, 4]
        x_dot = drone_state[..., 7:10]  # linear velocity of drone in world frame, [B, 3] 
        W_world = drone_state[..., 10:13]  # angular velocity of drone in world frame, [B, 3] 
        W_body = quat_rotate_inverse(quaternion, W_world)  # convert angular velocity to body frame, [B, 3]
        R = quaternion_to_rotation_matrix(quaternion).squeeze(1)  # rotation matrix, [B, 3, 3]
        R_vec = R.reshape(-1, 9)  # flattened rotation matrix, [B, 9]

        # Payload state extraction
        y = self.get_env_poses(self.payload.get_world_poses())[0].unsqueeze(1)  # payload position in world frame, [B, 1, 3]
        y_dot = self.payload.get_velocities()[..., :3]  # payload linear velocity in world frame, [B, 3]

        # Bar state computation
        q = ensure_S2((y - x) / self.bar_length)  # compute direction vector q from drone to payload, [B, 1, 3]        
        q_dot = (y_dot - x_dot) / self.bar_length  # Time derivative of q, [B, 3]
        w = torch.cross(q, q_dot, dim=-1)  # compute angular velocity of the bar in world, [B, 1, 3]
        # w = self.bar.get_velocities()[..., 3:]  # bar angular velocity in world frame, [B, 3]
        w = w - (w * q).sum(dim=-1, keepdim=True) * q  # bar angular velocity projected perpendicular to q, [B, 1, 3] 

        # Concatenate into full state vector [B, 3+3+3+3+9+3] = [B, 24]
        y = y.squeeze(1)  # payload position, [B, 3] 
        y_dot = y_dot.squeeze(1)  # payload velocity, [B, 3] 
        q = q.squeeze(1)  # bar direction (unit vector), [B, 3] 
        w = w.squeeze(1)  # bar angular velocity, [B, 3] 
        W = W_body.squeeze(1)  # drone angular velocity in body frame, [B, 3] 
        # print(torch.dot(q[0],w[0]))
        state = torch.cat([y, y_dot, q, w, R_vec, W], dim=-1)

        '''
        # --- Get bar orientation in world ---
        bar_pose = self.bar.get_world_poses()
        bar_quat = bar_pose[1]  # [B, 4]
        R_bar = quaternion_to_rotation_matrix(bar_quat).squeeze(1)  # [B, 3, 3]

        # --- Compute q from bar orientation ---
        # Bar's local axis pointing toward payload: -Z or +Z depending on inverted setting
        b3_bar = torch.tensor([0., 0., -1.], device=R_bar.device) if not self.inverted_pendulum \
                else torch.tensor([0., 0., 1.], device=R_bar.device)
        b3_bar = b3_bar.expand(R_bar.shape[0], 3)  # [B, 3]
        q_pred = torch.einsum("bij,bj->bi", R_bar, b3_bar)  # Rotate into world frame
        q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True)  # normalize just in case

        # --- Get bar's angular velocity ---
        omega_bar = self.bar.get_velocities()[..., 3:]  # [B, 3]

        # --- Get drone velocity ---
        x_dot = drone_state[..., 7:10]  # linear velocity of drone in world, [B, 3]

        # --- Predict payload velocity using: y_dot = x_dot + ω × (q * L) ---
        y_dot_pred = x_dot + torch.cross(omega_bar, q_pred * self.bar_length, dim=-1)  # [B, 3]

        # --- Measure payload velocity from sim ---
        y_dot_measured = self.payload.get_velocities()[..., :3]  # [B, 3]

        # --- Compute and report error ---
        error = torch.norm(y_dot_pred - y_dot_measured, dim=-1)  # [B]
        print(y_dot, y_dot_pred)
        print(f"[Payload Velocity Check] mean error: {error.mean().item():.6f}, max error: {error.max().item():.6f}")
        
        error = torch.norm(q_pred - q, dim=-1)  # [B]
        print(q_pred, q)
        print(f"[q Check] mean error: {error.mean().item():.6f}, max error: {error.max().item():.6f}")
        '''
        return state.squeeze(0).detach().cpu().numpy()