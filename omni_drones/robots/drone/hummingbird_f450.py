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
from torch.func import vmap

from omni.isaac.core.prims import RigidPrimView
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH
from omni_drones.utils.torch import quat_rotate, quat_axis


class Hummingbird_F450(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/hummingbird_f450.usd"
    param_path: str = ASSET_PATH + "/usd/hummingbird_f450.yaml"

    def vmap_fMs(self, cmds):
        f_norm, M = cmds[..., 0], cmds[..., 1:]  # extract normalized force and moment components [-1,1]
        # f_norm, M = cmds[..., 0], cmds[..., 1:]*0  # extract normalized force and moment components [-1,1]

        f_total = torch.clamp(4*(self.scale_act * f_norm + self.avrg_act), 4*self.min_force, 4*self.max_force)

        # Ensure force is only in the z-direction in the drone's body frame
        f = torch.stack([torch.zeros_like(f_total), torch.zeros_like(f_total), f_total], dim=-1) # [N]
        # f = torch.stack([torch.zeros_like(f_total), torch.zeros_like(f_total), torch.ones_like(f_total)*self.hover_force*4], dim=-1) # [N]

        return f.unsqueeze(-2), M.unsqueeze(-2)  # Expand to match expected shape


    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        fMs = actions.expand(*self.shape, self.num_rotors)  # [f, M]s from RL algos

        # Apply vmap to directly process force and moments per drone
        forces, torques = vmap(self.vmap_fMs)(fMs)
        forces = forces.squeeze(-3)  # ensure [num_envs, 1, 3]
        torques = torques.squeeze(-3)  # ensure [num_envs, 1, 3]

        # Store force and torque in the simulation
        self.forces[:] = forces
        self.torques[:] = torques
        '''
        print(self.forces, self.torques)
        tensor([[[ 0.0000,  0.0000, 13.6167]], [[ 0.0000,  0.0000, 10.9408]]], device='cuda:0') 
        tensor([[[-0.7194,  0.9023, -0.8981]], [[-0.2725, -1.0000, -1.0000]]], device='cuda:0')
        '''

        # Apply forces and torques to the drone body
        self.base_link.apply_forces_and_torques_at_pos(
            self.forces.reshape(-1, 3),
            self.torques.reshape(-1, 3),
            is_global=False
        )

        # # spin spinning rotors
        # self.dof_vel = self.drone.base_link.get_joint_velocities()
        # prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        # self.dof_vel[:, 0] = prop_rot[:, 0]
        # self.dof_vel[:, 1] = -1.0 * prop_rot[:, 1]
        # self.dof_vel[:, 2] = prop_rot[:, 2]
        # self.dof_vel[:, 3] = -1.0 * prop_rot[:, 3]
        # self.drone.base_link.set_joint_velocities(self.dof_vel)

        # self.M_b = torch.tensor([self.M_b[0], -self.M_b[1], -self.M_b[2]], device=self.device)
        # self.drone.base_link.rigid_body.apply_forces_and_torques_at_pos(self.f_b, self.M_b, is_global = False)
        # # print(self.f_b, self.M_b)
        return self.forces.sum(-1)

    def apply_action_backup(self, actions: torch.Tensor) -> torch.Tensor:
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)  # from RL algos
        
        last_throttle = self.throttle.clone()
        '''
        print(rotor_cmds, self.throttle)
            tensor([[[ 0.8599,  0.0038, -1.0000, -0.0456]]], device='cuda:0') tensor([[[0.7822, 0.7662, 0.2454, 0.5947]]], device='cuda:0')
            tensor([[[ 0.5043, -0.5897,  1.0000,  0.7166]]], device='cuda:0') tensor([[[0.8188, 0.6315, 0.5699, 0.7374]]], device='cuda:0')
            tensor([[[-0.3484, -0.3177,  1.0000, -0.4968]]], device='cuda:0') tensor([[[0.7121, 0.6111, 0.7548, 0.6360]]], device='cuda:0')
            tensor([[[ 0.5959, -0.0589,  0.8307,  1.0000]]], device='cuda:0') tensor([[[0.7900, 0.6433, 0.8416, 0.7925]]], device='cuda:0')
            tensor([[[ 0.4062, -1.0000,  0.5383, -0.1198]]], device='cuda:0') tensor([[[0.8109, 0.3667, 0.8569, 0.7370]]], device='cuda:0')
            tensor([[[ 1.0000, -0.9293, -0.4549, -0.4778]]], device='cuda:0') tensor([[[0.8922, 0.2899, 0.7129, 0.6398]]], device='cuda:0')
            tensor([[[ 0.1976, -0.4307,  0.2229, -0.1881]]], device='cuda:0') tensor([[[0.8413, 0.3947, 0.7426, 0.6387]]], device='cuda:0')
            tensor([[[-0.8205, -0.6747,  0.7582, -0.7014]]], device='cuda:0') tensor([[[0.6084, 0.3984, 0.8264, 0.5302]]], device='cuda:0')
            tensor([[[-1.0000, -0.9057, -0.6477,  0.4805]]], device='cuda:0') tensor([[[0.3468, 0.3204, 0.6515, 0.6722]]], device='cuda:0')
        '''

        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(rotor_cmds, self.rotor_params)
        '''
        print(thrusts, moments)
            tensor([[[2.2093, 1.6761, 2.5931, 4.4671]]], device='cuda:0') tensor([[[0.0353, -0.0268,  0.0415, -0.0715]]], device='cuda:0')
            tensor([[[3.1690, 2.0944, 3.8866, 4.5373]]], device='cuda:0') tensor([[[0.0507, -0.0335,  0.0622, -0.0726]]], device='cuda:0')
            tensor([[[2.4683, 3.5287, 3.0456, 5.1182]]], device='cuda:0') tensor([[[0.0395, -0.0565,  0.0487, -0.0819]]], device='cuda:0')
            tensor([[[3.7989, 4.2006, 3.6186, 3.6345]]], device='cuda:0') tensor([[[0.0608, -0.0672,  0.0579, -0.0582]]], device='cuda:0')
            tensor([[[2.0717, 3.1580, 1.1757, 1.1808]]], device='cuda:0') tensor([[[0.0331, -0.0505,  0.0188, -0.0189]]], device='cuda:0')
            tensor([[[2.4013, 1.5176, 1.0692, 2.0909]]], device='cuda:0') tensor([[[0.0384, -0.0243,  0.0171, -0.0335]]], device='cuda:0')
            tensor([[[0.7802, 2.5914, 2.6993, 2.5059]]], device='cuda:0') tensor([[[0.0125, -0.0415,  0.0432, -0.0401]]], device='cuda:0')
            tensor([[[1.6803, 3.5442, 3.9602, 0.8142]]], device='cuda:0') tensor([[[0.0269, -0.0567,  0.0634, -0.0130]]], device='cuda:0')
            tensor([[[1.5625, 1.3070, 4.7868, 1.7933]]], device='cuda:0') tensor([[[0.0250, -0.0209,  0.0766, -0.0287]]], device='cuda:0')
        '''

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))  # Extract the Z-axis direction of each rotor (since thrust is usually along the Z-axis in local space).
        '''
        print(torque_axis)  # four axes share the same values because they are attached to the drone body
            tensor([[[[-0.0036, -0.4266,  0.9044],                     # rotor1
                      [-0.0036, -0.4266,  0.9044],                     # rotor2
                      [-0.0036, -0.4266,  0.9044],                     # rotor3
                      [-0.0036, -0.4266,  0.9044]]]], device='cuda:0') # rotor4
            tensor([[[[-0.0958, -0.4079,  0.9080],
                      [-0.0958, -0.4079,  0.9080],
                      [-0.0958, -0.4079,  0.9080],
                      [-0.0958, -0.4079,  0.9080]]]], device='cuda:0')
            tensor([[[[-0.1904, -0.3870,  0.9022],
                      [-0.1904, -0.3870,  0.9022],
                      [-0.1904, -0.3870,  0.9022],
                      [-0.1904, -0.3870,  0.9022]]]], device='cuda:0')
            tensor([[[[-0.2851, -0.3656,  0.8860],
                      [-0.2851, -0.3656,  0.8860],
                      [-0.2851, -0.3656,  0.8860],
                      [-0.2851, -0.3656,  0.8860]]]], device='cuda:0')
        '''

        self.thrusts[..., 2] = thrusts  # the [..., 2] selects the Z-component of the self.thrusts tensor
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)  # Each rotor's torque magnitude is multiplied by its `torque axis`. This distributes the torque along the correct world-frame direction.
        '''
        print(moments.unsqueeze(-1) * torque_axis, (moments.unsqueeze(-1) * torque_axis).sum(-2))  # sum(-2) = Sums up the torques from all rotors, giving the net torque acting on the drone.
            tensor([[[[ 0.0238, -0.0013, -0.0296],
                      [-0.0367,  0.0020,  0.0455],
                      [ 0.0132, -0.0007, -0.0163],
                      [-0.0423,  0.0023,  0.0524]]]], device='cuda:0') tensor([[[-0.0420,  0.0023,  0.0520]]], device='cuda:0')
            tensor([[[[ 0.0283, -0.0109, -0.0241],
                      [-0.0484,  0.0186,  0.0413],
                      [ 0.0074, -0.0028, -0.0063],
                      [-0.0160,  0.0061,  0.0136]]]], device='cuda:0') tensor([[[-0.0288,  0.0110,  0.0245]]], device='cuda:0')
            tensor([[[[ 0.0436, -0.0270, -0.0238],
                      [-0.0353,  0.0219,  0.0193],
                      [ 0.0148, -0.0092, -0.0081],
                      [-0.0144,  0.0089,  0.0079]]]], device='cuda:0') tensor([[[ 0.0086, -0.0054, -0.0047]]], device='cuda:0')
        '''

        self.forces.zero_()
        # TODO: global downwash
        if self.n > 1:
            self.forces[:] += vmap(self.downwash)(
                self.pos,
                self.pos,
                quat_rotate(self.rot, self.thrusts.sum(-2)),
                kz=0.3
            ).sum(-2)
        self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]

        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3), # Converts thrust values into 3D force vectors. Likely [0, 0, thrust], meaning force is applied upwards in rotor's local frame.
            is_global=False  # applied in the local frame
        )
        self.base_link.apply_forces_and_torques_at_pos(
            self.forces.reshape(-1, 3),
            self.torques.reshape(-1, 3),
            is_global=True  # applied in the world frame
        )
        '''
        self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)  # the net torque affects yaw, pitch, and roll of the drone.
        self.forces  = torch.zeros(*self.shape, 3, device=self.device)

        print(self.thrusts, self.torques, self.forces)
            tensor([[[[0., 0., 2.4013],
                      [0., 0., 1.5176],
                      [0., 0., 1.0692],
                      [0., 0., 2.0909]]]], device='cuda:0') tensor([[[ 0.0019, -0.0007,  0.0008]]], device='cuda:0') tensor([[[0., 0., 0.]]], device='cuda:0')
            tensor([[[[0., 0., 0.7802],
                      [0., 0., 2.5914],
                      [0., 0., 2.6993],
                      [0., 0., 2.5059]]]], device='cuda:0') tensor([[[ 0.0216, -0.0073,  0.0122]]], device='cuda:0') tensor([[[0., 0., 0.]]], device='cuda:0')
            tensor([[[[0., 0., 1.6803],
                      [0., 0., 3.5442],
                      [0., 0., 3.9602],
                      [0., 0., 0.8142]]]], device='cuda:0') tensor([[[-0.0162,  0.0049, -0.0116]]], device='cuda:0') tensor([[[0., 0., 0.]]], device='cuda:0')
        '''
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)
