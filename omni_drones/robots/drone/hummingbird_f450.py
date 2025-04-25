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

import logging
from typing import Type, Dict

import torch
import torch.distributions as D
from torch.func import vmap

# from omni.isaac.core.prims import RigidPrimView
from omni_drones.views import RigidPrimView
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH
from omni_drones.robots import RobotBase, RobotCfg

import pprint

class Hummingbird_F450(MultirotorBase):
    usd_path: str = ASSET_PATH + "/usd/hummingbird_f450.usd"
    param_path: str = ASSET_PATH + "/usd/hummingbird_f450.yaml"

    def __init__(
        self,
        name: str = None,
        cfg: RobotCfg=None,
        is_articulation: bool = True,
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "com": UnboundedContinuousTensorSpec(3),
            "c_tf": UnboundedContinuousTensorSpec(1),
            # "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            # "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            # "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            # "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            # "drag_coef": UnboundedContinuousTensorSpec(1),
        }).to(self.device)


    def initialize(
        self,
        prim_paths_expr: str = None,
        track_contact_forces: bool = False
    ):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                track_contact_forces=track_contact_forces,
                shape=self.shape,
            )
            self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
            rotor_joint_indices = [
                i for i, dof_name in enumerate(self._view._dof_names)
                if dof_name.startswith("rotor")
            ]
            if len(rotor_joint_indices):
                self.rotor_joint_indices = torch.tensor(
                    rotor_joint_indices,
                    device=self.device
                )
            else:
                self.rotor_joint_indices = None
        else:
            super().initialize(prim_paths_expr=f"{prim_paths_expr}/base_link")
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        '''
        self.rotors_view = RigidPrimView(
            # prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]",
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_*",
            name="rotors",
            shape=(*self.shape, self.num_rotors)
        )
        self.rotors_view.initialize()

        rotor_config = self.params["rotor_configuration"]
        self.rotors = RotorGroup(rotor_config, dt=self.dt).to(self.device)

        rotor_params = make_functional(self.rotors)
        self.KF_0 = rotor_params["KF"].clone()
        self.KM_0 = rotor_params["KM"].clone()
        self.MAX_ROT_VEL = (
            torch.as_tensor(rotor_config["max_rotation_velocities"])
            .float()
            .to(self.device)
        )
        self.rotor_params = rotor_params.expand(self.shape).clone()

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]
        self.throttle = self.rotor_params["throttle"]
        self.directions = self.rotor_params["directions"]
        '''

        # used in `apply_action` func
        self.torques = torch.zeros(*self.shape, 3, device=self.device)
        self.forces = torch.zeros(*self.shape, 3, device=self.device)
        # self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)

        # used in `get_state` func
        self.pos, self.rot = self.get_world_poses(True)
        # self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.acc = self.acc_w = torch.zeros(*self.shape, 6, device=self.device)
        self.acc_b = torch.zeros_like(self.acc_w)

        # self.jerk = torch.zeros(*self.shape, 6, device=self.device)
        self.alpha = 0.9

        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        self.c_tfs = torch.full_like(self.masses, 2.2)  # thrust-to-weight coefficients
        # Default/initial parameters
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = (
            self.base_link
            .get_inertias()
            .reshape(*self.shape, 3, 3)[0]
            .diagonal(0, -2, -1)
            .clone()
        )
        self.c_tf_0 = self.c_tfs[0].clone()
        '''
        print(self.MASS_0, self.INERTIA_0, self.c_tf_0)
        tensor([[1.7540]], device='cuda:0') tensor([[0.0220, 0.0220, 0.0350]], device='cuda:0') tensor([[2.2000]], device='cuda:0')
        '''

        # self.THRUST2WEIGHT_0 = self.KF_0 / (self.MASS_0 * 9.81) # TODO: get the real g
        # self.FORCE2MOMENT_0 = torch.broadcast_to(self.KF_0 / self.KM_0, self.THRUST2WEIGHT_0.shape)

        logging.info(str(self))

        # self.drag_coef = torch.zeros(*self.shape, 1, device=self.device) * self.params["drag_coef"]
        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()
        # self.intrinsics = self.intrinsics_spec.expand(self.shape).clone()
        self.intrinsics["mass"] = self.MASS_0.expand(*self.shape, -1).clone()
        self.intrinsics["inertia"] = self.INERTIA_0.expand(*self.shape, -1).clone()
        self.intrinsics["c_tf"] = self.c_tf_0.expand(*self.shape, -1).clone()
        self.intrinsics["com"] = torch.zeros(*self.shape, 3, device=self.device).clone()


    def setup_randomization(self, cfg):
        if not self.initialized:
            raise RuntimeError

        for phase in ("train", "eval"):
            if phase not in cfg: continue
            mass_scale = cfg[phase].get("mass_scale", None)
            if mass_scale is not None:
                low = self.MASS_0 * mass_scale[0]
                high = self.MASS_0 * mass_scale[1]
                self.randomization[phase]["mass"] = D.Uniform(low, high)
                '''
                print(low, high, self.randomization[phase]["mass"])
                tensor([[0.4560]], device='cuda:0') tensor([[3.0520]], device='cuda:0') 
                Uniform(low: tensor([[0.4560]], device='cuda:0'), high: tensor([[3.0520]], device='cuda:0'))
                '''
            inertia_scale = cfg[phase].get("inertia_scale", None)
            if inertia_scale is not None:
                low = self.INERTIA_0 * torch.as_tensor(inertia_scale[0], device=self.device)
                high = self.INERTIA_0 * torch.as_tensor(inertia_scale[1], device=self.device)
                self.randomization[phase]["inertia"] = D.Uniform(low, high)
            com = cfg[phase].get("com", None)
            if com is not None:
                self.randomization[phase]["com"] = D.Uniform(
                    torch.tensor(com[0], device=self.device),
                    torch.tensor(com[1], device=self.device)
                )
            c_tf_scale = cfg[phase].get("c_tf_scale", None)
            if c_tf_scale is not None:
                low = self.c_tf_0 * c_tf_scale[0]
                high = self.c_tf_0 * c_tf_scale[1]
                self.randomization[phase]["c_tf"] = D.Uniform(low, high)
            '''
            t2w_scale = cfg[phase].get("t2w_scale", None)
            if t2w_scale is not None:
                low = self.THRUST2WEIGHT_0 * torch.as_tensor(t2w_scale[0], device=self.device)
                high = self.THRUST2WEIGHT_0 * torch.as_tensor(t2w_scale[1], device=self.device)
                self.randomization[phase]["thrust2weight"] = D.Uniform(low, high)
            f2m_scale = cfg[phase].get("f2m_scale", None)
            if f2m_scale is not None:
                low = self.FORCE2MOMENT_0 * torch.as_tensor(f2m_scale[0], device=self.device)
                high = self.FORCE2MOMENT_0 * torch.as_tensor(f2m_scale[1], device=self.device)
                self.randomization[phase]["force2moment"] = D.Uniform(low, high)
            drag_coef_scale = cfg[phase].get("drag_coef_scale", None)
            if drag_coef_scale is not None:
                low = self.params["drag_coef"] * drag_coef_scale[0]
                high = self.params["drag_coef"] * drag_coef_scale[1]
                self.randomization[phase]["drag_coef"] = D.Uniform(
                    torch.tensor(low, device=self.device),
                    torch.tensor(high, device=self.device)
                )
            tau_up = cfg[phase].get("tau_up", None)
            if tau_up is not None:
                self.randomization[phase]["tau_up"] = D.Uniform(
                    torch.tensor(tau_up[0], device=self.device),
                    torch.tensor(tau_up[1], device=self.device)
                )
            tau_down = cfg[phase].get("tau_down", None)
            if tau_down is not None:
                self.randomization[phase]["tau_down"] = D.Uniform(
                    torch.tensor(tau_down[0], device=self.device),
                    torch.tensor(tau_down[1], device=self.device)
                )
            '''
            if not len(self.randomization[phase]) == len(cfg[phase]):
                unkown_keys = set(cfg[phase].keys()) - set(self.randomization[phase].keys())
                raise ValueError(
                    f"Unknown randomization {unkown_keys}."
                )

        logging.info(f"Setup randomization:\n" + pprint.pformat(dict(self.randomization)))

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        # self.thrusts[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.forces[env_ids] = 0.0

        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        # self.jerk[env_ids] = 0.
        if train and "train" in self.randomization:
            self._randomize(env_ids, self.randomization["train"])
        elif "eval" in self.randomization:
            self._randomize(env_ids, self.randomization["eval"])
        # init_throttle = self.gravity[env_ids] / self.KF[env_ids].sum(-1, keepdim=True)
        # self.throttle[env_ids] = self.rotors.f_inv(init_throttle)
        # self.throttle_difference[env_ids] = 0.0
        return env_ids

    def _randomize(self, env_ids: torch.Tensor, distributions: Dict[str, D.Distribution]):
        shape = env_ids.shape
        if "mass" in distributions:
            masses = distributions["mass"].sample(shape)
            self.base_link.set_masses(masses, env_indices=env_ids)
            self.masses[env_ids] = masses
            self.gravity[env_ids] = masses * 9.81
            #self.intrinsics["mass"][env_ids] = (masses / self.MASS_0)  # normalized parameters
            '''
            print(self.masses)
            tensor([[[1.7410]],
                    [[1.7279]],
                    [[1.7630]]], device='cuda:0')
            '''
            self.intrinsics["mass"][env_ids] = masses
        if "inertia" in distributions:
            inertias = distributions["inertia"].sample(shape)
            self.inertias[env_ids] = inertias
            self.base_link.set_inertias(
                torch.diag_embed(inertias).flatten(-2), env_indices=env_ids
            )
            #self.intrinsics["inertia"][env_ids] = inertias / self.INERTIA_0  # normalized parameters
            self.intrinsics["inertia"][env_ids] = inertias  
        if "com" in distributions:
            coms = distributions["com"].sample((*shape, 3))
            self.base_link.set_coms(coms, env_indices=env_ids)
            self.intrinsics["com"][env_ids] = coms.reshape(*shape, 1, 3)
        if "c_tf" in distributions:
            c_tf = distributions["c_tf"].sample(shape)
            self.c_tfs[env_ids] = c_tf
            self.intrinsics["c_tf"][env_ids] = c_tf
        '''
        if "thrust2weight" in distributions:
            thrust2weight = distributions["thrust2weight"].sample(shape)
            KF = thrust2weight * self.masses[env_ids] * 9.81
            self.KF[env_ids] = KF
            self.intrinsics["KF"][env_ids] = KF / self.KF_0
        if "force2moment" in distributions:
            force2moment = distributions["force2moment"].sample(shape)
            KM = self.KF[env_ids] / force2moment
            self.KM[env_ids] = KM
            self.intrinsics["KM"][env_ids] = KM / self.KM_0
        if "drag_coef" in distributions:
            drag_coef = distributions["drag_coef"].sample(shape).reshape(-1, 1, 1)
            self.drag_coef[env_ids] = drag_coef
            self.intrinsics["drag_coef"][env_ids] = drag_coef
        if "tau_up" in distributions:
            tau_up = distributions["tau_up"].sample(shape+self.rotors_view.shape[1:])
            self.tau_up[env_ids] = tau_up
            self.intrinsics["tau_up"][env_ids] = tau_up
        if "tau_down" in distributions:
            tau_down = distributions["tau_down"].sample(shape+self.rotors_view.shape[1:])
            self.tau_down[env_ids] = tau_down
            self.intrinsics["tau_down"][env_ids] = tau_down
        '''

    def vmap_fMs(self, cmds, masses, c_tfs):
        # reg_M = 0.2 # TODO: # moment regularization scaling

        # normalized force and moment components [-1,1]
        f_norm, M = cmds[..., 0:1], cmds[..., 1:]#*reg_M  
        '''
        print("f_norm:", f_norm, "M:", M)
        f_norm: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[[-0.1070]],
                    [[ 1.0000]],
                    [[-0.6761]]], device='cuda:0')) 
        M: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[[ 0.7770, -1.0000, -0.9944]],
                    [[ 1.0000, -0.2193,  0.5457]],
                    [[-0.8583,  0.3920,  0.6740]]], device='cuda:0'))
        '''

        # Get per-env parameters
        total_masses = masses + (0.099*4)  # because each rotor's mass = 0.099
        '''
        print("total_masses:", total_masses)
        total_masses: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[2.2954],
                    [2.0683],
                    [2.2033]], device='cuda:0'))
        '''
        hover_force = total_masses*9.81/4.  # hovering thrust magnitude of each motor, [N]
        '''
        print("hover_force:", hover_force)
        hover_force: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[5.6295],
                    [5.0726],
                    [5.4035]], device='cuda:0'))
        '''
        max_force = c_tfs * hover_force  # maximum thrust of each motor, [N]
        min_force = torch.full_like(max_force, 0.5) # minimum thrust of each motor, 0.5 [N]
        '''
        print("max_force:", max_force, "min_force:", min_force)
        max_force: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[12.2633],
                    [10.2252],
                    [11.1215]], device='cuda:0')) 
        min_force: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[2.],
                    [2.],
                    [2.]], device='cuda:0'))
        '''

        # Compute per-env total thrust (denormalized from [-1, 1] to [min, max])
        avrg_act = (min_force + max_force)/2
        scale_act = max_force - avrg_act  # actor scaling
        '''
        print("avrg_act:", avrg_act, "scale_act:", scale_act)
        avrg_act: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[7.1317],
                    [6.1126],
                    [6.5608]], device='cuda:0')) 
        scale_act: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[5.1317],
                    [4.1126],
                    [4.5608]], device='cuda:0'))
        '''
        f_total = torch.clamp(4.*(scale_act*f_norm + avrg_act), 4.*min_force, 4.*max_force)
        '''
        print("f_total:", f_total)
        f_total: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[[26.3301]],
                    [[40.9006]],
                    [[13.9086]]], device='cuda:0'))
        '''
        # Ensure force is only in the z-direction in the drone's body frame
        f = torch.cat([torch.zeros_like(f_total), torch.zeros_like(f_total), f_total], dim=-1) # [N]
        '''
        print("f:", f)
        f: BatchedTensor(lvl=1, bdim=0, value=
            tensor([[[ 0.0000,  0.0000, 26.3301]],
                    [[ 0.0000,  0.0000, 40.9006]],
                    [[ 0.0000,  0.0000, 13.9086]]], device='cuda:0'))
        '''
        return f.unsqueeze(-2), M.unsqueeze(-2)  # Expand to match expected shape

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        fMs = actions.expand(*self.shape, self.num_rotors)  # [f, M]s from RL algos

        # Apply vmap to directly process force and moments per drone
        forces, torques = vmap(self.vmap_fMs)(
            fMs, self.masses.squeeze(-1), self.c_tfs.squeeze(-1))
        forces = forces.squeeze(-3)  # ensure [num_envs, 1, 3]
        torques = torques.squeeze(-3)  # ensure [num_envs, 1, 3]

        # Store force and torque in the simulation
        self.forces[:] = forces
        self.torques[:] = torques
        '''
        print("self.forces:", self.forces, "self.torques:", self.torques)
        self.forces: tensor([[[ 0.0000,  0.0000, 26.3301]],
                [[ 0.0000,  0.0000, 40.9006]],
                [[ 0.0000,  0.0000, 13.9086]]], device='cuda:0') 
        self.torques: tensor([[[ 0.7770, -1.0000, -0.9944]],
                [[ 1.0000, -0.2193,  0.5457]],
                [[-0.8583,  0.3920,  0.6740]]], device='cuda:0')
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


    """
    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
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
        """
