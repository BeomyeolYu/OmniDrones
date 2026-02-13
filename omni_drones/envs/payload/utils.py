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
import omni.physx.scripts.utils as script_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import omni.isaac.core.objects as objects

import torch
import omni_drones.utils.kit as kit_utils
from ..utils import create_bar


def attach_payload(
    drone_prim_path: str,
    inverted_pendulum: bool,
    bar_length: float,
    payload_radius: float = 0.04,
    payload_mass: float = 0.3,
    drone_scale: float = 1.35
):
    # Compensate the bar's size and offsets
    scaled_bar_length = bar_length / drone_scale
    if inverted_pendulum:
        scaled_translation = scaled_bar_length / 2.0
        scaled_payload_offset = scaled_bar_length
    else:
        scaled_translation = -scaled_bar_length / 2.0
        scaled_payload_offset = -scaled_bar_length

    bar = prim_utils.create_prim(
        prim_path=drone_prim_path + "/bar",
        prim_type="Capsule",
        translation=(0., 0., scaled_translation),
        attributes={"radius": 0.01, "height": scaled_bar_length}
    )
    bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
    UsdPhysics.RigidBodyAPI.Apply(bar)
    UsdPhysics.CollisionAPI.Apply(bar)
    massAPI = UsdPhysics.MassAPI.Apply(bar)
    massAPI.CreateMassAttr().Set(0.001)
    massAPI.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-4, 1e-4, 5e-4))

    base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "D6", bar, base_link)
    
    UsdPhysics.DriveAPI.Apply(joint, "rotX")
    UsdPhysics.DriveAPI.Apply(joint, "rotY")
    UsdPhysics.DriveAPI.Apply(joint, "rotZ")
    '''
    joint.GetAttribute("limit:rotX:physics:low").Set(-120)  # joint angles
    joint.GetAttribute("limit:rotX:physics:high").Set(120)
    joint.GetAttribute("limit:rotY:physics:low").Set(-120)
    joint.GetAttribute("limit:rotY:physics:high").Set(120)
    joint.GetAttribute("limit:rotZ:physics:low").Set(-120)
    joint.GetAttribute("limit:rotZ:physics:high").Set(120)
    joint.GetAttribute("drive:rotX:physics:damping").Set(2e-4)
    joint.GetAttribute("drive:rotY:physics:damping").Set(2e-4)
    joint.GetAttribute("drive:rotZ:physics:damping").Set(2e-4)
    joint.GetAttribute("drive:rotX:physics:stiffness").Set(1e-5)
    joint.GetAttribute("drive:rotY:physics:stiffness").Set(1e-5)
    joint.GetAttribute("drive:rotZ:physics:stiffness").Set(1e-5)
    '''
    joint.GetAttribute("limit:rotX:physics:low").Set(-torch.inf) 
    joint.GetAttribute("limit:rotX:physics:high").Set(torch.inf)
    joint.GetAttribute("limit:rotY:physics:low").Set(-torch.inf)
    joint.GetAttribute("limit:rotY:physics:high").Set(torch.inf)
    joint.GetAttribute("limit:rotZ:physics:low").Set(-torch.inf)
    joint.GetAttribute("limit:rotZ:physics:high").Set(torch.inf)
    joint.GetAttribute("drive:rotX:physics:damping").Set(0)
    joint.GetAttribute("drive:rotY:physics:damping").Set(0)
    joint.GetAttribute("drive:rotZ:physics:damping").Set(0)
    joint.GetAttribute("drive:rotX:physics:stiffness").Set(0)
    joint.GetAttribute("drive:rotY:physics:stiffness").Set(0)
    joint.GetAttribute("drive:rotZ:physics:stiffness").Set(0)
    

    payload = objects.DynamicSphere(
        prim_path=drone_prim_path + "/payload",
        translation=(0., 0., scaled_payload_offset),
        radius=payload_radius / drone_scale,  # optional: also shrink radius
        mass=payload_mass
    )
    joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)

    kit_utils.set_collision_properties(drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0)
    kit_utils.set_collision_properties(drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0)

    '''
    # Add a single cube to visualize z-axis rotation
    cube_offset = -scaled_payload_offset/2  # Offset from bar center along X-axis
    cube_size = (1.0, 0.01, 0.01)  # Size of the cube

    cube_path = f"{drone_prim_path}/bar/rotation_marker"
    cube = prim_utils.create_prim(
        prim_path=cube_path,
        prim_type="Cube",
        translation=(0.0, 0.0, cube_offset),  # Offset on Z-axis
        scale=cube_size,
        attributes={"size": 1.0}
    )
    cube.GetAttribute('primvars:displayColor').Set([(0., 1., 0.)])

    # Connect the cube to the bar using a fixed joint
    script_utils.createJoint(stage, "Fixed", bar, cube)
    '''

# def attach_payload(
#     drone_prim_path: str,
#     inverted_pendulum: bool,
#     bar_length: float,
#     payload_radius: float = 0.04,
#     payload_mass: float = 0.3,
#     drone_scale: float = 1.35,
#     com_offset: tuple[float, float, float] = (0.02, 0.02, 0.02)  # NEW: COM offset
# ):
#     # Compensate bar length and offsets
#     scaled_bar_length = bar_length / drone_scale
#     if inverted_pendulum:
#         scaled_translation = scaled_bar_length / 2.0
#         scaled_payload_offset = scaled_bar_length
#     else:
#         scaled_translation = -scaled_bar_length / 2.0
#         scaled_payload_offset = -scaled_bar_length

#     # Include CoM offset
#     scaled_translation += com_offset[2]/drone_scale  # adjust z translation
#     scaled_payload_offset += com_offset[2]/drone_scale

#     # Create the bar relative to CoM
#     bar = prim_utils.create_prim(
#         prim_path=drone_prim_path + "/bar",
#         prim_type="Capsule",
#         translation=(com_offset[0]/ drone_scale, com_offset[1]/ drone_scale, scaled_translation),
#         attributes={"radius": 0.01, "height": scaled_bar_length}
#     )
#     bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
#     UsdPhysics.RigidBodyAPI.Apply(bar)
#     UsdPhysics.CollisionAPI.Apply(bar)
#     massAPI = UsdPhysics.MassAPI.Apply(bar)
#     massAPI.CreateMassAttr().Set(0.001)
#     massAPI.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-4, 1e-4, 5e-4))

#     base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
#     stage = prim_utils.get_current_stage()
#     joint = script_utils.createJoint(stage, "D6", bar, base_link)
    
#     UsdPhysics.DriveAPI.Apply(joint, "rotX")
#     UsdPhysics.DriveAPI.Apply(joint, "rotY")
#     UsdPhysics.DriveAPI.Apply(joint, "rotZ")
#     joint.GetAttribute("limit:rotX:physics:low").Set(-torch.inf) 
#     joint.GetAttribute("limit:rotX:physics:high").Set(torch.inf)
#     joint.GetAttribute("limit:rotY:physics:low").Set(-torch.inf)
#     joint.GetAttribute("limit:rotY:physics:high").Set(torch.inf)
#     joint.GetAttribute("limit:rotZ:physics:low").Set(-torch.inf)
#     joint.GetAttribute("limit:rotZ:physics:high").Set(torch.inf)
#     joint.GetAttribute("drive:rotX:physics:damping").Set(0)
#     joint.GetAttribute("drive:rotY:physics:damping").Set(0)
#     joint.GetAttribute("drive:rotZ:physics:damping").Set(0)
#     joint.GetAttribute("drive:rotX:physics:stiffness").Set(0)
#     joint.GetAttribute("drive:rotY:physics:stiffness").Set(0)
#     joint.GetAttribute("drive:rotZ:physics:stiffness").Set(0)

#     # Create payload relative to CoM
#     payload = objects.DynamicSphere(
#         prim_path=drone_prim_path + "/payload",
#         translation=(com_offset[0]/ drone_scale, com_offset[1]/ drone_scale, scaled_payload_offset),
#         radius=payload_radius / drone_scale,
#         mass=payload_mass
#     )
#     # Fixed joint between bar and payload
#     script_utils.createJoint(stage, "Fixed", bar, payload.prim)

#     # Collision properties
#     kit_utils.set_collision_properties(drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0)
#     kit_utils.set_collision_properties(drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0)



# def attach_payload(
#     drone_prim_path: str,
#     inverted_pendulum: bool,
#     bar_length: float,
#     payload_radius: float = 0.04,
#     payload_mass: float = 0.3,
#     drone_scale: float = 1.35
# ):
#     # Compensate the bar's size and offsets
#     scaled_bar_length = bar_length / drone_scale
#     if inverted_pendulum:
#         scaled_translation = scaled_bar_length / 2.0
#         scaled_payload_offset = scaled_bar_length
#     else:
#         scaled_translation = -scaled_bar_length / 2.0
#         scaled_payload_offset = -scaled_bar_length

#     # Create the bar (a capsule attached below the drone)
#     bar = prim_utils.create_prim(
#         prim_path=drone_prim_path + "/bar",
#         prim_type="Capsule",
#         translation=(0., 0., scaled_translation),
#         attributes={"radius": 0.01, "height": scaled_bar_length}
#     )
#     bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
#     UsdPhysics.RigidBodyAPI.Apply(bar)
#     UsdPhysics.CollisionAPI.Apply(bar)
#     massAPI = UsdPhysics.MassAPI.Apply(bar)
#     massAPI.CreateMassAttr().Set(0.001)

#     # Create a spherical joint between base_link and bar
#     base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
#     stage = prim_utils.get_current_stage()

#     # Replace D6 joint with a Spherical joint
#     spherical_joint = script_utils.createJoint(stage, "Spherical", bar, base_link)

#     # Optional: Add damping via drive API if needed (not always supported for spherical)
#     UsdPhysics.DriveAPI.Apply(spherical_joint, "angular")
#     spherical_joint.GetAttribute("drive:angular:physics:damping").Set(0.0)
#     spherical_joint.GetAttribute("drive:angular:physics:stiffness").Set(0.0)

#     # Create payload (a sphere attached to end of bar)
#     payload = objects.DynamicSphere(
#         prim_path=drone_prim_path + "/payload",
#         translation=(0., 0., scaled_payload_offset),
#         radius=payload_radius / drone_scale,
#         mass=payload_mass
#     )

#     # Attach payload to bar with a fixed joint
#     script_utils.createJoint(stage, "Fixed", bar, payload.prim)

#     # Set collision properties for physical contact stability
#     kit_utils.set_collision_properties(drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0)
#     kit_utils.set_collision_properties(drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0)


# def attach_payload(
#     drone_prim_path: str,
#     bar_length: str,
#     payload_radius: float=0.04,
#     payload_mass: float=0.3
# ):

#     bar = prim_utils.create_prim(
#         prim_path=drone_prim_path + "/bar",
#         prim_type="Capsule",
#         translation=(0., 0., -bar_length / 2.),
#         attributes={"radius": 0.01, "height": bar_length}
#     )
#     bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
#     UsdPhysics.RigidBodyAPI.Apply(bar)
#     UsdPhysics.CollisionAPI.Apply(bar)
#     massAPI = UsdPhysics.MassAPI.Apply(bar)
#     massAPI.CreateMassAttr().Set(0.001)

#     base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
#     stage = prim_utils.get_current_stage()
#     joint = script_utils.createJoint(stage, "D6", bar, base_link)
#     joint.GetAttribute("limit:rotX:physics:low").Set(-120)  # joint angles
#     joint.GetAttribute("limit:rotX:physics:high").Set(120)
#     joint.GetAttribute("limit:rotY:physics:low").Set(-120)
#     joint.GetAttribute("limit:rotY:physics:high").Set(120)
#     UsdPhysics.DriveAPI.Apply(joint, "rotX")
#     UsdPhysics.DriveAPI.Apply(joint, "rotY")
#     joint.GetAttribute("drive:rotX:physics:damping").Set(2e-6)
#     joint.GetAttribute("drive:rotY:physics:damping").Set(2e-6)
#     # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

#     payload = objects.DynamicSphere(
#         prim_path=drone_prim_path + "/payload",
#         translation=(0., 0., -bar_length),
#         radius=payload_radius,
#         mass=payload_mass
#     )
#     joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)
#     kit_utils.set_collision_properties(
#         drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0
#     )
#     kit_utils.set_collision_properties(
#         drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0
#     )


# def attach_payload(
#     drone_prim_path: str,
#     bar_length: float,
#     payload_radius: float=0.04,
#     payload_mass: float=0.3
# ):

#     payload = objects.DynamicSphere(
#         prim_path=drone_prim_path + "/payload",
#         translation=(0., 0., -bar_length),
#         radius=payload_radius,
#         mass=payload_mass
#     )
#     create_bar(
#         drone_prim_path + "/bar",
#         length=bar_length,
#         from_prim=drone_prim_path + "/base_link",
#         to_prim=drone_prim_path + "/payload",
#         joint_to_attributes={}
#     )
#     kit_utils.set_collision_properties(
#         drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0
#     )

