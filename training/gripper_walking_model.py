"""Latest local pick-place MJCF with the legacy walking policy interface."""
import copy
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.robots.pal_kangaroo import kangaroo_constants as kc

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'assets/kangaroo_pick_place_63ba43a/kangaroo_grippers.xml'
LEGACY_JOINTS = tuple(j.attrib['name'] for j in ET.parse(ROOT / 'assets/kangaroo/xmls/kangaroo.xml').findall('.//worldbody//joint') if j.attrib.get('type') != 'free')

def gripper_spec():
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for j in spec.joints:
        if j.name.startswith('gripper_') and not re.fullmatch(r'gripper_(left|right)_finger_joint', j.name):
            j.armature = 0.005  # Pick-place tabletop linkage stabilization.
    return spec

def apply_gripper_model(cfg):
    robot = copy.deepcopy(kc.get_kangaroo_grippers_robot_cfg())
    robot.spec_fn = gripper_spec
    robot.articulation.actuators += (BuiltinPositionActuatorCfg(
        target_names_expr=(r'gripper_.*_finger_joint',), stiffness=8., damping=.5,
        effort_limit=20., armature=1e-4),)
    cfg.scene.entities['robot'] = robot
    # Original regex also matches arm 5 on the new model; pin original actuated names.
    action = cfg.actions['joint_pos']
    names = tuple(n for n in LEGACY_JOINTS if any(re.fullmatch(rx,n) for rx in action.actuator_names))
    action.scale = {n: next(v for rx,v in action.scale.items() if re.fullmatch(rx,n)) for n in names}
    action.actuator_names = names
    for group in cfg.observations.values():
        for name in ('joint_pos','joint_vel'):
            group.terms[name].params['asset_cfg'] = SceneEntityCfg('robot',joint_names=LEGACY_JOINTS,preserve_order=True)
    pose = cfg.rewards['pose'].params
    pose['asset_cfg'] = SceneEntityCfg('robot', joint_names=names, preserve_order=True)
    for key in ('std_standing', 'std_walking', 'std_running'):
        pose[key] = {n: next(v for rx,v in pose[key].items() if re.fullmatch(rx,n)) for n in names}
    # New wrist/jaw targets remain zero (their nominal pose) after reset; original
    # arm/pelvis targets continue to be written by the walking policy every step.
    return cfg
