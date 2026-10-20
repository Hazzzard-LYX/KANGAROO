"""Derive consistent gripper linkage coordinates for the saved tray handle pose."""
import json
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from gripper_walking_model import ROOT, gripper_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-gap-m", type=float, default=0.046,
        help="Distance between fingertip sites; does not by itself validate collision clearance or grasp force.",
    )
    args = parser.parse_args()
    # Preserve the calibrated arm placement when recomputing gripper linkage
    # angles. The standalone editor pose is only the fallback for a fresh setup.
    pose_path = ROOT / "assets/tray/carry_pose.json"
    if pose_path.is_file():
        pose = json.loads(pose_path.read_text())
    else:
        pose = json.loads((ROOT / "assets/tray/initial_pose.json").read_text())
    spec = gripper_spec()
    terrain = spec.worldbody.add_body(name="terrain")
    terrain.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(3.0, 3.0, 0.1))
    model = spec.compile()
    data = mujoco.MjData(model)
    for name, value in pose["joint_positions_rad"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        ids = [j for j in range(model.njnt) if model.joint(j).name.startswith(f"gripper_{side}_")]
        adr = model.jnt_qposadr[ids]
        low, high = model.jnt_range[ids, 0] + 1e-6, model.jnt_range[ids, 1] - 1e-6
        pads = [model.site(f"gripper_{side}_fingertip_{finger}_site").id for finger in ("left", "right")]
        eq_ids = []
        for eq in range(model.neq):
            if model.eq_type[eq] == mujoco.mjtEq.mjEQ_CONNECT:
                if model.body(model.eq_obj1id[eq]).name.startswith(f"gripper_{side}_"):
                    eq_ids.append(eq)
            elif model.eq_type[eq] == mujoco.mjtEq.mjEQ_JOINT:
                if model.joint(model.eq_obj1id[eq]).name.startswith(f"gripper_{side}_"):
                    eq_ids.append(eq)

        def residual(q):
            data.qpos[adr] = q
            mujoco.mj_forward(model, data)
            equality = data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            selected = equality & np.isin(data.efc_id[:data.nefc], eq_ids)
            gap = np.linalg.norm(data.site_xpos[pads[0]] - data.site_xpos[pads[1]])
            return np.r_[20.0 * data.efc_pos[:data.nefc][selected], gap - args.target_gap_m]

        result = least_squares(residual, np.clip(data.qpos[adr], low, high), bounds=(low, high), max_nfev=600)
        residual(result.x)
        actual_gap = float(np.linalg.norm(data.site_xpos[pads[0]] - data.site_xpos[pads[1]]))
        equality = data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        selected = equality & np.isin(data.efc_id[:data.nefc], eq_ids)
        print(f"{side}: target gap={args.target_gap_m * 1000:.1f} mm, fitted={actual_gap * 1000:.2f} mm, max equality residual={np.max(np.abs(data.efc_pos[:data.nefc][selected])):.5f}")
        for joint_id, value in zip(ids, result.x, strict=True):
            pose["joint_positions_rad"][model.joint(joint_id).name] = float(value)
    pose["target_handle_gap_m"] = args.target_gap_m
    pose["note"] = "Calibrated arm pose; gripper linkage solved for the specified fingertip-site gap. Static fit only; stable physical grasp is not implied."
    pose_path.write_text(json.dumps(pose, indent=2))


if __name__ == "__main__":
    main()
