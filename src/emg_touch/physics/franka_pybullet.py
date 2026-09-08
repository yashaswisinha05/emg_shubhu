"""Transfer live model end-effector pose and grasp state to PyBullet Franka."""
from __future__ import annotations

import numpy as np

from .rotation_6d import (matrix_to_quaternion_numpy,
                          quaternion_to_matrix_numpy)


def normalized_quaternion(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = np.linalg.norm(result)
    if norm < 1e-8:
        raise ValueError("quaternion norm is zero")
    return result / norm


class PoseMapper:
    """Rigid VIVE-world to Panda-base mapping, explicit or first-pose anchored."""
    def __init__(self, translation=None, quaternion_wxyz=None,
                 home_position=(.45, 0., .50), home_quaternion_wxyz=(0., 1., 0., 0.)):
        if (translation is None) != (quaternion_wxyz is None):
            raise ValueError("explicit calibration needs both translation and quaternion")
        self.translation = None if translation is None else np.asarray(translation, dtype=float)
        if self.translation is not None and (self.translation.shape != (3,) or
                                              not np.isfinite(self.translation).all()):
            raise ValueError("calibration translation must contain three finite values")
        self.calibration_rotation = (None if quaternion_wxyz is None else
                                     quaternion_to_matrix_numpy(normalized_quaternion(
                                         quaternion_wxyz)))
        self.home_position = np.asarray(home_position, dtype=float)
        self.home_rotation = quaternion_to_matrix_numpy(normalized_quaternion(
            home_quaternion_wxyz))
        if self.home_position.shape != (3,) or not np.isfinite(self.home_position).all():
            raise ValueError("home position must contain three finite values")
        self.reset()

    def reset(self):
        self.first_position = self.first_rotation = None

    def map(self, position, quaternion_wxyz):
        position = np.asarray(position, dtype=float)
        rotation = quaternion_to_matrix_numpy(normalized_quaternion(quaternion_wxyz))
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("model position must contain three finite values")
        if self.calibration_rotation is not None:
            mapped_position = self.calibration_rotation @ position + self.translation
            mapped_rotation = self.calibration_rotation @ rotation
            mode = "calibrated VIVE-to-Panda transform"
        else:
            if self.first_position is None:
                self.first_position, self.first_rotation = position.copy(), rotation.copy()
            relative_position = self.first_rotation.T @ (position - self.first_position)
            relative_rotation = self.first_rotation.T @ rotation
            mapped_position = self.home_position + self.home_rotation @ relative_position
            mapped_rotation = self.home_rotation @ relative_rotation
            mode = "synthetic first-pose anchor"
        return mapped_position, matrix_to_quaternion_numpy(mapped_rotation), mode


def quaternion_angle_degrees(first_wxyz, second_wxyz):
    first, second = normalized_quaternion(first_wxyz), normalized_quaternion(second_wxyz)
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(first, second)), 0, 1))))


class LiveFrankaPyBulletController:
    """7-DoF pose IK and Panda finger control driven by model predictions."""
    REST = np.array([0., -.4, 0., -2.2, 0., 2.0, .8])

    def __init__(self, gui=True, mapper=None, base_position=(0., 0., 0.),
                 simulation_steps=10, open_width_m=.04, closed_width_m=0.,
                 bullet=None, data_path=None):
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be positive")
        if not 0 <= closed_width_m < open_width_m <= .04:
            raise ValueError("Panda finger positions require 0 <= closed < open <= 0.04 m")
        if bullet is None:
            try:
                import pybullet as bullet
                import pybullet_data
            except ImportError as error:
                raise RuntimeError("PyBullet is required: pip install pybullet") from error
            data_path = pybullet_data.getDataPath()
        self.p = bullet
        self.client = self.p.connect(self.p.GUI if gui else self.p.DIRECT)
        if self.client < 0:
            raise RuntimeError("could not connect to PyBullet")
        if data_path is not None:
            self.p.setAdditionalSearchPath(str(data_path), physicsClientId=self.client)
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.p.setTimeStep(1 / 240, physicsClientId=self.client)
        self.robot = self.p.loadURDF("franka_panda/panda.urdf",
            basePosition=list(base_position), useFixedBase=True,
            physicsClientId=self.client)
        self.mapper = mapper or PoseMapper()
        self.simulation_steps = int(simulation_steps)
        self.widths = {"open": float(open_width_m), "closed": float(closed_width_m)}
        self._discover_joints()
        self.reset()

    def _discover_joints(self):
        joint_by_name, link_by_name = {}, {}
        for index in range(self.p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = self.p.getJointInfo(self.robot, index, physicsClientId=self.client)
            joint_by_name[info[1].decode()] = (index, info)
            link_by_name[info[12].decode()] = index
        arm_names = [f"panda_joint{index}" for index in range(1, 8)]
        finger_names = ["panda_finger_joint1", "panda_finger_joint2"]
        missing = [name for name in arm_names + finger_names if name not in joint_by_name]
        if "panda_grasptarget" not in link_by_name:
            missing.append("panda_grasptarget link")
        if missing:
            raise RuntimeError("unexpected Franka URDF; missing: " + ", ".join(missing))
        self.arm_joints = [joint_by_name[name][0] for name in arm_names]
        self.finger_joints = [joint_by_name[name][0] for name in finger_names]
        self.ee_link = link_by_name["panda_grasptarget"]
        infos = [joint_by_name[name][1] for name in arm_names]
        self.lower = np.array([info[8] for info in infos])
        self.upper = np.array([info[9] for info in infos])
        self.ranges = self.upper - self.lower

    def reset(self):
        self.mapper.reset()
        self.gripper = "open"
        for joint, value in zip(self.arm_joints, self.REST):
            self.p.resetJointState(self.robot, joint, float(value),
                                   physicsClientId=self.client)
        for joint in self.finger_joints:
            self.p.resetJointState(self.robot, joint, self.widths["open"],
                                   physicsClientId=self.client)

    @staticmethod
    def _xyzw(wxyz):
        w, x, y, z = wxyz
        return [float(x), float(y), float(z), float(w)]

    @staticmethod
    def _wxyz(xyzw):
        x, y, z, w = xyzw
        return np.array([w, x, y, z], dtype=float)

    def attach(self, prediction):
        command = "hold"
        ordered = [(prediction["trigger_time_s"].get(name), name)
                   for name in ("grasp", "release")
                   if prediction["triggered"].get(name)
                   and prediction["trigger_time_s"].get(name) is not None]
        for _, name in sorted(ordered, key=lambda item: item[0]):
            self.gripper = "closed" if name == "grasp" else "open"
            command = "close" if name == "grasp" else "open"
        prediction["gripper"] = {"state": self.gripper, "command": command,
                                  "finger_joint_position_m": self.widths[self.gripper]}
        if not prediction["valid"] or prediction["position_m"] is None:
            prediction["franka"] = None
            return prediction
        position = np.array([prediction["position_m"][axis] for axis in "xyz"])
        quaternion = prediction["orientation_quaternion_wxyz"]
        target_position, target_quaternion, mode = self.mapper.map(position, quaternion)
        solution = self.p.calculateInverseKinematics(
            self.robot, self.ee_link, targetPosition=target_position.tolist(),
            targetOrientation=self._xyzw(target_quaternion),
            lowerLimits=self.lower.tolist(), upperLimits=self.upper.tolist(),
            jointRanges=self.ranges.tolist(), restPoses=self.REST.tolist(),
            jointDamping=[.1] * 7, maxNumIterations=100, residualThreshold=1e-5,
            physicsClientId=self.client)
        target_joints = np.clip(np.asarray(solution[:7]), self.lower, self.upper)
        self.p.setJointMotorControlArray(self.robot, self.arm_joints,
            self.p.POSITION_CONTROL, targetPositions=target_joints.tolist(),
            forces=[87.] * 7, physicsClientId=self.client)
        finger = self.widths[self.gripper]
        self.p.setJointMotorControlArray(self.robot, self.finger_joints,
            self.p.POSITION_CONTROL, targetPositions=[finger, finger],
            forces=[20., 20.], physicsClientId=self.client)
        for _ in range(self.simulation_steps):
            self.p.stepSimulation(physicsClientId=self.client)
        actual_joints = [self.p.getJointState(self.robot, joint,
            physicsClientId=self.client)[0] for joint in self.arm_joints]
        state = self.p.getLinkState(self.robot, self.ee_link,
            computeForwardKinematics=True, physicsClientId=self.client)
        actual_position = np.asarray(state[4], dtype=float)
        actual_quaternion = self._wxyz(state[5])
        prediction["franka"] = {
            "mapping": mode, "target_position_m": target_position.tolist(),
            "target_orientation_wxyz": target_quaternion.tolist(),
            "target_joint_angles_deg": np.degrees(target_joints).tolist(),
            "actual_joint_angles_deg": np.degrees(actual_joints).tolist(),
            "actual_ee_position_m": actual_position.tolist(),
            "actual_ee_orientation_wxyz": actual_quaternion.tolist(),
            "position_tracking_error_cm": float(100 * np.linalg.norm(
                actual_position - target_position)),
            "orientation_tracking_error_deg": quaternion_angle_degrees(
                actual_quaternion, target_quaternion)}
        return prediction

    def close(self):
        if getattr(self, "client", -1) >= 0:
            self.p.disconnect(physicsClientId=self.client)
            self.client = -1
