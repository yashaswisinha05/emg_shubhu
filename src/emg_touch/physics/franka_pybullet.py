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
                 home_position=(.45, 0., .50), home_quaternion_wxyz=(0., 1., 0., 0.),
                 trajectory_z_rotation_deg=0.):
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
        angle = np.radians(float(trajectory_z_rotation_deg))
        if not np.isfinite(angle):
            raise ValueError("trajectory z rotation must be finite")
        cosine, sine = np.cos(angle), np.sin(angle)
        self.trajectory_rotation = np.array([
            [cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        self.trajectory_z_rotation_deg = float(trajectory_z_rotation_deg)
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
        # Rotate the complete robot-frame trajectory about its anchored start,
        # and rotate end-effector orientation by the same world-frame rotation.
        mapped_position = (self.home_position + self.trajectory_rotation
                           @ (mapped_position - self.home_position))
        mapped_rotation = self.trajectory_rotation @ mapped_rotation
        if abs(self.trajectory_z_rotation_deg) > 1e-9:
            mode += f" + z rotation {self.trajectory_z_rotation_deg:+g} deg"
        return mapped_position, matrix_to_quaternion_numpy(mapped_rotation), mode


def quaternion_angle_degrees(first_wxyz, second_wxyz):
    first, second = normalized_quaternion(first_wxyz), normalized_quaternion(second_wxyz)
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(first, second)), 0, 1))))


class LiveFrankaPyBulletController:
    """7-DoF pose IK and Panda finger control driven by model predictions."""
    REST = np.array([0., -.4, 0., -2.2, 0., 2.0, .8])

    def __init__(self, gui=True, mapper=None, base_position=(0., 0., 0.),
                 simulation_steps=24, open_width_m=.04, closed_width_m=0.,
                 grasp_probability_threshold=.9, release_probability_threshold=.9,
                 holding_close_threshold=.8, holding_open_threshold=.2,
                 holding_persistence_frames=2,
                 bullet=None, data_path=None, motion_filter=None):
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be positive")
        if not 0 <= closed_width_m < open_width_m <= .04:
            raise ValueError("Panda finger positions require 0 <= closed < open <= 0.04 m")
        if not (0 <= grasp_probability_threshold <= 1
                and 0 <= release_probability_threshold <= 1):
            raise ValueError("grasp and release thresholds must be between 0 and 1")
        if not 0 <= holding_open_threshold < holding_close_threshold <= 1:
            raise ValueError("holding thresholds require 0 <= open < close <= 1")
        if holding_persistence_frames < 1:
            raise ValueError("holding persistence must be positive")
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
        self.motion_filter = motion_filter
        self.simulation_steps = int(simulation_steps)
        self.widths = {"open": float(open_width_m), "closed": float(closed_width_m)}
        self.event_thresholds = {"grasp": float(grasp_probability_threshold),
                                 "release": float(release_probability_threshold)}
        self.holding_thresholds = {"close": float(holding_close_threshold),
                                   "open": float(holding_open_threshold)}
        self.holding_persistence_frames = int(holding_persistence_frames)
        self.debug_items = []
        self.reference_positions = None
        self._discover_joints()
        self.reset()

    def _discover_joints(self):
        joint_by_name, link_by_name, joint_records = {}, {}, []
        for index in range(self.p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = self.p.getJointInfo(self.robot, index, physicsClientId=self.client)
            joint_by_name[info[1].decode()] = (index, info)
            link_by_name[info[12].decode()] = index
            joint_records.append((index, info))
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
        # Bullet expects null-space and damping arrays for every movable DoF
        # in the body (seven arm joints plus both Panda fingers), even though
        # the grasp-target kinematic chain itself ends above the fingers.
        fixed_type = getattr(self.p, "JOINT_FIXED", 4)
        active = [(index, info) for index, info in joint_records
                  if info[2] != fixed_type]
        active_indices = [index for index, _ in active]
        self.arm_solution_indices = [active_indices.index(joint)
                                     for joint in self.arm_joints]
        rest_by_joint = dict(zip(self.arm_joints, self.REST))
        rest_by_joint.update({joint: self.widths["open"]
                              for joint in self.finger_joints})
        self.ik_lower = np.array([info[8] for _, info in active], dtype=float)
        self.ik_upper = np.array([info[9] for _, info in active], dtype=float)
        self.ik_ranges = self.ik_upper - self.ik_lower
        self.ik_rest = np.array([rest_by_joint.get(index, 0.)
                                 for index, _ in active], dtype=float)

    def reset(self):
        if hasattr(self.p, "removeAllUserDebugItems"):
            self.p.removeAllUserDebugItems(physicsClientId=self.client)
        self.mapper.reset()
        if self.motion_filter is not None:
            self.motion_filter.reset()
        self.gripper = "open"
        self._holding_candidate = None
        self._holding_candidate_frames = 0
        self.last_target = None
        self.last_request = None
        self.previous_model_position = None
        self.previous_actual_position = None
        self.reference_drawn = False
        self.status_text = -1
        for joint, value in zip(self.arm_joints, self.REST):
            self.p.resetJointState(self.robot, joint, float(value),
                                   physicsClientId=self.client)
        for joint in self.finger_joints:
            self.p.resetJointState(self.robot, joint, self.widths["open"],
                                   physicsClientId=self.client)
        self._update_status("OPEN", [0., .7, 0.])
        self._debug_text("BLACK: withheld VIVE   CYAN: model request   ORANGE: Franka EE",
                         [-.35, 0., 1.15], [.1, .1, .1], 1.25)

    def set_reference_trajectory(self, positions):
        """Store withheld VIVE XYZ for a black comparison-only GUI trace."""
        values = np.asarray(positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("reference trajectory must have shape [frames, 3]")
        self.reference_positions = values[np.isfinite(values).all(axis=1)]
        self.reference_drawn = False

    def _debug_line(self, first, second, color, width=3.):
        if not hasattr(self.p, "addUserDebugLine"):
            return
        item = self.p.addUserDebugLine(
            np.asarray(first).tolist(), np.asarray(second).tolist(), color,
            lineWidth=width, lifeTime=0, physicsClientId=self.client)
        self.debug_items.append(item)

    def _debug_text(self, text, position, color, size=1.5, replace=-1):
        if not hasattr(self.p, "addUserDebugText"):
            return replace
        kwargs = {"textColorRGB": color, "textSize": size, "lifeTime": 0,
                  "physicsClientId": self.client}
        if replace >= 0:
            kwargs["replaceItemUniqueId"] = replace
        item = self.p.addUserDebugText(text, np.asarray(position).tolist(), **kwargs)
        self.debug_items.append(item)
        return item

    def _update_status(self, text, color):
        self.status_text = self._debug_text(
            f"GRIPPER: {text}", [-.25, 0., 1.05], color, 2.5,
            getattr(self, "status_text", -1))

    def _marker(self, position, color, label):
        position = np.asarray(position, dtype=float)
        radius = .025
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = radius
            self._debug_line(position - offset, position + offset, color, 5.)
        self._debug_text(label, position + np.array([0., 0., .035]), color, 1.4)

    def _draw_reference(self):
        if self.reference_drawn or self.reference_positions is None:
            return
        mapped = []
        for position in self.reference_positions:
            value, _, _ = self.mapper.map(position, (1., 0., 0., 0.))
            mapped.append(value)
        for first, second in zip(mapped, mapped[1:]):
            self._debug_line(first, second, [.05, .05, .05], 4.)
        if mapped:
            self._marker(mapped[0], [.1, .1, .1], "VIVE START")
        self.reference_drawn = True

    @staticmethod
    def _xyzw(wxyz):
        w, x, y, z = wxyz
        return [float(x), float(y), float(z), float(w)]

    @staticmethod
    def _wxyz(xyzw):
        x, y, z, w = xyzw
        return np.array([w, x, y, z], dtype=float)

    def _command_gripper(self):
        finger = self.widths[self.gripper]
        self.p.setJointMotorControlArray(
            self.robot, self.finger_joints, self.p.POSITION_CONTROL,
            targetPositions=[finger, finger], forces=[40., 40.],
            physicsClientId=self.client)

    def _step(self, steps=None):
        for _ in range(self.simulation_steps if steps is None else int(steps)):
            self.p.stepSimulation(physicsClientId=self.client)

    def _finger_positions(self):
        return [float(self.p.getJointState(self.robot, joint,
            physicsClientId=self.client)[0]) for joint in self.finger_joints]

    def settle(self, steps=240):
        """Advance the bounded target to the final request, then let it converge."""
        if steps < 0:
            raise ValueError("settle steps cannot be negative")
        remaining = int(steps)
        catchup_steps = 0
        if (self.motion_filter is not None and self.last_request is not None
                and remaining > 0):
            requested_position, requested_quaternion = self.last_request
            while remaining >= self.simulation_steps:
                timestamp = (None if self.motion_filter.timestamp is None else
                             self.motion_filter.timestamp + self.motion_filter.default_dt)
                position, quaternion, _ = self.motion_filter.step(
                    requested_position, requested_quaternion, timestamp,
                    event_time_estimate_ms=None, gripper_state=self.gripper)
                self._command_arm(position, quaternion)
                self._command_gripper()
                self._step()
                catchup_steps += 1
                remaining -= self.simulation_steps
                if (np.linalg.norm(position - requested_position) < 1e-3
                        and quaternion_angle_degrees(
                            quaternion, requested_quaternion) < 1.):
                    break
        self._command_gripper()
        self._step(remaining)
        return {"simulation_steps": int(steps), "gripper_state": self.gripper,
                "terminal_catchup_updates": catchup_steps,
                "actual_finger_joint_positions_m": self._finger_positions()}

    def _holding_fallback(self, holding_probability):
        """Return a persistent state transition inferred from the holding head."""
        candidate = None
        if holding_probability is not None and np.isfinite(holding_probability):
            if (self.gripper == "open"
                    and holding_probability >= self.holding_thresholds["close"]):
                candidate = "closed"
            elif (self.gripper == "closed"
                  and holding_probability <= self.holding_thresholds["open"]):
                candidate = "open"
        if candidate is None:
            self._holding_candidate = None
            self._holding_candidate_frames = 0
            return None
        if candidate == self._holding_candidate:
            self._holding_candidate_frames += 1
        else:
            self._holding_candidate = candidate
            self._holding_candidate_frames = 1
        if self._holding_candidate_frames < self.holding_persistence_frames:
            return None
        self._holding_candidate = None
        self._holding_candidate_frames = 0
        return candidate

    def _command_arm(self, target_position, target_quaternion):
        solution = self.p.calculateInverseKinematics(
            self.robot, self.ee_link, targetPosition=target_position.tolist(),
            targetOrientation=self._xyzw(target_quaternion),
            lowerLimits=self.ik_lower.tolist(), upperLimits=self.ik_upper.tolist(),
            jointRanges=self.ik_ranges.tolist(), restPoses=self.ik_rest.tolist(),
            jointDamping=[.1] * len(self.ik_rest), maxNumIterations=100,
            residualThreshold=1e-5, physicsClientId=self.client)
        solution = np.asarray(solution)
        target_joints = np.clip(solution[self.arm_solution_indices],
                                self.lower, self.upper)
        self.last_target = (target_position.copy(), target_quaternion.copy(),
                            target_joints.copy())
        self.p.setJointMotorControlArray(
            self.robot, self.arm_joints, self.p.POSITION_CONTROL,
            targetPositions=target_joints.tolist(), forces=[87.] * 7,
            physicsClientId=self.client)
        return target_joints

    def attach(self, prediction):
        command = "hold"
        command_source = None
        trigger_probability = prediction.get("trigger_probability", {})

        def score(name):
            values = [prediction.get(f"{name}_probability"),
                      trigger_probability.get(name)]
            finite = [float(value) for value in values
                      if value is not None and np.isfinite(value)]
            return max(finite, default=None)

        grasp_score, release_score = score("grasp"), score("release")
        holding_probability = prediction.get("holding_probability")
        # Latched state machine: after closing, grasp probability cannot reopen
        # the fingers. Only a confident release can do that, and vice versa.
        if (self.gripper == "closed" and release_score is not None
                and release_score >= self.event_thresholds["release"]):
            self.gripper, command = "open", "open"
            command_source = "release_probability"
            self._update_status("OPEN", [0., .7, 0.])
        elif (self.gripper == "open" and grasp_score is not None
              and grasp_score >= self.event_thresholds["grasp"]):
            self.gripper, command = "closed", "close"
            command_source = "grasp_probability"
            self._update_status("CLOSED", [.9, .05, .05])
        else:
            holding_state = self._holding_fallback(holding_probability)
            if holding_state == "closed":
                self.gripper, command = "closed", "close"
                command_source = "holding_probability_rise"
                self._update_status("CLOSED", [.9, .05, .05])
            elif holding_state == "open":
                self.gripper, command = "open", "open"
                command_source = "holding_probability_fall"
                self._update_status("OPEN", [0., .7, 0.])
        prediction["gripper"] = {"state": self.gripper, "command": command,
                                  "command_source": command_source,
                                  "grasp_control_probability": grasp_score,
                                  "release_control_probability": release_score,
                                  "holding_control_probability": holding_probability,
                                  "finger_joint_position_m": self.widths[self.gripper]}
        # Grasp/release must still execute when the coincident wearable pose
        # frame is invalid. Arm motors retain their last valid position target.
        self._command_gripper()
        if not prediction["valid"] or prediction["position_m"] is None:
            self._step()
            prediction["gripper"]["actual_finger_joint_positions_m"] = (
                self._finger_positions())
            held_position = (None if self.last_target is None
                             else self.last_target[0].tolist())
            if command in {"close", "open"} and held_position is not None:
                color = [.9, 0., .9] if command == "close" else [.9, .1, .1]
                self._marker(held_position, color,
                             "GRASP" if command == "close" else "RELEASE")
            prediction["franka"] = {
                "pose_command": "held_last_valid" if self.last_target is not None
                                else "rest_pose",
                "held_target_position_m": held_position,
                "reason": "invalid wearable pose frame"}
            return prediction
        position = np.array([prediction["position_m"][axis] for axis in "xyz"])
        quaternion = prediction["orientation_quaternion_wxyz"]
        requested_position, requested_quaternion, mode = self.mapper.map(position, quaternion)
        self.last_request = (requested_position.copy(), requested_quaternion.copy())
        target_position, target_quaternion = requested_position, requested_quaternion
        control = None
        if self.motion_filter is not None:
            target_position, target_quaternion, control = self.motion_filter.step(
                requested_position, requested_quaternion,
                prediction.get("time_s"), prediction.get("position_uncertainty_cm"),
                prediction.get("orientation_uncertainty_deg"),
                prediction.get("event_time_estimate_ms"), self.gripper)
        self._draw_reference()
        target_joints = self._command_arm(target_position, target_quaternion)
        self._step()
        prediction["gripper"]["actual_finger_joint_positions_m"] = (
            self._finger_positions())
        actual_joints = [self.p.getJointState(self.robot, joint,
            physicsClientId=self.client)[0] for joint in self.arm_joints]
        state = self.p.getLinkState(self.robot, self.ee_link,
            computeForwardKinematics=True, physicsClientId=self.client)
        actual_position = np.asarray(state[4], dtype=float)
        actual_quaternion = self._wxyz(state[5])
        if self.previous_model_position is None:
            self._marker(requested_position, [0., .7, .7], "MODEL START")
        else:
            self._debug_line(self.previous_model_position, requested_position,
                             [0., .8, .9], 4.)
        if self.previous_actual_position is not None:
            self._debug_line(self.previous_actual_position, actual_position,
                             [1., .45, 0.], 4.)
        self.previous_model_position = requested_position.copy()
        self.previous_actual_position = actual_position.copy()
        if command == "close":
            self._marker(target_position, [.9, 0., .9], "GRASP")
        elif command == "open":
            self._marker(target_position, [.9, .1, .1], "RELEASE")
        prediction["franka"] = {
            "mapping": mode, "requested_position_m": requested_position.tolist(),
            "requested_orientation_wxyz": requested_quaternion.tolist(),
            "target_position_m": target_position.tolist(),
            "target_orientation_wxyz": target_quaternion.tolist(),
            "target_joint_angles_deg": np.degrees(target_joints).tolist(),
            "actual_joint_angles_deg": np.degrees(actual_joints).tolist(),
            "actual_ee_position_m": actual_position.tolist(),
            "actual_ee_orientation_wxyz": actual_quaternion.tolist(),
            "position_tracking_error_cm": float(100 * np.linalg.norm(
                actual_position - target_position)),
            "orientation_tracking_error_deg": quaternion_angle_degrees(
                actual_quaternion, target_quaternion),
            "confidence_aware_control": control}
        return prediction

    def close(self):
        if getattr(self, "client", -1) >= 0:
            self.p.disconnect(physicsClientId=self.client)
            self.client = -1
