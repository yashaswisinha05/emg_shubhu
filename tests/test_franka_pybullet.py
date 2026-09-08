import numpy as np

from emg_touch.physics.franka_pybullet import (LiveFrankaPyBulletController,
                                                PoseMapper)
from emg_touch.physics.confidence_se3 import ConfidenceAwareSE3Controller
from emg_touch.physics.rotation_6d import quaternion_to_matrix_numpy


class FakeBullet:
    GUI, DIRECT, POSITION_CONTROL = 1, 2, 3
    JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_FIXED = 0, 1, 4

    def __init__(self):
        self.states = {}
        self.ik_call = None
        self.motor_calls = []
        self.link_pose = None
        self.steps = 0
        self.debug_lines = []
        self.debug_text = []
        arm = [(f"panda_joint{i}", f"panda_link{i}", -2.9, 2.9,
                self.JOINT_REVOLUTE)
               for i in range(1, 8)]
        fixed = [("panda_joint8", "panda_hand", 0., -1., self.JOINT_FIXED),
                 ("panda_hand_joint", "panda_leftfinger", 0., -1., self.JOINT_FIXED)]
        fingers = [("panda_finger_joint1", "panda_leftfinger", 0., .04,
                    self.JOINT_PRISMATIC),
                   ("panda_finger_joint2", "panda_rightfinger", 0., .04,
                    self.JOINT_PRISMATIC)]
        target = [("panda_grasptarget_joint", "panda_grasptarget", 0., -1.,
                   self.JOINT_FIXED)]
        self.joints = arm + fixed + fingers + target

    def connect(self, mode):
        return 7

    def setAdditionalSearchPath(self, path, **kwargs):
        self.data_path = path

    def setGravity(self, *args, **kwargs):
        pass

    def setTimeStep(self, *args, **kwargs):
        pass

    def loadURDF(self, *args, **kwargs):
        return 3

    def getNumJoints(self, robot, **kwargs):
        return len(self.joints)

    def getJointInfo(self, robot, index, **kwargs):
        name, link, lower, upper, joint_type = self.joints[index]
        info = [None] * 13
        info[1], info[2], info[8], info[9], info[12] = (
            name.encode(), joint_type, lower, upper, link.encode())
        return tuple(info)

    def resetJointState(self, robot, joint, value, **kwargs):
        self.states[joint] = value

    def calculateInverseKinematics(self, robot, link, **kwargs):
        self.ik_call = kwargs
        self.link_pose = (kwargs["targetPosition"], kwargs["targetOrientation"])
        return kwargs["restPoses"]

    def setJointMotorControlArray(self, robot, joints, mode, **kwargs):
        self.motor_calls.append((list(joints), kwargs))
        for joint, value in zip(joints, kwargs["targetPositions"]):
            self.states[joint] = value

    def stepSimulation(self, **kwargs):
        self.steps += 1

    def getJointState(self, robot, joint, **kwargs):
        return (self.states[joint], 0., (), 0.)

    def getLinkState(self, robot, link, **kwargs):
        position, quaternion = self.link_pose
        return (position, quaternion, None, None, position, quaternion)

    def disconnect(self, **kwargs):
        self.disconnected = True

    def removeAllUserDebugItems(self, **kwargs):
        self.debug_lines.clear()
        self.debug_text.clear()

    def addUserDebugLine(self, first, second, color, **kwargs):
        self.debug_lines.append((first, second, color, kwargs))
        return len(self.debug_lines)

    def addUserDebugText(self, text, position, **kwargs):
        self.debug_text.append((text, position, kwargs))
        return len(self.debug_text)


def test_explicit_pose_mapping_applies_one_rigid_transform():
    # 90 degrees around z, followed by a base-frame translation.
    root_half = np.sqrt(.5)
    mapper = PoseMapper((1., 2., 3.), (root_half, 0., 0., root_half))
    position, quaternion, mode = mapper.map((1., 0., 0.), (1., 0., 0., 0.))
    np.testing.assert_allclose(position, (1., 3., 3.), atol=1e-7)
    np.testing.assert_allclose(
        quaternion_to_matrix_numpy(quaternion),
        quaternion_to_matrix_numpy((root_half, 0., 0., root_half)), atol=1e-7)
    assert mode == "calibrated VIVE-to-Panda transform"


def test_synthetic_mapper_anchors_first_pose_and_preserves_relative_motion():
    root_half = np.sqrt(.5)
    mapper = PoseMapper(home_position=(.4, -.1, .5),
                        home_quaternion_wxyz=(root_half, 0., 0., root_half))
    first_position, first_quaternion, mode = mapper.map(
        (4., 5., 6.), (root_half, root_half, 0., 0.))
    np.testing.assert_allclose(first_position, (.4, -.1, .5), atol=1e-7)
    np.testing.assert_allclose(
        quaternion_to_matrix_numpy(first_quaternion),
        quaternion_to_matrix_numpy((root_half, 0., 0., root_half)), atol=1e-7)
    # Moving one metre along the first pose's local y becomes one metre along
    # the configured Panda home pose's local y.
    moved_position, _, _ = mapper.map((4., 5., 7.),
                                      (root_half, root_half, 0., 0.))
    np.testing.assert_allclose(moved_position, (-.6, -.1, .5), atol=1e-7)
    assert mode == "synthetic first-pose anchor"


def test_post_mapping_z_rotation_is_anticlockwise_about_home_anchor():
    mapper = PoseMapper(home_position=(.45, 0., .5),
                        home_quaternion_wxyz=(1., 0., 0., 0.),
                        trajectory_z_rotation_deg=90.)
    first, _, _ = mapper.map((2., 3., 4.), (1., 0., 0., 0.))
    moved, quaternion, mode = mapper.map((2.1, 3., 4.), (1., 0., 0., 0.))
    np.testing.assert_allclose(first, (.45, 0., .5), atol=1e-7)
    # Positive model x becomes positive robot y after active +90 degree Rz.
    np.testing.assert_allclose(moved, (.45, .1, .5), atol=1e-7)
    expected = quaternion_to_matrix_numpy((np.sqrt(.5), 0., 0., np.sqrt(.5)))
    np.testing.assert_allclose(quaternion_to_matrix_numpy(quaternion), expected,
                               atol=1e-7)
    assert "z rotation +90 deg" in mode


def prediction(position=(.45, 0., .5), quaternion=(1., 0., 0., 0.),
               grasp=False, release=False, holding=0., valid=True):
    return {
        "valid": valid,
        "position_m": dict(zip("xyz", position)) if valid else None,
        "orientation_quaternion_wxyz": list(quaternion) if valid else None,
        "holding_probability": holding,
        "grasp_probability": .95 if grasp else .05,
        "release_probability": .95 if release else .05,
        "trigger_probability": {"grasp": .95 if grasp else None,
                                "release": .95 if release else None},
        "triggered": {"grasp": grasp, "release": release},
        "trigger_time_s": {"grasp": 1. if grasp else None,
                           "release": 2. if release else None},
    }


def last_finger_targets(fake, controller):
    return next(call[1]["targetPositions"] for call in reversed(fake.motor_calls)
                if call[0] == controller.finger_joints)


def test_model_pose_drives_franka_ik_and_events_drive_real_finger_joints():
    fake = FakeBullet()
    mapper = PoseMapper((0., 0., 0.), (1., 0., 0., 0.))
    controller = LiveFrankaPyBulletController(
        gui=False, mapper=mapper, simulation_steps=2, bullet=fake,
        data_path="fake-data")

    closed = controller.attach(prediction(grasp=True))
    assert fake.ik_call["targetPosition"] == [.45, 0., .5]
    assert fake.ik_call["targetOrientation"] == [0., 0., 0., 1.]
    assert len(fake.ik_call["jointDamping"]) == 9
    assert len(closed["franka"]["target_joint_angles_deg"]) == 7
    assert closed["gripper"]["state"] == "closed"
    assert last_finger_targets(fake, controller) == [0., 0.]
    assert closed["franka"]["position_tracking_error_cm"] == 0.
    assert closed["franka"]["orientation_tracking_error_deg"] == 0.

    opened = controller.attach(prediction(release=True))
    assert opened["gripper"]["state"] == "open"
    assert last_finger_targets(fake, controller) == [.04, .04]
    controller.close()
    assert fake.disconnected and controller.client == -1


def test_invalid_release_still_opens_fingers_and_keeps_stepping_arm():
    fake = FakeBullet()
    controller = LiveFrankaPyBulletController(
        gui=False, mapper=PoseMapper((0., 0., 0.), (1., 0., 0., 0.)),
        simulation_steps=2, bullet=fake)
    controller.attach(prediction(grasp=True))
    steps_before = fake.steps
    released = controller.attach(prediction(release=True, valid=False))
    assert released["gripper"]["state"] == "open"
    assert released["gripper"]["command_source"] == "release_probability"
    assert last_finger_targets(fake, controller) == [.04, .04]
    assert fake.steps == steps_before + 2
    assert released["franka"]["pose_command"] == "held_last_valid"


def test_probability_state_machine_latches_until_opposite_event():
    fake = FakeBullet()
    controller = LiveFrankaPyBulletController(gui=False, bullet=fake)
    closed = controller.attach(prediction(grasp=True))
    assert closed["gripper"]["state"] == "closed"
    # Low grasp probability cannot open the latched gripper.
    held = controller.attach(prediction())
    assert held["gripper"]["state"] == "closed"
    opened = controller.attach(prediction(release=True, valid=False))
    assert opened["gripper"]["state"] == "open"


def test_holding_state_is_persistent_fallback_for_grasp_and_release():
    fake = FakeBullet()
    controller = LiveFrankaPyBulletController(
        gui=False, bullet=fake, holding_persistence_frames=2)
    first_high = controller.attach(prediction(holding=.96))
    assert first_high["gripper"]["state"] == "open"
    closed = controller.attach(prediction(holding=.95))
    assert closed["gripper"]["state"] == "closed"
    assert closed["gripper"]["command_source"] == "holding_probability_rise"
    first_low = controller.attach(prediction(holding=.10))
    assert first_low["gripper"]["state"] == "closed"
    opened = controller.attach(prediction(holding=.05))
    assert opened["gripper"]["state"] == "open"
    assert opened["gripper"]["command_source"] == "holding_probability_fall"


def test_settle_advances_rate_limited_target_to_last_model_request():
    fake = FakeBullet()
    motion = ConfidenceAwareSE3Controller(
        workspace_lower=(0., -1., 0.), workspace_upper=(1., 1., 1.),
        max_velocity_mps=.2, max_acceleration_mps2=100., default_dt_s=.1)
    controller = LiveFrankaPyBulletController(
        gui=False, bullet=fake, simulation_steps=2, motion_filter=motion,
        mapper=PoseMapper((0., 0., 0.), (1., 0., 0., 0.)))
    controller.attach(prediction(position=(.2, 0., .5)))
    controller.attach(prediction(position=(.5, 0., .5)))
    before = controller.last_target[0].copy()
    report = controller.settle(steps=40)
    after = controller.last_target[0]
    assert np.linalg.norm(after - np.array([.5, 0., .5])) < np.linalg.norm(
        before - np.array([.5, 0., .5]))
    assert report["terminal_catchup_updates"] > 0


def test_pybullet_debug_overlay_contains_reference_model_actual_and_markers():
    fake = FakeBullet()
    controller = LiveFrankaPyBulletController(gui=False, bullet=fake)
    controller.set_reference_trajectory([[.45, 0., .5], [.44, .01, .49]])
    controller.attach(prediction())
    controller.attach(prediction((.44, .01, .49), grasp=True))
    colors = [line[2] for line in fake.debug_lines]
    assert [.05, .05, .05] in colors  # withheld VIVE
    assert [0., .8, .9] in colors     # model request
    assert [1., .45, 0.] in colors    # actual Franka end effector
    labels = [item[0] for item in fake.debug_text]
    assert any("GRIPPER: CLOSED" in label for label in labels)
    assert "GRASP" in labels
