import numpy as np

from emg_touch.physics.franka_pybullet import (LiveFrankaPyBulletController,
                                                PoseMapper)
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


def prediction(position=(.45, 0., .5), quaternion=(1., 0., 0., 0.),
               grasp=False, release=False, holding=0., valid=True):
    return {
        "valid": valid,
        "position_m": dict(zip("xyz", position)) if valid else None,
        "orientation_quaternion_wxyz": list(quaternion) if valid else None,
        "holding_probability": holding,
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
    assert released["gripper"]["command_source"] == "release_event"
    assert last_finger_targets(fake, controller) == [.04, .04]
    assert fake.steps == steps_before + 2
    assert released["franka"]["pose_command"] == "held_last_valid"


def test_holding_hysteresis_recovers_missed_event_pulses():
    fake = FakeBullet()
    controller = LiveFrankaPyBulletController(gui=False, bullet=fake)
    closed = controller.attach(prediction(holding=.9))
    assert closed["gripper"]["state"] == "closed"
    assert closed["gripper"]["command_source"] == "holding_fallback"
    opened = controller.attach(prediction(holding=.1, valid=False))
    assert opened["gripper"]["state"] == "open"
    assert opened["gripper"]["command_source"] == "holding_fallback"
