import numpy as np

from emg_touch.physics.franka_pybullet import (LiveFrankaPyBulletController,
                                                PoseMapper)
from emg_touch.physics.rotation_6d import quaternion_to_matrix_numpy


class FakeBullet:
    GUI, DIRECT, POSITION_CONTROL = 1, 2, 3

    def __init__(self):
        self.states = {}
        self.ik_call = None
        self.motor_calls = []
        self.link_pose = None
        arm = [(f"panda_joint{i}", f"panda_link{i}", -2.9, 2.9)
               for i in range(1, 8)]
        fixed = [("panda_joint8", "panda_hand", 0., -1.),
                 ("panda_hand_joint", "panda_leftfinger", 0., -1.)]
        fingers = [("panda_finger_joint1", "panda_leftfinger", 0., .04),
                   ("panda_finger_joint2", "panda_rightfinger", 0., .04)]
        target = [("panda_grasptarget_joint", "panda_grasptarget", 0., -1.)]
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
        name, link, lower, upper = self.joints[index]
        info = [None] * 13
        info[1], info[8], info[9], info[12] = name.encode(), lower, upper, link.encode()
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
        pass

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
               grasp=False, release=False):
    return {
        "valid": True,
        "position_m": dict(zip("xyz", position)),
        "orientation_quaternion_wxyz": list(quaternion),
        "triggered": {"grasp": grasp, "release": release},
        "trigger_time_s": {"grasp": 1. if grasp else None,
                           "release": 2. if release else None},
    }


def test_model_pose_drives_franka_ik_and_events_drive_real_finger_joints():
    fake = FakeBullet()
    mapper = PoseMapper((0., 0., 0.), (1., 0., 0., 0.))
    controller = LiveFrankaPyBulletController(
        gui=False, mapper=mapper, simulation_steps=2, bullet=fake,
        data_path="fake-data")

    closed = controller.attach(prediction(grasp=True))
    assert fake.ik_call["targetPosition"] == [.45, 0., .5]
    assert fake.ik_call["targetOrientation"] == [0., 0., 0., 1.]
    assert len(closed["franka"]["target_joint_angles_deg"]) == 7
    assert closed["gripper"]["state"] == "closed"
    assert fake.motor_calls[-1][1]["targetPositions"] == [0., 0.]
    assert closed["franka"]["position_tracking_error_cm"] == 0.
    assert closed["franka"]["orientation_tracking_error_deg"] == 0.

    opened = controller.attach(prediction(release=True))
    assert opened["gripper"]["state"] == "open"
    assert fake.motor_calls[-1][1]["targetPositions"] == [.04, .04]
    controller.close()
    assert fake.disconnected and controller.client == -1
