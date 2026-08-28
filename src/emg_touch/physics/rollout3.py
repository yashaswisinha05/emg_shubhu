"""Differentiable forward rollout of the 3-DOF arm, driven by what each
modality can actually observe: IMU prescribes the shoulder, EMG drives the
elbow through predicted joint torque.

The four EMG/IMU units sit on AD, LD, BB and TB - anterior deltoid, lateral
deltoid, biceps and triceps. Every one of them is proximal to the elbow, and
their integrated-orientation traces correlate at |r| = 0.86-0.999 on real
trials, i.e. all four are measuring essentially one rigid segment: the
shoulder/upper-arm complex. That has a hard consequence for this model.
Shoulder rotation is well observed. Elbow flexion has no sensor anywhere -
it is only inferable from muscle activity, and BB/TB are exactly the elbow
flexor and extensor.

Earlier versions ignored this and guessed all six state variables (three
angles, three velocities) from a pooled EMG context vector, then integrated
~200 steps against a single endpoint loss. That is very weak supervision for
a 6-D latent trajectory, and it showed: physics_blend stayed pinned at its
~0.018 initialisation across every run and every loss weight tried, and a
direct read of raw_blend.grad found no consistent incentive to raise it -
physics_prediction simply was not accurate enough to be worth blending in.
Per-participant anthropometry calibration was tried against the same
symptom and came back negative too (all six participants converged on
near-identical corrections).

So the shoulder is now prescribed from its measurement instead of imagined,
and the physics is left to do the one job it uniquely can: infer the
unobservable elbow from muscle activity, under the full rigid-body coupling.

Prescribing the shoulder also removes this model's worst numerical hazard.
Solving the unconstrained M(q) qdd = tau - C qd - g needs the mass matrix
inverted, and its smaller eigenvalue drops to 0.011 over the joint space -
near-singular enough that shoulder torque could dominate, and even reverse,
elbow acceleration through the coupling term (the root cause of the
elbow-pinning bug this model inherited). With the shoulder prescribed, only
the elbow row is solved:

    M[2,2] qdd3 = tau3 - (C qd)[2] - g[2] - b3 qd3
                  - M[2,0] qdd1 - M[2,1] qdd2

and M[2,2] - the forearm's own inertia about the elbow - is a positive
constant, measured at 0.088339 with exactly zero variation across the whole
joint space. A scalar divide by a constant: no inversion, no eigenvalue
floor, unconditionally well posed. The measured shoulder still couples into
the elbow through M[2,0], M[2,1] and the Coriolis term, so the rigid-body
physics is intact, not bypassed.

There is still no ground truth for torque or joint angle anywhere in this
dataset - only the final touch location - so the elbow remains supervised
only indirectly, through the endpoint.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..data.grid_trajectory import grid_imu_orientation_indices
from .arm3 import EndpointToScreen3, ThreeDofArm
from .participant_calibration import ParticipantCalibration

# Bound on shoulder angular displacement from the trial's starting posture.
# The readout below maps robust-scaled orientation channels, whose units are
# not radians, so the range is imposed rather than inherited - roughly +-86
# degrees of travel per axis, comfortably past any reach in this task.
SHOULDER_DISPLACEMENT_LIMIT = 1.5
# Measured shoulder acceleration is a second difference of an integrated
# gyro signal, so it is the noisiest quantity in the rollout. It enters only
# through the M[2,0]/M[2,1] coupling term; clamping keeps a spike from
# swamping the elbow's own torque without touching the well-conditioned
# angle and velocity terms.
SHOULDER_ACCELERATION_LIMIT = 100.0


class TorqueHead(nn.Module):
    """EMG + joint state + pooled context -> joint torque, per decimated step."""

    def __init__(self, d_model: int, context_dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.context_project = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, context_dim), nn.GELU()
        )
        input_dim = 4 + 3 + 3 + context_dim  # emg, angles, velocity, context
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        # Zero-initialised so the rollout starts torque-free (gravity/damping
        # only) rather than at some arbitrary scale.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        emg_amplitude: torch.Tensor,
        angles: torch.Tensor,
        velocity: torch.Tensor,
        context_embed: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([emg_amplitude, angles, velocity, context_embed], dim=-1)
        return self.net(features)


class PhysicsBranch3(nn.Module):
    """IMU-prescribed shoulder + EMG-driven elbow, on 3-DOF rigid-body dynamics."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        data = config["data"]
        physics = config.get("physics", {})
        d_model = int(model["d_model"])
        self.sample_rate_hz = float(data["sample_rate_hz"])
        self.decimation = int(physics.get("decimation", 4))
        self.substeps = int(physics.get("substeps", 8))
        # Config-gated so the prescribed-shoulder formulation can be ablated
        # against the previous free-integration one without a code change.
        self.imu_driven_shoulder = bool(physics.get("imu_driven_shoulder", True))
        # Hand the rollout the gravity torque it already knows, so the learned
        # term only has to explain voluntary drive. Without this the head must
        # first reproduce a force it is given for free: gravity applies up to
        # ~3.4 N*m at the elbow while as little as 2 N*m of net torque drives
        # the joint into a limit (measured), so nearly the whole output range
        # would map to a saturated joint and the useful band would be a sliver
        # around exact cancellation. This is the tau = known + learned split.
        self.gravity_compensation = bool(physics.get("gravity_compensation", True))
        # Bounds the learned term to the range that actually moves this joint
        # without slamming it into a stop, same tanh-times-scale convention the
        # Hill branch uses for its residual torque.
        self.torque_scale = float(physics.get("torque_scale", 1.0))

        self.arm = ThreeDofArm()
        self.to_screen = EndpointToScreen3()
        self.torque_head = TorqueHead(d_model)
        self.calibration = ParticipantCalibration(config["paths"]["split_file"])

        orientation = grid_imu_orientation_indices(data)
        self.register_buffer(
            "orientation_indices", torch.tensor(orientation, dtype=torch.long),
            persistent=False,
        )
        # Sensor axes do not align with this model's joint convention, and the
        # channels arrive robust-scaled, so the map from measured orientation
        # to shoulder angle is learned rather than assumed. It is a 12->2
        # linear readout: small enough to stay identifiable against the
        # endpoint loss, which is what ties an angle to a screen position
        # through the arm's fixed geometry.
        self.shoulder_readout = nn.Linear(len(orientation), 2)
        nn.init.normal_(self.shoulder_readout.weight, std=0.02)
        nn.init.zeros_(self.shoulder_readout.bias)

        # orientation_rel measures displacement since the trial start, so the
        # absolute starting posture is still unobserved and has to be
        # inferred: three initial angles plus the elbow's initial velocity.
        # The shoulder's initial velocity comes from the measurement itself.
        self.initial_state = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 4)
        )

    def _measured_shoulder(
        self, imu: torch.Tensor, imu_mask: torch.Tensor, indices: list[int]
    ) -> torch.Tensor:
        """Shoulder angular displacement at each decimated step, (B, K, 2)."""
        orientation = imu.index_select(-1, self.orientation_indices)
        valid = imu_mask.index_select(-1, self.orientation_indices).to(orientation.dtype)
        displacement = torch.tanh(
            self.shoulder_readout(orientation * valid)
        ) * SHOULDER_DISPLACEMENT_LIMIT
        # Re-zero at t=0: the readout's bias and the tanh would otherwise put
        # a constant offset into what must be a displacement from the trial's
        # own starting posture, which initial_state is separately responsible
        # for.
        displacement = displacement - displacement[:, :1]
        step_index = torch.tensor(indices, device=imu.device, dtype=torch.long)
        step_index = step_index.clamp(max=displacement.size(1) - 1)
        return displacement.index_select(1, step_index)

    def forward(
        self,
        emg: torch.Tensor,
        emg_mask: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
        subject: list[str],
        imu: torch.Tensor,
        imu_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = emg.shape
        amplitude = emg[:, :, :4] * emg_mask[:, :, :4].to(emg.dtype)

        context_embed = self.torque_head.context_project(context)

        state = self.initial_state(context)
        angles = state[:, 0:3] * 0.5
        angles = angles + angles.new_tensor([0.0, 0.0, 1.4])  # elbow centred mid-range
        velocity = torch.stack(
            [torch.zeros_like(state[:, 3]), torch.zeros_like(state[:, 3]), state[:, 3] * 0.5],
            dim=-1,
        )

        dt_step = self.decimation / self.sample_rate_hz
        dt = dt_step / self.substeps
        indices = list(range(0, steps, self.decimation))
        lower = torch.tensor([-1.8, -1.8, 0.0], device=angles.device, dtype=angles.dtype)
        upper = torch.tensor([1.8, 1.8, 2.8], device=angles.device, dtype=angles.dtype)

        shoulder = shoulder_velocity = shoulder_acceleration = None
        if self.imu_driven_shoulder:
            displacement = self._measured_shoulder(imu, imu_mask, indices)
            shoulder = angles[:, None, 0:2] + displacement
            shoulder = shoulder.clamp(lower[0:2], upper[0:2])
            shoulder_velocity = torch.zeros_like(shoulder)
            shoulder_velocity[:, 1:] = (shoulder[:, 1:] - shoulder[:, :-1]) / dt_step
            shoulder_acceleration = torch.zeros_like(shoulder)
            shoulder_acceleration[:, 1:] = (
                shoulder_velocity[:, 1:] - shoulder_velocity[:, :-1]
            ) / dt_step
            shoulder_acceleration = shoulder_acceleration.clamp(
                -SHOULDER_ACCELERATION_LIMIT, SHOULDER_ACCELERATION_LIMIT
            )

        trajectory = []
        torques = []
        for position, step in enumerate(indices):
            active = (step < lengths).to(angles.dtype).unsqueeze(-1)
            if self.imu_driven_shoulder:
                # Hold the measurement while the trial is still running; once
                # past its end the prescribed shoulder freezes along with
                # everything else, so padded steps cannot advance the state.
                held = active * shoulder[:, position] + (1.0 - active) * angles[:, 0:2]
                angles = torch.cat([held, angles[:, 2:3]], dim=-1)
                velocity = torch.cat(
                    [active * shoulder_velocity[:, position], velocity[:, 2:3]], dim=-1
                )
            learned = torch.tanh(
                self.torque_head(amplitude[:, step], angles, velocity, context_embed)
            ) * self.torque_scale
            torques.append(learned)
            for _ in range(self.substeps):
                gravity = self.arm.gravity(angles)
                torque = learned + gravity if self.gravity_compensation else learned
                if self.imu_driven_shoulder:
                    mass = self.arm.mass_matrix(angles)
                    coupling = (
                        mass[..., 2, 0] * shoulder_acceleration[:, position, 0]
                        + mass[..., 2, 1] * shoulder_acceleration[:, position, 1]
                    )
                    residual = (
                        torque[..., 2]
                        - self.arm.coriolis(angles, velocity)[..., 2]
                        - gravity[..., 2]
                        - self.arm.damping[2] * velocity[..., 2]
                        - coupling
                    )
                    # M[2,2] is the forearm's own inertia about the elbow: a
                    # positive constant over the whole joint space, so this
                    # scalar solve needs no regularisation.
                    elbow_acceleration = residual / mass[..., 2, 2]
                    elbow_velocity = velocity[..., 2] + active[..., 0] * dt * elbow_acceleration
                    elbow_velocity = elbow_velocity.clamp(-25.0, 25.0)
                    unclamped = angles[..., 2] + active[..., 0] * dt * elbow_velocity
                    elbow_angle = unclamped.clamp(lower[2], upper[2])
                    # Inelastic joint stop: zero the velocity that drove into
                    # the limit, or the joint pins there permanently no matter
                    # how the torque later changes.
                    elbow_velocity = torch.where(
                        unclamped != elbow_angle,
                        torch.zeros_like(elbow_velocity),
                        elbow_velocity,
                    )
                    angles = torch.cat([angles[..., 0:2], elbow_angle.unsqueeze(-1)], dim=-1)
                    velocity = torch.cat(
                        [velocity[..., 0:2], elbow_velocity.unsqueeze(-1)], dim=-1
                    )
                else:
                    acceleration = self.arm.acceleration(angles, velocity, torque)
                    velocity = velocity + active * dt * acceleration
                    velocity = velocity.clamp(-25.0, 25.0)
                    unclamped = angles + active * dt * velocity
                    angles = unclamped.clamp(lower, upper)
                    at_limit = unclamped != angles
                    velocity = torch.where(at_limit, torch.zeros_like(velocity), velocity)
            trajectory.append(angles)

        final = angles
        endpoint = self.arm.endpoint(final)
        subject_indices = self.calibration.indices_for(subject, endpoint.device)
        participant_scale = self.calibration.scale(subject_indices)
        participant_offset = self.calibration.offset_for(subject_indices)
        endpoint = endpoint * participant_scale.unsqueeze(-1)
        physics_prediction = self.to_screen(endpoint) + participant_offset
        outputs = {
            "physics_prediction": physics_prediction,
            "physics_participant_scale": participant_scale,
            "physics_participant_offset": participant_offset,
            "physics_angles": final,
            "physics_velocity": velocity,
            # The learned term only, excluding any gravity compensation: this
            # is what grid_point_loss charges for, and charging the model for
            # holding the arm up against a force it was handed for free would
            # penalise correct behaviour.
            "physics_torque": torch.stack(torques, dim=1),
            "physics_trajectory": torch.stack(trajectory, dim=1),
        }
        if self.imu_driven_shoulder:
            outputs["physics_shoulder_measured"] = shoulder
        return outputs
