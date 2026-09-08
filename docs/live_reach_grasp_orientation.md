# Live EMG+IMU reach/grasp/orientation inference

Use the fused checkpoint produced by the orientation experiment:

```bash
python scripts/live_reach_grasp_orientation.py \
  --checkpoint runs/reach_grasp_orientation_hybrid_seed42/emg_imu_best.pt \
  --device cuda
```

The default mode reads newline-delimited JSON from stdin. Print the exact
protocol and channel order with:

```bash
python scripts/live_reach_grasp_orientation.py \
  --checkpoint runs/reach_grasp_orientation_hybrid_seed42/emg_imu_best.pt \
  --print-protocol
```

Each raw sample contains only four EMG and 24 IMU measurements. VIVE is not an
accepted input. The process emits causal predictions every 40 ms after a 200 ms
warmup: holding/grasp/release probabilities and triggers, XYZ in metres,
quaternion `(w,x,y,z)`, and ZYX yaw/pitch/roll in degrees. Send `start` between
trials to clear filter, transformer, and event-decoder history.

On a machine using the same `EMGCollector.py` Delsys interface as data
collection, the script can connect directly:

```bash
python scripts/live_reach_grasp_orientation.py \
  --checkpoint runs/reach_grasp_orientation_hybrid_seed42/emg_imu_best.pt \
  --device cuda \
  --delsys-sdk-path /path/to/Example-Applications/Python
```

The live scan must expose the exact trained names `EMG 1_S0/S4/S8/S12` and the
ACC/GYRO XYZ channels for those sensors. It fails loudly on missing or duplicate
channels rather than silently changing sensor order.

The online path reproduces the training transformations causally: raw EMG
band-pass, 20/50 ms trailing RMS, IMU trailing median and low-pass, 100 Hz
previous-sample resampling, checkpoint normalization, and validity masks.
Predictions are fixed-weight inference; no VIVE, annotation, calibration target,
or online parameter update is used.
