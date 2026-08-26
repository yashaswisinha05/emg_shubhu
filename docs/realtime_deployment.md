# Real-time continual touch inference

The deployment path performs **continual inference with fixed weights**. It does
not update the neural-network parameters while a person is moving. A prediction
is emitted every 200 ms as new causal samples arrive, followed by a final
prediction at touch.

## Saved mix7 models

| Mode | Checkpoint | Required raw channels |
|---|---|---|
| EMG only | `runs/continual_attention/mix7/fold-0/grid_emg/best.pt` | 4 EMG |
| EMG + IMU | `runs/continual_attention/mix7/fold-0/grid_fusion/best.pt` | 4 EMG + 24 IMU |

Both use:

`artifacts/continual_attention/mix7/fold-0/scaler.npz`

These checkpoints were trained for the **mix7 sensor placement/configuration**.
Do not deploy them with a different placement and interpret the output as a
validated result.

## Hardware requirements

- Use a monotonic timestamp in seconds.
- Preserve the channel order printed by `--print-protocol`.
- Begin buffering at least 300 ms before movement starts.
- During the first 300 ms, the user should remain at rest. Fusion uses this
  interval for per-trajectory IMU bias, gravity and orientation calibration.
- Send the movement-start event at the same semantic point used during data
  collection.
- Keep acquisition running through the touch event.
- Supply the real screen/canvas dimensions to the command.

The program causally resamples irregular hardware timestamps to 148.148 Hz. It
then applies the same trailing median filter, training-fold scaler and calibrated
IMU feature construction used during training.

## Channel order

EMG:

```text
EMG RMS 1_S0, EMG RMS 1_S4, EMG RMS 1_S8, EMG RMS 1_S12
```

IMU contains ACC X/Y/Z followed by GYRO X/Y/Z for each sensor:

```text
S0, S4, S8, S12
```

Run this to print the complete machine-readable protocol:

```bash
python scripts/realtime_predict.py --print-protocol
```

## Start EMG + IMU inference

```bash
conda activate smss
cd /Users/yashaswi/phd_iisc/emg_shubhu/emg_touch

python scripts/realtime_predict.py \
  --kind grid_fusion \
  --device mps \
  --screen-width 1536 \
  --screen-height 774
```

## Start EMG-only inference

```bash
python scripts/realtime_predict.py \
  --kind grid_emg \
  --device mps \
  --screen-width 1536 \
  --screen-height 774
```

The process reads newline-delimited JSON from standard input. This allows a BLE,
serial, LSL or socket collector to remain independent of the neural network.

## Live message sequence

Reset for a new trajectory:

```json
{"event":"start"}
```

Send resting samples first:

```json
{"event":"sample","time_s":100.000,"emg":[0.1,0.2,0.1,0.2],"imu":[0.0,0.0,9.8,0.0,0.0,0.0,0.0,0.0,9.8,0.0,0.0,0.0,0.0,0.0,9.8,0.0,0.0,0.0,0.0,0.0,9.8,0.0,0.0,0.0]}
```

After at least 300 ms of rest, mark movement onset:

```json
{"event":"movement_start","time_s":100.300}
```

Continue sending sample events. The program emits predictions at movement times
0.0 s, 0.2 s, 0.4 s and so forth.

At touch, first send the latest sensor sample and then:

```json
{"event":"touch","time_s":101.250}
```

For EMG-only mode, omit the `imu` field. Optional `emg_mask` and `imu_mask`
boolean arrays can mark unavailable channels. Without explicit masks, finite
values are treated as valid and NaN/Infinity as unavailable.

## Output

Each prediction is one JSON line containing:

- normalized `x_norm`, `y_norm`;
- screen coordinates `x_px`, `y_px`;
- movement time and number of resampled samples;
- inference latency;
- auxiliary grid cell/confidence;
- EMG reliability and 300/500 ms lookback weights when available.

The auxiliary grid confidence is a diagnostic from the training head. It is not
a calibrated probability that the final X/Y prediction is correct.

## Replay an existing recording

CSV replay is the safest integration test before connecting live hardware:

```bash
python scripts/realtime_predict.py \
  --kind grid_fusion \
  --device mps \
  --input-csv /absolute/path/to/trial_001.csv \
  --movement-start-s 3.1205 \
  --touch-s 3.3795247
```

The CSV must contain the original timing and signal-column names. Replay emits
the same 200 ms prediction sequence as live mode.

## Production boundary

The current script provides model inference and a stable stream protocol. The
hardware-specific collector must still:

1. Read EMG and IMU devices.
2. Synchronize their timestamps.
3. Put values in the documented channel order.
4. Emit `movement_start` and `touch` events.
5. Pipe JSON lines into `realtime_predict.py`.

Online weight adaptation should not occur during a movement. If a true touch
coordinate becomes available after the interaction, save the complete labelled
trajectory and retrain or calibrate offline with leakage-safe participant splits.

