# EMG-only grasp-onset experiment

This experiment deliberately removes release, holding, position and orientation
objectives. Its single question is whether causal forearm EMG can detect grasp
onset better than protocol timing and EMG-free controls.

The model consumes the four sensors' causal 20/50 ms RMS features at 200 Hz.
It adds causal first differences and uses a multi-scale dilated temporal CNN.
Manual timestamp uncertainty is represented by a Gaussian target truncated to
the requested uncertainty interval. The loss is a soft-label focal BCE.

Run:

```bash
bash scripts/train_emg_grasp_onset.sh
```

The decoder threshold and persistence are selected on validation trials only.
`results.json` reports held-out test performance at 100, 150, 200, 300, 500 and
1500 ms, including zero-EMG, trial-shuffled-EMG, and schedule-only controls.
The 500 ms result is the configured primary result. A useful EMG claim requires the real
EMG result to beat all three controls; the 1500 ms result is only a sanity check.

To visualize an unseen trial on Franka, use `visualize_emg_grasp_franka.py`.
EMG alone controls grasp closure. The arm can either stay fixed (strict
EMG-only actuation) or follow the recorded VIVE pose for demonstration; VIVE
is never passed to the grasp detector.

For a completely model-driven Franka replay, use
`visualize_model_output_franka.py`: the full EMG+IMU model supplies XYZ,
orientation, and release; the dedicated EMG model supplies grasp onset. VIVE
is drawn only as a withheld black comparison trajectory.
