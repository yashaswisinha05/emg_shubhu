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
`results.json` reports held-out test performance at 100, 150, 200, 300 and
1500 ms, including zero-EMG, trial-shuffled-EMG, and schedule-only controls.
The 200 ms result is the primary result. A useful EMG claim requires the real
EMG result to beat all three controls; the 1500 ms result is only a sanity check.
