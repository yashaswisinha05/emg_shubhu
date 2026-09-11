# MAHG-EMG transfer benchmark

This benchmark tests whether the EMG representation learned by the causal GRU
transfers to the five-class MAHG-EMG task. It does **not** treat windows from the
same person as independent train and test examples.

Download and extract the dataset from the [official IEEE DataPort page](https://ieee-dataport.org/documents/multi-angle-hand-gesture-emg-dataset-mahg-emg).
The directory should contain `EMGData1.csv` through `EMGData100.csv`.

Run:

```bash
python scripts/finetune_mahg_emg.py \
  --root /path/to/extracted/MAHG-EMG \
  --checkpoint runs/gripper_neuromuscular_future_fused/emg_imu_best.pt \
  --protocols scratch linear_probe finetune \
  --device cuda \
  --epochs 50 \
  --output-dir runs/mahg_emg_transfer_seed42
```

The default split is subjects 1–8 for training, subject 9 for validation, and
subject 10 for the untouched test. The documented ten-files-per-subject layout
is inferred from the numeric `EMGData<N>.csv` names. Do not randomly split
overlapping windows.

The three protocols isolate distinct claims:

- `scratch`: identical EMG encoder topology initialized randomly.
- `linear_probe`: pretrained causal EMG encoder frozen; only a new five-class head trains.
- `finetune`: pretrained encoder plus a new head; only its final temporal block and head train.

Both the causal-GRU architecture-study checkpoint and the selected
`gripper_neuromuscular_future_v1` checkpoint are supported. For the latter,
transfer uses only its causal EMG patch encoder: MAHG has no matching IMU
channels, and fabricating IMU inputs would invalidate the experiment.

Preprocessing is causal and matches this project's EMG representation: a
20–450 Hz band-pass, trailing 20 ms and 50 ms RMS features, bounded forward fill,
and 100 Hz output. Normalization statistics come only from the MAHG training
subjects. Results and confusion matrices are saved to `results.json`; each
protocol also writes its own `*_best.pt` checkpoint.

Evaluate the untouched subject recorded in the checkpoint:

```bash
python scripts/test_mahg_emg.py \
  --root /path/to/extracted/MAHG-EMG \
  --checkpoint runs/mahg_neuromuscular_transfer/finetune_best.pt \
  --device cuda \
  --output runs/mahg_neuromuscular_transfer/test_results.json
```

Use `--subjects 8 9 10` only when deliberately evaluating those subjects. The
default is safer for paper reporting because it reads the untouched test-subject
split saved by training.

For a defensible transfer claim, report test macro-F1 and balanced accuracy for
all three protocols over several seeds. Transfer is supported when the frozen
linear probe or partial fine-tune consistently beats the same GRU trained from
scratch—not merely when one fine-tuned run achieves high training accuracy.
