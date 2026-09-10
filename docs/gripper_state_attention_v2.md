# State-conditioned attention V2

V2 leaves the successful V1 XYZ/future design intact and targets its two test
regressions. The gripper decoder restores the causal local convolution,
sustained-context classifier, and masked-EMG auxiliary reconstruction used by
the earlier high-F1 model. It remains strictly EMG-only.

The pixel output blends a direct intent prediction with a learned mapping from
the predicted current XYZ. Pixel gradients are detached at XYZ, so screen loss
cannot corrupt metric position. The combined and geometric predictions receive
strong fourth-quarter weighting; a uniform direct-head auxiliary preserves
early intent.

```bash
python scripts/train_gripper_state_attention_v2.py \
  --root data/shubham_open data/shubham_close \
         data/shubham1_open data/shubham1_closed \
         data/gazania_open data/gazania_closed \
  --models emg+imu --device cuda --epochs 60 \
  --output-dir runs/gripper_state_attention_v2
```

Compare V2 against V1 on the identical saved split. The acceptance targets are
gripper macro-F1 above 0.98, fourth-quarter pixel error below 68.9 px, current
XYZ no worse than 3.02 cm, and 200 ms future XYZ no worse than 4.01 cm. Run
multiple seeds before treating any improvement as evidence.
