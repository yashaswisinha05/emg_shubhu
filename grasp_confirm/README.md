# Grasp confirmation from 4-channel forearm EMG

Detect whether the hand is **actually loaded by an object** or only closed on nothing, from four surface EMG channels on the forearm. Standalone — nothing here imports from the rest of this repository.

Three classes:

| Class | What it is | Difficulty |
| --- | --- | --- |
| `REST` | hand open, relaxed, nothing happening | trivial — any RMS threshold gets it |
| `AIR` | hand closed and clenched on nothing | ⟵ |
| `GRASP` | hand closed on an object, bearing load | **the entire result lives in this one boundary** |

`REST` is there so a deployed model can run continuously without firing between actions. All the design effort goes into making `AIR` and `GRASP` *not* separable by amplitude.

---

## Why this is possible from EMG alone

No motion capture, no glove, no accelerometer, no force sensor. Loading the hand changes forearm EMG in four ways that have nothing to do with kinematics:

1. **Sustain shape.** Closing on an object arrests the fingers at the object's diameter and the flexors go isometric against a real reaction force — amplitude *plateaus*. Closing on air runs to full flexion and the burst *decays*.
2. **Co-activation geometry.** An object applies a reaction torque at the wrist, so extensors recruit to stabilise it. The flexor:extensor ratio under load differs from free-air closure *at the same total amplitude*. This lives in the between-channel structure, not in any single channel.
3. **Envelope micro-modulation.** Holding a real object is closed-loop — grip force is continuously corrected against anticipated slip, producing 0.5–5 Hz fluctuation in the envelope. Squeezing air is open-loop and smoother.
4. **Recruitment.** Sustained load recruits higher-threshold motor units; the spectral drift over a hold differs from unloaded co-contraction.

**Amplitude is the confound.** A person can squeeze air harder than they hold a mug. Everything below is arranged so the model cannot cheat with loudness, and `qc.py` checks that the arrangement worked.

---

## Trial structure: start → act immediately → stop

The hand is **pre-positioned**: open, around the object, *not touching it* — or around the same empty space for `AIR`. On the GO beep the subject closes immediately and holds until STOP.

There is no reach phase. That is deliberate and it is the strongest feature of this protocol: `AIR` and `GRASP` differ **only** in whether an object is inside the closing hand, so reach kinematics cannot leak into the label.

```
0.0  cue text appears  ("BOTTLE - firm" / "AIR - hard" / "NOTHING")
     subject positions the open hand, NOT touching the object
2.0  beep GO      -> close immediately and hold
5.0  beep STOP    -> open, relax
6.5  trial ends
```

The labelled window is **not** at a fixed offset. Reaction time varies, so it is anchored to the EMG onset detected inside GO..STOP:

```
onset                first sustained crossing of the baseline threshold
onset + 0.30 s       window starts (skips the closing transient)
       + 2.00 s      window ends, clipped at STOP
```

The onset threshold is `baseline_mean + 4 × baseline_std`, computed from **that session's own rest recording**, so it rescales automatically across sessions and electrode re-applications. `NOTHING` trials have no onset by construction — if one is detected the subject moved and the trial is dropped, which `dataset.py` reports. All of this lives in [`timeline.py`](timeline.py); change it there and re-segment, never re-record.

---

## What data you need

### Hardware

- 4 surface EMG channels, **raw signal at ≥1 kHz** (2 kHz preferred). Not the smoothed RMS channel — an envelope stream at ~150 Hz destroys the covariance and spectral features that items 2–4 above depend on. `qc.py` warns if it sees a low rate.
- Nothing else.

### Electrode placement — 2 flexors + 2 extensors, not 4 flexors

The co-activation axis has to be well conditioned or item 2 is unavailable:

| Ch | Muscle | Site |
| --- | --- | --- |
| 1 | flexor digitorum superficialis | volar, ~⅓ down from elbow, ulnar side |
| 2 | flexor carpi radialis | volar, radial |
| 3 | extensor digitorum communis | dorsal, ~⅓ down from elbow |
| 4 | extensor carpi ulnaris | dorsal, ulnar |

Record what you actually did in `--placement-notes`; it lands in `session.json`.

### Objects — measure them, don't guess

Edit [`objects.json`](objects.json) with **measured** values (kitchen scale, ruler) and set `"measured": true`. `schedule.py` warns on every run until you do.

| Object | Mass | Grasp | Why it is in the set |
| --- | --- | --- | --- |
| empty plastic/paper cup | ~15 g | wide power | **low EMG but genuinely loaded** — the case that breaks amplitude-only baselines |
| full 500 ml bottle | ~530 g | wide power | high EMG, high load |
| pen / thin marker | ~10 g | precision pinch | light, completely different synergy |
| foam / stress ball | ~60 g | compliant power | contact with no rigid arrest; held out as the unseen object |

These property vectors are not decoration. They are the **anchor modality** the cross-modal model aligns EMG to, and the space in which unseen-object generalisation is measured.

### Session composition — 84 trials, ~9 min of cue time

| Condition | n | Instruction |
| --- | --- | --- |
| `AIR_LIGHT` | 12 | "close and hold **gently**, like an egg" |
| `AIR_HARD` | 12 | "close and **squeeze hard**, as if the bottle were slipping" |
| `GRASP` | 48 | 4 objects × 2 efforts (normal / firm) × 6 |
| `NOTHING` | 12 | "do nothing, stay still" |

`AIR_LIGHT` and `AIR_HARD` **bracket** the GRASP amplitude distribution from below and above. That bracketing is the only thing standing between this experiment and an RMS threshold. Read the instructions **verbatim** to every subject — the wording is doing experimental work.

### Scale

**3 subjects × 2 sessions × 84 trials = 504 trials**, about 45 min per subject including setup.

Peel and re-apply the electrodes between the two sessions. Those five minutes are what turn a random split into leave-one-session-out.

### Controls that cost nothing

- **Identical hand position for AIR and GRASP.** Mark the spot. On `AIR` trials the hand closes around that same empty space, same posture, same height. If posture co-varies with class, you have measured posture.
- **Per-session MVC and rest baseline.** `record.py` collects both automatically. The baseline is not optional — the onset detector derives its threshold from it.
- **Self-report.** After each trial the subject types a note with the other hand if anything went wrong. Stored in `report.csv`, surfaced by `dataset.py`.

---

## Running it

Needs `numpy`, `scipy`, `scikit-learn`, `matplotlib`. On this machine that is the `smss` conda env:

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/smss/bin/python -m pytest grasp_confirm/test_features.py -q
```

**1. Dry run — no hardware, fake data, proves the pipeline end to end.**

```bash
python grasp_confirm/schedule.py --subject dry --session s1 --out-root /tmp/gc
```

```bash
python grasp_confirm/record.py --subject dry --session s1 --out-root /tmp/gc --recorder synthetic --limit 8 --baseline-s 3 --mvc-reps 1
```

Do this once before anyone is wired up. The synthetic recorder invents data and stamps `synthetic: true` into `session.json`; `qc.py` refuses to issue a verdict when it sees that flag, so a dry run can never be mistaken for a result.

**2. Check the hardware, before anyone is wired up.**

```bash
python -m grasp_confirm.recorders.delsys --sdk-path /path/to/Example-Applications/Python
```

**3. Real session.**

```bash
python grasp_confirm/schedule.py --subject subj01 --session sess01
```

```bash
python grasp_confirm/record.py --subject subj01 --session sess01 --recorder delsys --recorder-arg sdk-path=/path/to/Example-Applications/Python --placement-notes "FDS/FCR volar, EDC/ECU dorsal, 1/3 from elbow"
```

**4. Segment and check what you have.**

```bash
python -m grasp_confirm.dataset --root data
```

Read the drop report. Trials dropped for "no onset detected" mean the subject did not act or the threshold is too high; trials dropped for "moved during NOTHING" are a genuine quality catch.

**5. The QC gate.**

```bash
python -m grasp_confirm.qc --root data
```

---

## The QC gate — run it before you train anything

1. **Amplitude overlap.** AUC of mean log-RMS, AIR vs GRASP. `0.5` = matched (what you want). Above `0.75` means GRASP is simply louder.
2. **Amplitude-only baseline.** Logistic regression on 4-channel log-RMS, held out by subject. Above `0.70` balanced accuracy means effort matching failed.
3. **Structure-only baseline.** The same classifier on gain-invariant features — trace-normalised channel covariance and 0.5–5 Hz envelope ripple. Multiplying every channel by a constant leaves these unchanged (there is a test for exactly that), so this classifier *cannot* be reading contraction level. **This is the number the night is for.**
4. **Per-condition breakdown.** `AIR_LIGHT` and `AIR_HARD` against GRASP separately. `AIR_HARD` is the adversarial pair — those trials are *louder* than the grasps and still have to come out as not-grasp.

A good outcome: check 1 near 0.5, check 2 near chance, check 3 well above it, check 4 showing no collapse on `AIR_HARD`.

**If the gate fails, re-record a block.** No architecture rescues a leaked protocol, and a 97% number from leaked data is worse than no number, because you will not find out until review.

---

## Files

| File | |
| --- | --- |
| `timeline.py` | trial timing and segmentation policy — the single source of truth |
| `objects.json` | object property vectors (the anchor modality). **Measure these.** |
| `schedule.py` | randomised trial order for one session |
| `record.py` | cue engine, beeps, preflight, baseline + MVC, continuous capture |
| `recorders/` | pluggable EMG sources — `base.py` is a three-method interface |
| `features.py` | filtering, envelope, onset detection, the two feature sets |
| `dataset.py` | onset-anchored segmentation into labelled windows |
| `qc.py` | the gate |
| `test_features.py` | proves the gain-invariance property. Run it after touching `features.py`. |

## Data layout

```
data/<subject>/<session>/
    session.json    metadata, channel names, sample rate, placement notes,
                    preflight RMS, cue jitter, synthetic flag
    schedule.csv    planned trial order
    objects.json    snapshot of the property vectors used
    cues.csv        every cue event with its ACTUAL perf_counter timestamp
    marks.csv       baseline / MVC / preflight intervals
    raw.npz         t (n,), x (n, C) float32 -- the continuous recording
    report.csv      per-trial self-report flags
```

`raw.npz` plus `cues.csv` are the source of truth. Nothing is cut at record time, so redefining a window is a re-segmentation, never a re-recording.

---

## Known limitations, stated plainly

- **No within-trial control.** An earlier version of this protocol had a transition trial (grasp → drop → re-clench) whose GRASP and AIR windows came from the same trial one second apart. It was dropped from the design. Without it, nothing in the dataset rules out session, posture or intent as the explanation for a positive result. The strongest remaining claim is the amplitude dissociation in QC check 4: `AIR_HARD` is louder than GRASP and still classified as not-grasp. Scope the writing to that, not to overall accuracy.
- **Cued, not surprise, failures.** The subject knows before the beep whether an object is there, so `AIR` trials carry feedforward intent as well as absent load.
- **`recorders/delsys.py` is untested.** It needs the AeroPy SDK and an attached base station, neither of which existed where this was written. It reuses your existing `EMGCollector` rather than reimplementing hardware calls. Run its self-test before the session. Everything else has been executed end to end against the synthetic recorder.
- **Beep latency is logged, not compensated.** Cues are played by spawning `afplay`/`aplay`, costing tens of milliseconds. This is tolerable because the labelled window is anchored to the detected EMG onset, not to the beep. `session.json` records the worst cue jitter observed.
- **Detected onsets sit ~30–60 ms early.** The envelope uses a zero-phase filter, which smears energy backwards. Harmless for windowing; do not report it as physiological anticipation.
- **n=3 is n=3.** Report per-subject numbers, not a mean and a standard error over three people.
