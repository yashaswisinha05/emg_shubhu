#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from all completed EMG-touch results."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
RUNS = ROOT / "runs"
OUTPUT = ROOT / "reports" / "emg_touch_results_dashboard.html"


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: scalar(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_record(
    row: dict[str, Any], experiment: str, source: Path
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": row.get("model_kind") or row.get("model") or "unknown",
        "configuration": row.get("configuration") or "unknown",
        "cutoff": str(row.get("requested_cutoff") or row.get("cutoff") or "touch"),
        "trials": row.get("held_out_trials") or row.get("count"),
        "participants": row.get("participants"),
        "bootstrapUnit": row.get("bootstrap_unit"),
        "median": row.get("median_pixel_error"),
        "medianLow": row.get("median_pixel_error_ci95_low"),
        "medianHigh": row.get("median_pixel_error_ci95_high"),
        "mean": row.get("mean_pixel_error"),
        "p90": row.get("p90_pixel_error"),
        "within50": row.get("accuracy_within_50px"),
        "within100": row.get("accuracy_within_100px"),
        "targetBox": row.get("target_box_accuracy")
        if row.get("target_box_accuracy") is not None
        else row.get("within_target_box"),
        "screenRegion": row.get("screen_region_accuracy")
        if row.get("screen_region_accuracy") is not None
        else row.get("grid_cell_accuracy"),
        "meanNormalized": row.get("mean_normalized_error"),
        "source": str(source.relative_to(ROOT)),
    }


def summary_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in sorted(EVALUATION.glob("*/*_configuration_accuracy.csv")):
        experiment = path.parent.name
        for row in read_csv(path):
            record = normalized_record(row, experiment, path)
            key = (
                record["experiment"],
                str(record["model"]),
                str(record["configuration"]),
                record["cutoff"],
            )
            if key not in seen:
                seen.add(key)
                records.append(record)

    # The long-context study was interrupted before aggregation. Its completed
    # IMU and EMG checkpoints still have valid held-out continual metrics.
    long_root = RUNS / "hf_patchtst_long_context" / "mix7" / "fold-0"
    for model_dir in sorted(long_root.glob("hf_patchtst_*")):
        path = model_dir / "continual_metrics.json"
        if not path.is_file() or not (model_dir / "predictions.csv").is_file():
            continue
        for cutoff, metrics in read_json(path).items():
            row = dict(metrics)
            row.update(
                {
                    "model_kind": model_dir.name,
                    "configuration": "mix7",
                    "requested_cutoff": cutoff,
                    "held_out_trials": metrics.get("count"),
                    "participants": 4,
                }
            )
            record = normalized_record(row, "hf_patchtst_long_context", path)
            key = (
                record["experiment"],
                str(record["model"]),
                str(record["configuration"]),
                record["cutoff"],
            )
            if key not in seen:
                seen.add(key)
                records.append(record)

    # Earliest test/dev baseline evaluation, retained as its own protocol.
    baseline_root = EVALUATION / "a1" / "test-dev"
    for path in sorted(baseline_root.glob("*/metrics.json")):
        model = path.parent.name
        for cutoff, metrics in read_json(path).items():
            row = dict(metrics)
            row.update(
                {
                    "model_kind": model,
                    "configuration": "a1",
                    "requested_cutoff": cutoff,
                    "held_out_trials": metrics.get("count"),
                }
            )
            records.append(normalized_record(row, "baseline_a1_test_dev", path))
    return records


def load_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVALUATION.glob(pattern)):
        experiment = path.parent.name
        for row in read_csv(path):
            row["experiment"] = experiment
            row["source"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def patchtst_histories() -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for experiment in ("hf_patchtst_exact", "hf_patchtst_long_context"):
        root = RUNS / experiment
        for path in sorted(root.glob("*/*/hf_patchtst_*/history.csv")):
            parts = path.relative_to(root).parts
            points = []
            for row in read_csv(path):
                points.append(
                    {
                        "epoch": row.get("epoch"),
                        "selection": row.get("selection_value"),
                        "valMean": row.get("val_mean_pixel_error"),
                        "valEndpoint": row.get("val_endpoint_mean_pixel_error"),
                        "trainLoss": row.get("train_loss"),
                        "learningRate": row.get("learning_rate"),
                    }
                )
            complete = (path.parent / "test_metrics.json").is_file() and (
                path.parent / "predictions.csv"
            ).is_file()
            histories.append(
                {
                    "experiment": experiment,
                    "configuration": parts[0],
                    "fold": parts[1],
                    "model": parts[2],
                    "complete": complete,
                    "points": points,
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return histories


def statuses() -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/study_status.json")):
        value = read_json(path)
        value["experiment"] = path.parent.name
        value["source"] = str(path.relative_to(ROOT))
        model_root = path.parent
        completed_models = []
        partial_models = []
        candidate_dirs = {
            result.parent
            for pattern in ("test_metrics.json", "predictions.csv", "history.csv", "best.pt")
            for result in model_root.rglob(pattern)
        }
        for model_dir in sorted(candidate_dirs):
            has_final = (model_dir / "test_metrics.json").is_file() and (
                model_dir / "predictions.csv"
            ).is_file()
            has_partial = (model_dir / "history.csv").is_file() or (
                model_dir / "best.pt"
            ).is_file()
            label = "/".join(model_dir.relative_to(model_root).parts)
            if has_final:
                completed_models.append(label)
            elif has_partial:
                partial_models.append(label)
        value["completedModels"] = completed_models
        value["partialModels"] = partial_models
        values.append(value)
    return values


def inventory() -> dict[str, Any]:
    return {
        "testMetricFiles": len(list(RUNS.glob("**/test_metrics.json"))),
        "metricFiles": len(list(RUNS.glob("**/metrics.json"))),
        "predictionFiles": len(list(RUNS.glob("**/predictions.csv"))),
        "historyFiles": len(list(RUNS.glob("**/history.csv"))),
        "evaluationFiles": len([path for path in EVALUATION.rglob("*") if path.is_file()]),
    }


def build_payload() -> dict[str, Any]:
    return {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "summary": summary_records(),
        "temporal": load_rows("*/temporal_window_results.csv"),
        "selectedTemporal": load_rows("*/validation_selected_test_results.csv"),
        "fusionPairs": load_rows("*/fusion_vs_imu.csv"),
        "directional": load_rows("*/directional_error.csv"),
        "attention": load_rows("*/channel_attention_summary.csv"),
        "histories": patchtst_histories(),
        "statuses": statuses(),
        "inventory": inventory(),
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EMG Touch — Results Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #07111f;
      --surface: #0d1a2b;
      --surface-2: #122238;
      --surface-3: #182b44;
      --line: #28405d;
      --text: #ecf4ff;
      --muted: #93a7c1;
      --blue: #55a7ff;
      --cyan: #50dfd4;
      --green: #65d68a;
      --orange: #ffb45e;
      --pink: #ef7bb4;
      --purple: #a88cff;
      --red: #ff7c78;
      --shadow: 0 20px 70px rgba(0,0,0,.27);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background:
        radial-gradient(900px 520px at 8% -10%, rgba(85,167,255,.18), transparent 65%),
        radial-gradient(800px 520px at 95% 0%, rgba(80,223,212,.10), transparent 62%),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    a { color: inherit; }
    .shell { max-width: 1480px; margin: 0 auto; padding: 0 28px 80px; }
    .topbar {
      position: sticky; top: 0; z-index: 20;
      display: flex; align-items: center; justify-content: space-between; gap: 24px;
      min-height: 64px; margin: 0 -28px; padding: 0 28px;
      background: rgba(7,17,31,.86); border-bottom: 1px solid rgba(40,64,93,.7);
      backdrop-filter: blur(18px);
    }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 750; letter-spacing: -.02em; }
    .brand-mark { width: 30px; height: 30px; border-radius: 9px; background: linear-gradient(135deg,var(--blue),var(--cyan)); box-shadow: 0 0 28px rgba(80,223,212,.28); }
    nav { display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none; }
    nav a { text-decoration: none; color: var(--muted); font-size: 13px; font-weight: 650; padding: 8px 10px; border-radius: 8px; white-space: nowrap; }
    nav a:hover { color: var(--text); background: var(--surface-2); }
    .hero { padding: 74px 0 30px; display: grid; grid-template-columns: minmax(0,1.3fr) minmax(320px,.7fr); gap: 34px; align-items: end; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .16em; font-size: 12px; font-weight: 800; }
    h1 { font-size: clamp(38px,6vw,76px); line-height: .98; letter-spacing: -.055em; margin: 12px 0 22px; max-width: 900px; }
    .hero p { color: var(--muted); font-size: 17px; max-width: 800px; margin: 0; }
    .run-state { border: 1px solid var(--line); background: linear-gradient(135deg,rgba(18,34,56,.95),rgba(13,26,43,.82)); border-radius: 18px; padding: 22px; box-shadow: var(--shadow); }
    .run-state strong { display: block; color: var(--green); font-size: 18px; margin-bottom: 6px; }
    .run-state span { color: var(--muted); font-size: 13px; }
    .kpis { display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap: 12px; margin: 20px 0 52px; }
    .kpi { min-height: 146px; border: 1px solid var(--line); background: rgba(13,26,43,.82); border-radius: 16px; padding: 19px; }
    .kpi .label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .kpi .value { margin: 15px 0 4px; font-size: clamp(25px,3vw,38px); line-height: 1; letter-spacing: -.04em; font-weight: 780; }
    .kpi .detail { color: var(--muted); font-size: 12px; }
    section { scroll-margin-top: 80px; margin: 0 0 70px; }
    .section-head { display: flex; justify-content: space-between; gap: 20px; align-items: end; margin-bottom: 20px; }
    .section-head h2 { font-size: 30px; letter-spacing: -.035em; margin: 0 0 6px; }
    .section-head p { color: var(--muted); margin: 0; max-width: 780px; }
    .protocol-note { max-width: 470px; color: var(--orange); font-size: 12px; text-align: right; }
    .grid-2 { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 16px; }
    .panel { border: 1px solid var(--line); background: rgba(13,26,43,.88); border-radius: 18px; padding: 20px; box-shadow: var(--shadow); overflow: hidden; }
    .panel.wide { grid-column: 1/-1; }
    .panel-head { display: flex; align-items: start; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
    .panel h3 { margin: 0; font-size: 17px; letter-spacing: -.015em; }
    .panel .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .controls { display: flex; flex-wrap: wrap; gap: 9px; align-items: end; margin-bottom: 15px; }
    label.control { display: grid; gap: 5px; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
    select, input { min-width: 150px; border: 1px solid var(--line); border-radius: 9px; color: var(--text); background: var(--surface-2); padding: 9px 10px; font: inherit; font-size: 13px; outline: none; }
    input { min-width: 240px; }
    select:focus, input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(85,167,255,.12); }
    .chart { min-height: 300px; width: 100%; position: relative; }
    .chart svg { display: block; width: 100%; height: auto; overflow: visible; }
    .axis { stroke: var(--line); stroke-width: 1; }
    .gridline { stroke: rgba(40,64,93,.48); stroke-width: 1; }
    .chart text { fill: var(--muted); font-size: 11px; }
    .chart .value-label, .chart .series-label { fill: var(--text); font-weight: 700; }
    .chart .axis-title { fill: var(--muted); font-size: 11px; font-weight: 700; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin: 6px 0 10px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .tooltip { position: fixed; z-index: 40; pointer-events: none; opacity: 0; transform: translate(12px,12px); background: #f4f8ff; color: #08111e; border-radius: 9px; padding: 9px 11px; font-size: 12px; box-shadow: 0 12px 34px rgba(0,0,0,.34); max-width: 300px; transition: opacity .08s; }
    .callout { padding: 14px 16px; border-left: 3px solid var(--orange); background: rgba(255,180,94,.08); color: #ffd8aa; font-size: 13px; border-radius: 5px 12px 12px 5px; margin-bottom: 15px; }
    .callout.good { border-color: var(--green); background: rgba(101,214,138,.08); color: #bff1ce; }
    .table-wrap { overflow: auto; max-height: 620px; border: 1px solid var(--line); border-radius: 12px; }
    table { border-collapse: collapse; width: 100%; min-width: 1000px; font-size: 12px; }
    th { position: sticky; top: 0; z-index: 2; background: var(--surface-3); color: #c7d7eb; text-align: left; padding: 11px 12px; border-bottom: 1px solid var(--line); cursor: pointer; white-space: nowrap; }
    td { padding: 10px 12px; border-bottom: 1px solid rgba(40,64,93,.5); color: #c8d4e4; white-space: nowrap; }
    tbody tr:hover td { background: rgba(85,167,255,.055); }
    .num { font-variant-numeric: tabular-nums; text-align: right; }
    .status-list { display: grid; gap: 11px; }
    .status-row { display: grid; grid-template-columns: 210px 150px 1fr; gap: 16px; align-items: center; padding: 14px 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface-2); }
    .badge { width: fit-content; padding: 4px 8px; border-radius: 99px; background: rgba(101,214,138,.12); color: var(--green); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
    .badge.warn { background: rgba(255,180,94,.12); color: var(--orange); }
    .status-detail { color: var(--muted); font-size: 12px; }
    .source { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; overflow-wrap: anywhere; }
    .foot { border-top: 1px solid var(--line); padding-top: 24px; color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; gap: 20px; }
    @media (max-width: 1100px) { .kpis { grid-template-columns: repeat(3,1fr); } .hero { grid-template-columns: 1fr; } }
    @media (max-width: 760px) { .shell { padding: 0 16px 60px; } .topbar { margin: 0 -16px; padding: 0 16px; } nav { display: none; } .kpis, .grid-2 { grid-template-columns: 1fr; } .section-head { display: block; } .protocol-note { text-align: left; margin-top: 8px; } .status-row { grid-template-columns: 1fr; gap: 7px; } h1 { font-size: 46px; } }
    @media print { body { background: #fff; color: #111; } .topbar { position: static; } .panel,.kpi,.run-state { box-shadow: none; break-inside: avoid; } nav,.controls { display: none; } }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand"><span class="brand-mark"></span>EMG Touch Results</div>
    <nav>
      <a href="#overview">Overview</a><a href="#configurations">Configurations</a><a href="#continual">Continual</a><a href="#temporal">EMG timing</a><a href="#diagnostics">Diagnostics</a><a href="#all-results">All results</a><a href="#status">Status</a>
    </nav>
  </header>

  <main>
    <div class="hero">
      <div>
        <div class="eyebrow">Touch endpoint prediction · consolidated evidence</div>
        <h1>What worked, what did not, and where EMG adds value.</h1>
        <p>Every finalized configuration summary is presented with its own evaluation protocol. Lower pixel error is better; confidence intervals and held-out trial counts are retained wherever they were produced.</p>
      </div>
      <div class="run-state"><strong>All training stopped</strong><span>No EMG-touch training or evaluation process was running when this report was generated. The 15.6-second PatchTST fusion experiment is explicitly marked interrupted.</span></div>
    </div>

    <div class="kpis" id="kpis"></div>

    <section id="overview">
      <div class="section-head"><div><h2>Executive comparison</h2><p>A compact view of the main mix7 endpoint results. This is useful for direction-setting, but bars from different protocols must not be treated as a single leaderboard.</p></div><div class="protocol-note">Full trajectory and temporal-touch are five-fold / 800 held-out trials; hybrid, continual-attention and PatchTST are fold-0 / 167 trials.</div></div>
      <div class="grid-2">
        <div class="panel wide">
          <div class="panel-head"><div><h3>mix7 endpoint error by architecture and modality</h3><div class="sub">Median Euclidean pixel error · lower is better</div></div></div>
          <div class="legend" id="modalityLegend"></div>
          <div class="chart" id="architectureChart"></div>
        </div>
        <div class="panel">
          <div class="panel-head"><div><h3>Exact PatchTST context comparison</h3><div class="sub">Same mix7 fold, short versus 15.6-second fixed context</div></div></div>
          <div class="callout">Long-context fusion has no final result: it was stopped after severe MPS paging. IMU and EMG values are valid completed checkpoints.</div>
          <div class="chart" id="contextChart"></div>
        </div>
        <div class="panel">
          <div class="panel-head"><div><h3>PatchTST validation learning curves</h3><div class="sub">Weighted mean pixel error used for checkpoint selection</div></div></div>
          <div class="controls"><label class="control">Run<select id="historySelect"></select></label></div>
          <div class="chart" id="historyChart"></div>
        </div>
      </div>
    </section>

    <section id="configurations">
      <div class="section-head"><div><h2>Configuration comparison</h2><p>Compare a model across a1–a4, b1–b3 and mix configurations. The full-trajectory study aggregates held-out predictions over available folds; unequal participant counts are allowed by design.</p></div></div>
      <div class="panel">
        <div class="controls">
          <label class="control">Experiment<select id="configExperiment"></select></label>
          <label class="control">Model<select id="configModel"></select></label>
          <label class="control">Cutoff<select id="configCutoff"></select></label>
          <label class="control">Metric<select id="configMetric"><option value="median">Median pixel error</option><option value="mean">Mean pixel error</option><option value="p90">P90 pixel error</option><option value="within100">Accuracy within 100 px</option><option value="targetBox">Target-box accuracy</option></select></label>
        </div>
        <div class="chart" id="configurationChart"></div>
      </div>
    </section>

    <section id="continual">
      <div class="section-head"><div><h2>Continual inference</h2><p>How error changes as new causal information arrives. “Touch” uses the complete available causal trajectory; the 0.4-second point contains only the subset of trials that reached that duration.</p></div></div>
      <div class="grid-2">
        <div class="panel wide">
          <div class="controls"><label class="control">Experiment<select id="continualExperiment"></select></label><label class="control">Metric<select id="continualMetric"><option value="median">Median pixel error</option><option value="mean">Mean pixel error</option><option value="within100">Accuracy within 100 px</option><option value="targetBox">Target-box accuracy</option></select></label></div>
          <div class="chart" id="continualChart"></div>
        </div>
        <div class="panel">
          <div class="panel-head"><div><h3>Fusion versus IMU</h3><div class="sub">Positive paired mean gain means EMG helped</div></div></div>
          <div class="controls"><label class="control">Experiment<select id="pairExperiment"></select></label></div>
          <div class="chart" id="pairChart"></div>
        </div>
        <div class="panel">
          <div class="panel-head"><div><h3>Current conclusion</h3></div></div>
          <div class="callout good">The continual-attention fusion model reached 122.2 px median at touch versus 126.4 px for IMU, but the paired mean gain was only +0.69 px. The temporal-touch five-fold study found a small, statistically positive +3.08 px mean gain for all causal EMG.</div>
          <div class="callout">EMG-only remains much weaker. The evidence supports EMG as a small residual correction or gate on top of calibrated IMU—not as the primary endpoint sensor.</div>
        </div>
      </div>
    </section>

    <section id="temporal">
      <div class="section-head"><div><h2>Where EMG adds value</h2><p>Paired trajectory analysis across five folds. Confidence intervals crossing zero indicate that the window’s improvement is not established at the 95% level.</p></div></div>
      <div class="grid-2">
        <div class="panel wide">
          <div class="controls"><label class="control">Temporal study<select id="temporalExperiment"></select></label></div>
          <div class="chart" id="temporalGainChart"></div>
        </div>
        <div class="panel wide">
          <div class="panel-head"><div><h3>Standalone EMG versus fused endpoint error</h3><div class="sub">Median pixel error by temporal window</div></div></div>
          <div class="chart" id="temporalErrorChart"></div>
        </div>
      </div>
    </section>

    <section id="diagnostics">
      <div class="section-head"><div><h2>Bias and channel diagnostics</h2><p>Directionality reveals centre collapse more directly than radial error alone. Channel attention reports which sensor channels received the highest average learned weight.</p></div></div>
      <div class="grid-2">
        <div class="panel">
          <div class="controls"><label class="control">Experiment<select id="directionExperiment"></select></label><label class="control">Cutoff<select id="directionCutoff"></select></label></div>
          <div class="chart" id="directionChart"></div>
        </div>
        <div class="panel">
          <div class="controls"><label class="control">Model<select id="attentionModel"></select></label><label class="control">Cutoff<select id="attentionCutoff"></select></label></div>
          <div class="chart" id="attentionChart"></div>
        </div>
      </div>
    </section>

    <section id="all-results">
      <div class="section-head"><div><h2>All summarized metrics</h2><p>Search, filter and sort every finalized configuration-level result included in the report. Click a column heading to sort.</p></div></div>
      <div class="panel">
        <div class="controls"><label class="control">Search<input id="resultSearch" placeholder="model, config, experiment…"></label><label class="control">Experiment<select id="tableExperiment"></select></label><label class="control">Cutoff<select id="tableCutoff"></select></label></div>
        <div class="table-wrap"><table><thead><tr><th data-key="experiment">Experiment</th><th data-key="model">Model</th><th data-key="configuration">Config</th><th data-key="cutoff">Cutoff</th><th data-key="trials">Trials</th><th data-key="median">Median px</th><th data-key="mean">Mean px</th><th data-key="p90">P90 px</th><th data-key="within100">≤100 px</th><th data-key="targetBox">Target box</th><th data-key="source">Source</th></tr></thead><tbody id="resultsBody"></tbody></table></div>
      </div>
    </section>

    <section id="status">
      <div class="section-head"><div><h2>Run status and provenance</h2><p>Only directories with both final test metrics and prediction files are counted as completed model outputs.</p></div></div>
      <div class="panel"><div class="status-list" id="statusList"></div></div>
    </section>
  </main>

  <footer class="foot"><span id="generatedAt"></span><span id="inventory"></span></footer>
</div>
<div class="tooltip" id="tooltip"></div>
<script id="dashboard-data" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById('dashboard-data').textContent);
  const $ = id => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';
  const colors = {IMU:'#55a7ff', Fusion:'#50dfd4', EMG:'#ef7bb4', Baseline:'#ffb45e', Other:'#a88cff'};
  const expNames = {full_trajectory:'Full trajectory · 5-fold',grid_point:'Grid point · 5-fold',hybrid_point:'Hybrid point · fold 0',continual_attention:'Continual + channel attention · fold 0',hf_patchtst_exact:'Exact PatchTST · 256 samples',hf_patchtst_long_context:'Exact PatchTST · 2,304 samples',baseline_a1_test_dev:'Initial a1 test/dev baseline',emg_temporal_touch:'Temporal touch-aligned · 5-fold',emg_temporal_study:'Temporal reaction-aligned · 5-fold'};
  const pretty = value => String(value ?? '—').replace(/^hf_patchtst_/,'PatchTST ').replace(/^grid_/,'Grid ').replaceAll('_',' ').replace(/\b\w/g, c => c.toUpperCase());
  const expLabel = x => expNames[x] || pretty(x);
  const modality = model => /fusion|multimodal|residual/i.test(model) ? 'Fusion' : /imu/i.test(model) ? 'IMU' : /emg|student|tcn/i.test(model) ? 'EMG' : /mean|baseline/i.test(model) ? 'Baseline' : 'Other';
  const number = (v,d=1) => v == null || Number.isNaN(+v) ? '—' : (+v).toFixed(d);
  const percent = v => v == null || Number.isNaN(+v) ? '—' : `${(+v*100).toFixed(1)}%`;
  const tooltip = $('tooltip');
  const showTip = (event, html) => { tooltip.innerHTML = html; tooltip.style.left = `${event.clientX}px`; tooltip.style.top = `${event.clientY}px`; tooltip.style.opacity = 1; };
  const hideTip = () => tooltip.style.opacity = 0;
  const svgEl = (tag, attrs={}) => { const e=document.createElementNS(SVG,tag); Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v)); return e; };
  const clear = id => { const el=$(id); el.innerHTML=''; return el; };
  const uniq = values => [...new Set(values.filter(v => v != null))];
  const cutoffOrder = value => { const v=String(value).toLowerCase(); if(v==='touch'||v==='full') return 99; const n=parseFloat(v); return Number.isNaN(n)?98:n; };
  const setOptions = (select, values, formatter=v=>v, preferred=null, includeAll=false) => { const old=select.value; select.innerHTML=''; if(includeAll){const o=document.createElement('option');o.value='all';o.textContent='All';select.append(o);} values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=formatter(v);select.append(o);}); if(values.includes(preferred))select.value=preferred; else if([...select.options].some(o=>o.value===old))select.value=old; };
  function makeSvg(container, height, minWidth=620){ const width=Math.max(container.clientWidth||900,minWidth); const svg=svgEl('svg',{viewBox:`0 0 ${width} ${height}`,role:'img'}); container.append(svg); return {svg,width,height}; }
  function scale(value,a,b,c,d){ return c+(value-a)*(d-c)/Math.max(b-a,1e-9); }
  function ticks(max,count=5){ const step=max/count; return Array.from({length:count+1},(_,i)=>step*i); }

  function horizontalBars(id, rows, opts={}) {
    const container=clear(id); if(!rows.length){container.innerHTML='<div class="callout">No finalized rows for this selection.</div>';return;}
    const metric=opts.metric||'median', accuracy=/within|target|screen/i.test(metric);
    const values=rows.map(r=>+r[metric]).filter(Number.isFinite); const rawMax=Math.max(...values,accuracy?1:0); const max=accuracy?Math.min(1,Math.max(.1,rawMax*1.15)):Math.max(10,rawMax*1.14);
    const labelLength=Math.max(...rows.map(r=>String(opts.label?opts.label(r):r.label).length));
    const rowH=34, margin={left:Math.min(310,Math.max(145,labelLength*7)),right:58,top:14,bottom:42};
    const height=margin.top+rows.length*rowH+margin.bottom; const {svg,width}=makeSvg(container,height,680); const x0=margin.left,x1=width-margin.right;
    ticks(max,5).forEach(t=>{const x=scale(t,0,max,x0,x1);svg.append(svgEl('line',{x1:x,y1:margin.top,x2:x,y2:height-margin.bottom,class:'gridline'}));const tx=svgEl('text',{x,y:height-margin.bottom+18,'text-anchor':'middle'});tx.textContent=accuracy?`${Math.round(t*100)}%`:number(t,0);svg.append(tx);});
    rows.forEach((r,i)=>{const y=margin.top+i*rowH+6;const value=+r[metric];const label=opts.label?opts.label(r):r.label;const c=opts.color?opts.color(r):(colors[modality(r.model)]||colors.Other);const tx=svgEl('text',{x:x0-10,y:y+14,'text-anchor':'end',class:'series-label'});tx.textContent=label;svg.append(tx);const bar=svgEl('rect',{x:x0,y,width:Math.max(1,scale(value,0,max,x0,x1)-x0),height:18,rx:4,fill:c,opacity:.88});bar.addEventListener('mousemove',e=>showTip(e,opts.tip?opts.tip(r):`<b>${label}</b><br>${accuracy?percent(value):number(value)}${accuracy?'':' px'}`));bar.addEventListener('mouseleave',hideTip);svg.append(bar);const val=svgEl('text',{x:Math.min(x1+5,scale(value,0,max,x0,x1)+7),y:y+14,class:'value-label'});val.textContent=accuracy?percent(value):number(value);svg.append(val);});
    const title=svgEl('text',{x:(x0+x1)/2,y:height-7,'text-anchor':'middle',class:'axis-title'});title.textContent=accuracy?'Accuracy':'Euclidean pixel error (px)';svg.append(title);
  }

  function lineChart(id, series, opts={}) {
    const container=clear(id); if(!series.length){container.innerHTML='<div class="callout">No data available.</div>';return;}
    const labels=opts.labels||uniq(series.flatMap(s=>s.points.map(p=>p.x))); const all=series.flatMap(s=>s.points.map(p=>+p.y)).filter(Number.isFinite); const accuracy=opts.accuracy; let ymin=accuracy?0:Math.max(0,Math.min(...all)*.82), ymax=accuracy?Math.min(1,Math.max(.1,Math.max(...all)*1.18)):Math.max(...all)*1.12; if(ymax<=ymin)ymax=ymin+1;
    const margin={left:72,right:32,top:26,bottom:58};const {svg,width,height}=makeSvg(container,360,680);const x0=margin.left,x1=width-margin.right,y0=height-margin.bottom,y1=margin.top;
    ticks(ymax-ymin,5).map(v=>v+ymin).forEach(t=>{const y=scale(t,ymin,ymax,y0,y1);svg.append(svgEl('line',{x1:x0,y1:y,x2:x1,y2:y,class:'gridline'}));const tx=svgEl('text',{x:x0-10,y:y+4,'text-anchor':'end'});tx.textContent=accuracy?`${Math.round(t*100)}%`:number(t,0);svg.append(tx);});
    const labelStep=Math.max(1,Math.ceil(labels.length/7));
    labels.forEach((l,i)=>{if(i%labelStep!==0&&i!==labels.length-1)return;const x=scale(i,0,Math.max(1,labels.length-1),x0,x1);const tx=svgEl('text',{x,y:y0+22,'text-anchor':'middle'});tx.textContent=l;svg.append(tx);});
    series.forEach((s,si)=>{const c=colors[modality(s.model)]||Object.values(colors)[si%5];const pts=s.points.map(p=>({x:scale(labels.indexOf(p.x),0,Math.max(1,labels.length-1),x0,x1),y:scale(+p.y,ymin,ymax,y0,y1),raw:p}));const path=svgEl('path',{d:pts.map((p,i)=>`${i?'L':'M'} ${p.x} ${p.y}`).join(' '),fill:'none',stroke:c,'stroke-width':2.5});svg.append(path);pts.forEach(p=>{const dot=svgEl('circle',{cx:p.x,cy:p.y,r:5,fill:c,stroke:'var(--surface)','stroke-width':2});dot.addEventListener('mousemove',e=>showTip(e,`<b>${pretty(s.model)}</b><br>${p.raw.x}: ${accuracy?percent(p.raw.y):number(p.raw.y)+' px'}${p.raw.count?`<br>n=${number(p.raw.count,0)}`:''}`));dot.addEventListener('mouseleave',hideTip);svg.append(dot);});const lx=pts.at(-1);if(lx){const tx=svgEl('text',{x:lx.x-4,y:lx.y-10,'text-anchor':'end',class:'series-label',fill:c});tx.textContent=pretty(s.model);svg.append(tx);}});
    const xt=svgEl('text',{x:(x0+x1)/2,y:height-9,'text-anchor':'middle',class:'axis-title'});xt.textContent=opts.xTitle||'Causal information available';svg.append(xt);const yt=svgEl('text',{x:16,y:(y0+y1)/2,transform:`rotate(-90 16 ${(y0+y1)/2})`,'text-anchor':'middle',class:'axis-title'});yt.textContent=accuracy?'Accuracy':'Pixel error (px)';svg.append(yt);
  }

  function rangeChart(id, rows) {
    const container=clear(id);if(!rows.length){container.innerHTML='<div class="callout">No rows available.</div>';return;} const lows=rows.map(r=>+r.paired_mean_gain_ci95_low), highs=rows.map(r=>+r.paired_mean_gain_ci95_high);const min=Math.min(0,...lows)-1,max=Math.max(0,...highs)+1;const rowH=42,margin={left:220,right:56,top:18,bottom:45};const {svg,width}=makeSvg(container,margin.top+rows.length*rowH+margin.bottom,700);const x0=margin.left,x1=width-margin.right;const zero=scale(0,min,max,x0,x1);svg.append(svgEl('line',{x1:zero,y1:margin.top,x2:zero,y2:margin.top+rows.length*rowH,stroke:'#ecf4ff','stroke-width':1.2,opacity:.7}));
    rows.forEach((r,i)=>{const y=margin.top+i*rowH+18;const lo=scale(+r.paired_mean_gain_ci95_low,min,max,x0,x1),hi=scale(+r.paired_mean_gain_ci95_high,min,max,x0,x1),v=scale(+r.paired_mean_gain_px,min,max,x0,x1);const tx=svgEl('text',{x:x0-12,y:y+4,'text-anchor':'end',class:'series-label'});tx.textContent=pretty(r.window);svg.append(tx);svg.append(svgEl('line',{x1:lo,y1:y,x2:hi,y2:y,stroke:'var(--muted)','stroke-width':3}));svg.append(svgEl('line',{x1:lo,y1:y-6,x2:lo,y2:y+6,stroke:'var(--muted)'}));svg.append(svgEl('line',{x1:hi,y1:y-6,x2:hi,y2:y+6,stroke:'var(--muted)'}));const d=svgEl('circle',{cx:v,cy:y,r:7,fill:(+r.paired_mean_gain_ci95_low>0)?colors.Fusion:colors.Other});d.addEventListener('mousemove',e=>showTip(e,`<b>${pretty(r.window)}</b><br>Mean gain ${number(r.paired_mean_gain_px,2)} px<br>95% CI [${number(r.paired_mean_gain_ci95_low,2)}, ${number(r.paired_mean_gain_ci95_high,2)}]<br>Improved ${percent(r.trajectory_fraction_improved)}`));d.addEventListener('mouseleave',hideTip);svg.append(d);});
    ticks(max-min,6).map(v=>v+min).forEach(t=>{const x=scale(t,min,max,x0,x1);const tx=svgEl('text',{x,y:margin.top+rows.length*rowH+20,'text-anchor':'middle'});tx.textContent=number(t,0);svg.append(tx);});const title=svgEl('text',{x:(x0+x1)/2,y:margin.top+rows.length*rowH+40,'text-anchor':'middle',class:'axis-title'});title.textContent='Paired mean gain from EMG (px; positive is better)';svg.append(title);
  }

  function scatter(id, rows) {
    const container=clear(id);if(!rows.length){container.innerHTML='<div class="callout">No directional results.</div>';return;}const extent=Math.max(20,...rows.flatMap(r=>[Math.abs(+r.mean_signed_x_error_px),Math.abs(+r.mean_signed_y_error_px)]))*1.25;const margin={left:66,right:28,top:25,bottom:55};const {svg,width,height}=makeSvg(container,390,620);const x0=margin.left,x1=width-margin.right,y0=height-margin.bottom,y1=margin.top;const zx=scale(0,-extent,extent,x0,x1),zy=scale(0,-extent,extent,y0,y1);svg.append(svgEl('line',{x1:zx,y1:y1,x2:zx,y2:y0,class:'axis'}));svg.append(svgEl('line',{x1:x0,y1:zy,x2:x1,y2:zy,class:'axis'}));
    rows.forEach(r=>{const x=scale(+r.mean_signed_x_error_px,-extent,extent,x0,x1),y=scale(+r.mean_signed_y_error_px,-extent,extent,y0,y1),c=colors[modality(r.model_kind)]||colors.Other,rad=5+Math.min(12,(+r.mean_inward_error_px||0)/30);const d=svgEl('circle',{cx:x,cy:y,r:rad,fill:c,opacity:.82,stroke:'var(--surface)','stroke-width':2});d.addEventListener('mousemove',e=>showTip(e,`<b>${pretty(r.model_kind)}</b><br>Signed X ${number(r.mean_signed_x_error_px)} px<br>Signed Y ${number(r.mean_signed_y_error_px)} px<br>Mean inward ${number(r.mean_inward_error_px)} px<br>Inward fraction ${percent(r.fraction_errors_inward)}`));d.addEventListener('mouseleave',hideTip);svg.append(d);const tx=svgEl('text',{x:x+rad+4,y:y+4,class:'series-label'});tx.textContent=pretty(r.model_kind);svg.append(tx);});
    const xt=svgEl('text',{x:(x0+x1)/2,y:height-10,'text-anchor':'middle',class:'axis-title'});xt.textContent='Mean signed X error (px)';svg.append(xt);const yt=svgEl('text',{x:15,y:(y0+y1)/2,transform:`rotate(-90 15 ${(y0+y1)/2})`,'text-anchor':'middle',class:'axis-title'});yt.textContent='Mean signed Y error (px)';svg.append(yt);
  }

  function renderKpis(){const full=D.summary.filter(r=>r.experiment==='full_trajectory'&&r.cutoff==='full'&&r.median!=null).sort((a,b)=>a.median-b.median)[0];const cont=D.summary.filter(r=>r.experiment==='continual_attention'&&r.cutoff==='touch'&&r.median!=null).sort((a,b)=>a.median-b.median)[0];const short=D.summary.find(r=>r.experiment==='hf_patchtst_exact'&&r.model==='hf_patchtst_imu'&&r.cutoff==='touch');const long=D.summary.find(r=>r.experiment==='hf_patchtst_long_context'&&r.model==='hf_patchtst_imu'&&r.cutoff==='touch');const temporal=D.temporal.filter(r=>r.experiment==='emg_temporal_touch'&&r.split==='test').sort((a,b)=>b.paired_mean_gain_px-a.paired_mean_gain_px)[0];const values=[['Best full-trajectory',`${number(full?.median)} px`,`${full?.configuration} · ${pretty(full?.model)} · ${number(full?.trials,0)} trials`],['Best continual touch',`${number(cont?.median)} px`,`${pretty(cont?.model)} · mix7 fold 0`],['Exact PatchTST IMU',`${number(short?.median)} px`,`256 samples · mean ${number(short?.mean)} px`],['Long PatchTST IMU',`${number(long?.median)} px`,`2,304 samples · mean ${number(long?.mean)} px`],['Best paired EMG gain',`+${number(temporal?.paired_mean_gain_px,2)} px`,`${pretty(temporal?.window)} · CI ${number(temporal?.paired_mean_gain_ci95_low,2)} to ${number(temporal?.paired_mean_gain_ci95_high,2)}`]];$('kpis').innerHTML=values.map(v=>`<div class="kpi"><div class="label">${v[0]}</div><div class="value">${v[1]}</div><div class="detail">${v[2]}</div></div>`).join('');}
  function renderArchitecture(){const allowed=['full_trajectory','grid_point','hybrid_point','continual_attention','hf_patchtst_exact','hf_patchtst_long_context'];const rows=D.summary.filter(r=>r.configuration==='mix7'&&r.median!=null&&allowed.includes(r.experiment)&&((r.experiment==='full_trajectory'&&r.cutoff==='full')||(r.experiment!=='full_trajectory'&&r.cutoff==='touch'))).sort((a,b)=>a.median-b.median);horizontalBars('architectureChart',rows,{metric:'median',label:r=>`${expLabel(r.experiment)} · ${pretty(r.model)}`,color:r=>colors[modality(r.model)],tip:r=>`<b>${expLabel(r.experiment)}</b><br>${pretty(r.model)} · ${r.configuration}<br>Median ${number(r.median)} px · mean ${number(r.mean)} px<br>n=${number(r.trials,0)}`});$('modalityLegend').innerHTML=Object.entries(colors).slice(0,4).map(([k,v])=>`<span><i class="dot" style="background:${v}"></i>${k}</span>`).join('');}
  function renderContext(){const rows=D.summary.filter(r=>r.configuration==='mix7'&&r.cutoff==='touch'&&['hf_patchtst_exact','hf_patchtst_long_context'].includes(r.experiment)&&r.median!=null).sort((a,b)=>a.median-b.median);horizontalBars('contextChart',rows,{metric:'median',label:r=>`${r.experiment==='hf_patchtst_exact'?'256':'2,304'} · ${modality(r.model)}`,color:r=>colors[modality(r.model)]});}
  function initHistories(){const values=D.histories.map((h,i)=>({value:String(i),label:`${h.experiment==='hf_patchtst_exact'?'256':'2,304'} · ${pretty(h.model)}${h.complete?'':' · interrupted'}`}));const s=$('historySelect');s.innerHTML=values.map(v=>`<option value="${v.value}">${v.label}</option>`).join('');const pref=D.histories.findIndex(h=>h.experiment==='hf_patchtst_long_context'&&h.model==='hf_patchtst_fusion');s.value=String(pref>=0?pref:0);const draw=()=>{const h=D.histories[+s.value];lineChart('historyChart',[{model:h.model,points:h.points.map(p=>({x:String(number(p.epoch,0)),y:p.selection}))}],{labels:h.points.map(p=>String(number(p.epoch,0))),xTitle:'Epoch'});};s.addEventListener('change',draw);draw();}
  function initConfig(){const exps=uniq(D.summary.map(r=>r.experiment)).sort();setOptions($('configExperiment'),exps,expLabel,'full_trajectory');const draw=()=>{const e=$('configExperiment').value;const subset=D.summary.filter(r=>r.experiment===e);const models=uniq(subset.map(r=>r.model)).sort();setOptions($('configModel'),models,pretty,$('configModel').value||(/full/.test(e)?'imu_patch':models[0]));const m=$('configModel').value;const cutoffs=uniq(subset.filter(r=>r.model===m).map(r=>r.cutoff)).sort((a,b)=>cutoffOrder(a)-cutoffOrder(b));setOptions($('configCutoff'),cutoffs,pretty,$('configCutoff').value||(cutoffs.includes('full')?'full':cutoffs.includes('touch')?'touch':cutoffs[0]));const rows=subset.filter(r=>r.model===m&&r.cutoff===$('configCutoff').value&&r[$('configMetric').value]!=null).sort((a,b)=>+a[$('configMetric').value]-+b[$('configMetric').value]);horizontalBars('configurationChart',rows,{metric:$('configMetric').value,label:r=>`${r.configuration} · n=${number(r.trials,0)}`,color:r=>colors[modality(r.model)],tip:r=>`<b>${r.configuration}</b> · ${pretty(r.model)}<br>Median ${number(r.median)} px · mean ${number(r.mean)} px · P90 ${number(r.p90)} px<br>≤100 px ${percent(r.within100)} · target box ${percent(r.targetBox)}<br>Held out n=${number(r.trials,0)}`});};['configExperiment','configModel','configCutoff','configMetric'].forEach(id=>$(id).addEventListener('change',draw));draw();}
  function initContinual(){const eligible=uniq(D.summary.filter(r=>['0.0s','0.2s','0.4s','touch'].includes(r.cutoff)).map(r=>r.experiment)).sort();setOptions($('continualExperiment'),eligible,expLabel,'continual_attention');const draw=()=>{const e=$('continualExperiment').value,metric=$('continualMetric').value,rows=D.summary.filter(r=>r.experiment===e&&['0.0s','0.2s','0.4s','touch'].includes(r.cutoff)&&r[metric]!=null);const models=uniq(rows.map(r=>r.model));const labels=['0.0s','0.2s','0.4s','touch'];const series=models.map(model=>({model,points:labels.map(x=>rows.find(r=>r.model===model&&r.cutoff===x)).filter(Boolean).map(r=>({x:r.cutoff,y:r[metric],count:r.trials}))}));lineChart('continualChart',series,{labels,accuracy:/within|target/.test(metric)});};$('continualExperiment').addEventListener('change',draw);$('continualMetric').addEventListener('change',draw);draw();
    const pairExps=uniq(D.fusionPairs.map(r=>r.experiment)).sort();setOptions($('pairExperiment'),pairExps,expLabel,'continual_attention');const drawPair=()=>{const rows=D.fusionPairs.filter(r=>r.experiment===$('pairExperiment').value).sort((a,b)=>cutoffOrder(a.requested_cutoff)-cutoffOrder(b.requested_cutoff));const adapted=rows.map(r=>({...r,window:r.requested_cutoff}));rangeChart('pairChart',adapted);};$('pairExperiment').addEventListener('change',drawPair);drawPair();}
  function initTemporal(){const exps=uniq(D.temporal.map(r=>r.experiment)).sort();setOptions($('temporalExperiment'),exps,expLabel,'emg_temporal_touch');const draw=()=>{const rows=D.temporal.filter(r=>r.experiment===$('temporalExperiment').value&&String(r.split).toLowerCase()==='test').sort((a,b)=>b.paired_mean_gain_px-a.paired_mean_gain_px);rangeChart('temporalGainChart',rows);const bars=[];rows.forEach(r=>{bars.push({label:`${pretty(r.window)} · fusion`,model:'fusion',median:r.fusion_median_pixel_error,window:r.window});bars.push({label:`${pretty(r.window)} · EMG only`,model:'emg',median:r.emg_median_pixel_error,window:r.window});});horizontalBars('temporalErrorChart',bars,{metric:'median',label:r=>r.label,color:r=>colors[modality(r.model)]});};$('temporalExperiment').addEventListener('change',draw);draw();}
  function initDiagnostics(){const exps=uniq(D.directional.map(r=>r.experiment)).sort();setOptions($('directionExperiment'),exps,expLabel,'continual_attention');const draw=()=>{const e=$('directionExperiment').value,rows=D.directional.filter(r=>r.experiment===e);const cuts=uniq(rows.map(r=>String(r.requested_cutoff))).sort((a,b)=>cutoffOrder(a)-cutoffOrder(b));setOptions($('directionCutoff'),cuts,pretty,$('directionCutoff').value||(cuts.includes('touch')?'touch':cuts[0]));scatter('directionChart',rows.filter(r=>String(r.requested_cutoff)===$('directionCutoff').value));};$('directionExperiment').addEventListener('change',draw);$('directionCutoff').addEventListener('change',draw);draw();
    const models=uniq(D.attention.map(r=>r.model_kind)).sort();setOptions($('attentionModel'),models,pretty,models.find(m=>/fusion/.test(m))||models[0]);const drawAtt=()=>{const m=$('attentionModel').value,rows=D.attention.filter(r=>r.model_kind===m);const cuts=uniq(rows.map(r=>String(r.requested_cutoff))).sort((a,b)=>cutoffOrder(a)-cutoffOrder(b));setOptions($('attentionCutoff'),cuts,pretty,$('attentionCutoff').value||(cuts.includes('touch')?'touch':cuts[0]));const selected=rows.filter(r=>String(r.requested_cutoff)===$('attentionCutoff').value).sort((a,b)=>b.mean_attention-a.mean_attention).slice(0,18).map(r=>({...r,label:`${pretty(r.attention_type)} · ${r.channel}`}));horizontalBars('attentionChart',selected,{metric:'mean_attention',label:r=>r.label,color:r=>/emg/i.test(r.attention_type)?colors.EMG:colors.IMU,tip:r=>`<b>${r.channel}</b> · ${pretty(r.attention_type)}<br>Mean attention ${number(r.mean_attention,4)}<br>Top-channel fraction ${percent(r.top_attention_fraction)}`});};$('attentionModel').addEventListener('change',drawAtt);$('attentionCutoff').addEventListener('change',drawAtt);drawAtt();}
  function initTable(){const exps=uniq(D.summary.map(r=>r.experiment)).sort();setOptions($('tableExperiment'),exps,expLabel,null,true);const cuts=uniq(D.summary.map(r=>r.cutoff)).sort((a,b)=>cutoffOrder(a)-cutoffOrder(b));setOptions($('tableCutoff'),cuts,pretty,null,true);let sortKey='median',ascending=true;const draw=()=>{const q=$('resultSearch').value.toLowerCase(),e=$('tableExperiment').value,c=$('tableCutoff').value;const rows=D.summary.filter(r=>(e==='all'||r.experiment===e)&&(c==='all'||r.cutoff===c)&&(!q||`${r.experiment} ${r.model} ${r.configuration} ${r.source}`.toLowerCase().includes(q))).sort((a,b)=>{const av=a[sortKey],bv=b[sortKey];if(av==null)return 1;if(bv==null)return -1;return (typeof av==='number'?av-bv:String(av).localeCompare(String(bv)))*(ascending?1:-1);});$('resultsBody').innerHTML=rows.map(r=>`<tr><td>${expLabel(r.experiment)}</td><td>${pretty(r.model)}</td><td>${r.configuration}</td><td>${pretty(r.cutoff)}</td><td class="num">${number(r.trials,0)}</td><td class="num">${number(r.median)}</td><td class="num">${number(r.mean)}</td><td class="num">${number(r.p90)}</td><td class="num">${percent(r.within100)}</td><td class="num">${percent(r.targetBox)}</td><td class="source">${r.source}</td></tr>`).join('');};['resultSearch','tableExperiment','tableCutoff'].forEach(id=>$(id).addEventListener(id==='resultSearch'?'input':'change',draw));document.querySelectorAll('th[data-key]').forEach(th=>th.addEventListener('click',()=>{if(sortKey===th.dataset.key)ascending=!ascending;else{sortKey=th.dataset.key;ascending=true;}draw();}));draw();}
  function renderStatus(){const rows=D.statuses.map(s=>{const stopped=s.status==='stopped_by_user'||s.status==='completed_with_failures';const done=s.completedModels?.length||0,partial=s.partialModels?.length||0;return `<div class="status-row"><strong>${expLabel(s.experiment)}</strong><span class="badge ${stopped?'warn':''}">${pretty(s.status)}</span><div class="status-detail">${done} finalized model outputs${partial?` · ${partial} partial`:''}${s.finished?` · ended ${s.finished}`:''}<div class="source">${s.source}</div>${s.failures?.length?`<div style="color:var(--orange);margin-top:5px">${s.failures.map(f=>f.reason||f.label).join(' · ')}</div>`:''}</div></div>`;});$('statusList').innerHTML=rows.join('');}
  renderKpis();renderArchitecture();renderContext();initHistories();initConfig();initContinual();initTemporal();initDiagnostics();initTable();renderStatus();
  $('generatedAt').textContent=`Generated ${new Date(D.generatedAt).toLocaleString()} · ${D.root}`;
  $('inventory').textContent=`${D.summary.length} summary rows · ${D.inventory.testMetricFiles} test metric files · ${D.inventory.predictionFiles} prediction files · ${D.inventory.historyFiles} histories`;
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{renderArchitecture();renderContext();$('configMetric').dispatchEvent(new Event('change'));$('continualMetric').dispatchEvent(new Event('change'));$('temporalExperiment').dispatchEvent(new Event('change'));$('directionCutoff').dispatchEvent(new Event('change'));$('attentionCutoff').dispatchEvent(new Event('change'));$('historySelect').dispatchEvent(new Event('change'));},180);});
})();
</script>
</body>
</html>
'''


def main() -> None:
    payload = json.dumps(build_payload(), separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
