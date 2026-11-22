# Streamlit monitoring dashboard and persisted-prediction demo

This guide explains how to start the read-only dashboard and reproduce the
monitoring result from synthetic, API-shaped prediction events. The demo writes
directly to an isolated SQLite event store with the existing
`save_prediction_event` function; it is a deterministic persisted-event
replay, not an API endpoint-to-endpoint test.

For the monitoring data contract, validation rules, drift thresholds and
versioned reference details, see [Monitoring](monitoring.md). This guide adds
the dashboard workflow and the persisted-event demonstration without
duplicating those rules.

## Prepare and start

The repository and CI currently target Python 3.11. The local project
environment used for the dashboard is `.venv`; the checked dependency set
includes Streamlit 1.67.0, Evidently 0.7.23 and MLflow 3.17.2.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Set the source import path and the monitoring report directory before starting
the app. Use an absolute path so the dashboard does not depend on the current
working directory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:MONITORING_REPORT_DIR = (Resolve-Path .\artifacts\monitoring-demo\normal\reports).Path
```

Start the dashboard from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_dashboard.py
```

Open [http://localhost:8501](http://localhost:8501). Stop the process with
`Ctrl+C` in the dashboard terminal.

`MLFLOW_TRACKING_URI` is optional for the persisted monitoring report itself.
Without it, the champion and training-selection panels show that the Registry
data is not configured; the monitoring state and Evidently report remain
usable. To enable those panels, set the URI explicitly, for example:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

The dashboard only reads the monitoring files and MLflow metadata. The refresh
button rereads the current sources; it does not run monitoring, training,
retraining, model download or Registry mutation.

## Dashboard states

The monitoring loader exposes four states:

| State | Meaning |
| --- | --- |
| `no_run` | `monitoring_state.json` does not exist yet. |
| `insufficient_data` | The current window has fewer valid feature rows than the configured minimum. Drift is unknown, so the drift decision and report paths are null. |
| `ready` | The validated state points to a complete current report package and contains a boolean drift decision. |
| `unavailable` | State JSON or a referenced report is missing, invalid, inconsistent or temporarily unreadable. It is never shown as drift-free. |

The sample count is the number of valid feature rows after event and
feature validation. The configured default minimum is 100; exactly 100 valid
rows is `ready`. The UTC `[start, end)` window includes `start` and excludes
`end`. In `ready`, `False` means no detected drift and `True` means drift was
detected. In `insufficient_data`, the decision remains unknown (`null`).

The runtime section shows the window, generation time, valid current rows,
minimum, skipped event records and skipped feature rows. The reference
expander shows the DVC and split identifiers stored in the monitoring state.
In `ready`, the current HTML report is embedded with JavaScript support and
can be downloaded with the dashboard button. Only the report paths referenced
by the current state are read.

The champion panel reads the current `champion` alias of
`predictive-maintenance-model`; it does not load the model artifact. The other
panel reads the latest valid training-selection audit from the
`predictive-maintenance-baselines` MLflow experiment. The repository does not
persist a separate retraining marker, so that panel does not prove that Máté's
automation ran. Missing configuration, no audit data and an unavailable
backend are shown as non-success states.

## Reproduce the drift demonstration

The reference loader verifies the checked-out DVC dataset before deriving the
initial 3,000-row training reference. Ensure the dataset exists and is clean:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -m dvc status data/raw.dvc --json
```

If the CSV is absent, restore it using the repository's configured DVC
workflow (`dvc pull -r dagshub`) and local credentials described in the
[README setup section](../README.md#local-setup-powershell). Do not put a token
in this guide or in a command history shared with others.

Choose a new temporary output directory. The script refuses to reuse a
non-empty directory, which prevents old SQLite events or report state from
mixing with the demonstration:

```powershell
$demoRoot = Join-Path $env:TEMP ("mlsecops-monitoring-demo-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe .\scripts\run_persisted_monitoring_demo.py `
  --repository-root (Get-Location).Path `
  --output-root $demoRoot
```

The script uses the existing DVC-verified reference and creates three isolated
scenario directories:

```text
$demoRoot\normal\predictions.sqlite
$demoRoot\normal\reports\monitoring_state.json
$demoRoot\shifted\predictions.sqlite
$demoRoot\shifted\reports\monitoring_state.json
$demoRoot\insufficient_data\predictions.sqlite
$demoRoot\insufficient_data\reports\monitoring_state.json
```

Each event is written with `save_prediction_event`; the script does not call
the API. The normal and shifted scenarios each contain 3,000 valid rows. The
shifted sample uses the existing deterministic simulation: air and process
temperature increase by 10 K, rotational speed is multiplied by 1.4, and
torque increases by 20 Nm. Product type and tool wear remain unchanged.

The expected output is:

| Scenario | Expected state |
| --- | --- |
| `normal` | `ready`, 3,000 valid rows, `drift_decision: false`, no drifted fields |
| `shifted` | `ready`, 3,000 valid rows, `drift_decision: true`, four drifted numeric fields |
| `insufficient_data` | `insufficient_data`, one valid row, null drift decision and null report paths |

Inspect the generated JSON output for the exact paths and counts. To view one
scenario, point the dashboard at its report directory and restart Streamlit:

```powershell
$env:MONITORING_REPORT_DIR = (Resolve-Path "$demoRoot\normal\reports").Path
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_dashboard.py
```

Stop it with `Ctrl+C`, then repeat with `shifted\reports` or
`insufficient_data\reports`. No generated database, report or token belongs in
Git; the repository ignores `artifacts/` and `reports/`, while a path under
`$env:TEMP` is outside the repository entirely.

## Report publication and reruns

Every successful monitoring run writes `report.html`, Evidently JSON and
`summary.json` under a new `reports\monitoring\runs\<run-id>\` directory. The
root `monitoring_state.json` is atomically replaced only after the complete
new package has been written. Earlier published run directories remain on
disk, but they are not current unless the state file points to them. An
`insufficient_data` state keeps prior run directories but sets all current
report paths to `null`.

## Limitations

- The demonstration directly replays synthetic persisted events and is not an
  API request/response test.
- The DVC reference requires the local versioned dataset and a successful DVC
  status check; no remote or token is embedded in the script.
- The champion and training-selection panels require an explicitly configured
  MLflow Registry. The current MLflow experiment does not distinguish a
  retraining job from another training-selection run.
- No dashboard action starts monitoring, training, retraining, DVC pull or
  Registry changes.
