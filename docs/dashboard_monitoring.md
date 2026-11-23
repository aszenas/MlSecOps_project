# Streamlit monitoring dashboard and API-log monitoring

This guide explains how to start the read-only dashboard, monitor actual
API prediction events and reproduce the deterministic persisted-event demo.

The API-log monitoring command reads events written by actual API requests.

The separate deterministic demo writes synthetic API-shaped events directly
to isolated SQLite databases using `save_prediction_event`.
That demo is a persisted-event replay, not an API request/response test.

For the monitoring data contract, drift thresholds and versioned reference
details, see [Monitoring](monitoring.md).

## Prepare and start locally

The repository and CI target Python 3.11.
The project dependency set includes Streamlit 1.67.0,
Evidently 0.7.23 and MLflow 3.17.2.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Set the source import path and an absolute monitoring report directory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:MONITORING_REPORT_DIR = Join-Path (Get-Location).Path "reports\monitoring"
```

The report directory does not need to contain a monitoring run yet.
Without a state file, the dashboard displays `no_run`.

To enable the champion and training-selection panels, configure MLflow
before starting Streamlit:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_dashboard.py
```

Open http://localhost:8501.
Stop the process with `Ctrl+C` in the dashboard terminal.

`MLFLOW_TRACKING_URI` is optional for the persisted monitoring report itself.
Without it, the champion and training-selection panels show that the Registry
is not configured; persisted monitoring results remain usable.

The dashboard only reads monitoring files and MLflow metadata.
Its refresh button rereads the current sources.
It does not start monitoring, training, retraining, model downloads
or Registry changes.

## Dashboard states

The monitoring loader exposes four states:

| State | Meaning |
| --- | --- |
| `no_run` | `monitoring_state.json` does not exist yet |
| `insufficient_data` | Too few valid feature rows exist in the selected window; the drift decision and report paths are null |
| `ready` | The validated state points to a complete current report package and contains a boolean drift decision |
| `unavailable` | State JSON or a referenced report is missing, invalid, inconsistent or unreadable |

An unavailable result is never displayed as drift-free.

The sample count is the number of valid feature rows after event and
feature validation.

The default minimum is 100.
Exactly 100 valid rows is sufficient for report generation.

The UTC-normalized `[start, end)` window includes `start` and excludes `end`.

In `ready`:

- `drift_decision: false` means no drift was detected.
- `drift_decision: true` means drift was detected.

In `insufficient_data`, the drift decision remains unknown (`null`).

The runtime section displays:

- The monitoring window.
- The report generation time.
- Valid current samples and the minimum sample count.
- Skipped invalid event and feature counts.
- Missing samples when the result is `insufficient_data`.

The reference expander displays the DVC and split identifiers stored
in the monitoring state.

In `ready`, the current Evidently HTML report is embedded with JavaScript
support and can be downloaded using the dashboard button.

Only the report paths referenced by the current state are read.

The loader rejects inconsistent drift decisions when
`drift_decision` differs from the summary's `threshold_reached` value.

### Operational panels

The champion panel reads the current `champion` alias of
`predictive-maintenance-model`.
It does not load the model artifact.

The training-selection panel reads the latest valid audit from the
`predictive-maintenance-baselines` MLflow experiment.

The repository does not yet persist a separate automatic-retraining marker,
so this panel does not prove that an automatic retraining job occurred.

Missing configuration, no audit data and an unavailable backend
are displayed as non-success states.

The operational panels and monitoring report are loaded independently.
A Registry failure does not invalidate an otherwise readable monitoring report.

## Monitoring actual API prediction logs in Compose

Docker Compose starts `app/streamlit_dashboard.py`.

The dashboard reads `/app/reports/monitoring` from the shared report volume
and queries the internal MLflow backend at `http://mlflow:5000`.

Monitoring is started explicitly from the API container:

```powershell
docker compose exec -T api python /app/scripts/run_api_monitoring.py --since "2026-11-23T00:00:00Z" --until "2026-11-24T00:00:00Z"
```

Replace the example timestamps with the desired window.
Both timestamps must include a timezone.
The start is inclusive and the end is exclusive.

The command uses the existing monitoring pipeline.
It reads successful prediction events, validates their features,
verifies the DVC reference dataset and publishes the monitoring state.

It reads the SQLite database directly.
MLflow prediction export is not a prerequisite.

### Configuration

| Setting | Purpose |
| --- | --- |
| `TRAINING_REPOSITORY_ROOT` | Repository containing the verified raw DVC dataset |
| `PREDICTION_LOG_DB` | SQLite prediction database written by the API |
| `REPORTS_DIR` | Report storage root; monitoring uses its `monitoring` subdirectory |
| `MONITORING_REPORT_DIR` | Optional direct override for the monitoring output directory |

Without environment overrides, the command uses:

- Repository root: the repository containing the script.
- Database: `artifacts/predictions.db` under that repository.
- Reports: `reports/monitoring` under that repository.

An explicit `MONITORING_REPORT_DIR` takes precedence over `REPORTS_DIR`.

The command also accepts:

| Option | Purpose |
| --- | --- |
| `--repository-root PATH` | Override the reference-data repository |
| `--database PATH` | Override the prediction database |
| `--output-dir PATH` | Override the monitoring output directory |
| `--since TIMESTAMP` | Inclusive window start, with timezone |
| `--until TIMESTAMP` | Exclusive window end, with timezone |
| `--minimum-samples INTEGER` | Positive sample threshold; default 100 |

Explicit command-line paths override the corresponding environment defaults.

Compose already supplies the API database, repository and report-root settings.
The dashboard mounts the shared report volume read-only.

### Running the command locally

Use the same prediction database as the locally running API:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe .\scripts\run_api_monitoring.py `
  --since "2026-11-23T00:00:00Z" `
  --until "2026-11-24T00:00:00Z"
```

Replace the timestamps with the desired window.

A local default database is separate from the Compose named volume.
Use the command inside the API container to inspect Compose events.

### Results and errors

- `insufficient_data`: monitoring succeeded, but there are too few valid
  samples. The drift decision and current report paths remain null.
- `ready`: enough samples exist and a complete report package has
  been published. The drift decision is a boolean.
- Invalid arguments return exit code 2 without starting monitoring.
- Handled operational failures print an error to stderr and return exit code 1.
- Both successful monitoring states return exit code 0.

A missing or unreadable prediction database is an operational error.
It is not treated as a successful empty monitoring window.

After a successful run, click **Refresh monitoring data** in the dashboard.
The refresh button reads the result; it does not start another monitoring run.

## Verify the actual API-to-dashboard flow

A champion must be available before successful prediction requests can be made.

In PowerShell, record a window and submit one real prediction:

```powershell
$monitoringSince = [DateTimeOffset]::UtcNow.ToString("o")

$predictionBody = @{
    type = "L"
    air_temperature_k = 300.8
    process_temperature_k = 310.3
    rotational_speed_rpm = 1538
    torque_nm = 36.1
    tool_wear_min = 198
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $predictionBody `
    -TimeoutSec 60 `
    -ErrorAction Stop

$monitoringUntil = [DateTimeOffset]::UtcNow.ToString("o")

docker compose exec -T api python /app/scripts/run_api_monitoring.py --since $monitoringSince --until $monitoringUntil
```

With one valid event and the default threshold, the expected result is:

- `status: insufficient_data`
- `valid_current_count: 1`
- `missing_current_count: 99`
- `drift_decision: null`
- Null report paths

Refresh the dashboard and check the warning and sample counts.

For a separate report-generation window, submit 100 requests:

```powershell
$monitoringSince = [DateTimeOffset]::UtcNow.ToString("o")

foreach ($i in 1..100) {
    Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/predict" `
        -Method Post `
        -ContentType "application/json" `
        -Body $predictionBody `
        -TimeoutSec 60 `
        -ErrorAction Stop | Out-Null

    Write-Host "Prediction $i / 100 completed"
}

$monitoringUntil = [DateTimeOffset]::UtcNow.ToString("o")

docker compose exec -T api python /app/scripts/run_api_monitoring.py --since $monitoringSince --until $monitoringUntil
```

Do not continue to monitoring if any request fails.
These exact sample counts assume no other requests arrive in the same window.

Expected result:

- `status: ready`
- `valid_current_count: 100`
- `missing_current_count: 0`
- A boolean `drift_decision`
- Paths for the HTML, JSON and summary reports

Refresh the dashboard.
The current report, drift status and HTML download button should appear.

### Observed manual verification

The API-to-dashboard flow was exercised in Docker Compose:

1. One real prediction produced `insufficient_data`, with one valid sample
   and 99 missing samples.
2. The dashboard displayed the matching warning and counts.
3. A separate window of 100 successful API requests produced `ready`.
4. No invalid event or feature rows were skipped in that window.
5. The dashboard displayed detected drift and the embedded Evidently report.
6. The HTML report download button was visible.

The 100-request check repeated the same input as synthetic test traffic.
Its detected drift verifies the integration and does not establish
drift in representative production traffic.

## Reproduce the deterministic persisted-event demo

This separate demo writes synthetic API-shaped events directly to isolated
SQLite databases. It does not call the API.

The reference loader verifies the checked-out DVC dataset before deriving
the initial 3,000-row training reference.

Ensure the dataset exists and is clean:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -m dvc status data/raw.dvc --json
```

If the CSV is absent, restore it using the configured DVC workflow:

```powershell
.\.venv\Scripts\python.exe -m dvc pull data/raw.dvc -r dagshub
```

Configure credentials using the
[README setup instructions](../README.md#local-setup-powershell).

Choose a new temporary output directory.
The script refuses to reuse a non-empty directory:

```powershell
$demoRoot = Join-Path $env:TEMP ("mlsecops-monitoring-demo-" + [guid]::NewGuid())

.\.venv\Scripts\python.exe .\scripts\run_persisted_monitoring_demo.py `
  --repository-root (Get-Location).Path `
  --output-root $demoRoot
```

The script creates three isolated scenarios:

```text
$demoRoot\normal\predictions.sqlite
$demoRoot\normal\reports\monitoring_state.json
$demoRoot\shifted\predictions.sqlite
$demoRoot\shifted\reports\monitoring_state.json
$demoRoot\insufficient_data\predictions.sqlite
$demoRoot\insufficient_data\reports\monitoring_state.json
```

Each event is written with `save_prediction_event`.
The normal and shifted scenarios each contain 3,000 valid rows.

The shifted sample uses the deterministic simulation:

- Air and process temperature increase by 10 K.
- Rotational speed is multiplied by 1.4.
- Torque increases by 20 Nm.
- Product type and tool wear remain unchanged.

Expected results:

| Scenario | Expected state |
| --- | --- |
| `normal` | `ready`, 3,000 valid rows, no detected drift |
| `shifted` | `ready`, 3,000 valid rows, detected drift in four numeric fields |
| `insufficient_data` | One valid row, null drift decision and report paths |

Inspect the generated JSON output for the exact paths and counts.

To view one scenario locally:

```powershell
$env:MONITORING_REPORT_DIR = (Resolve-Path "$demoRoot\normal\reports").Path
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_dashboard.py
```

Stop it with `Ctrl+C`, then repeat with `shifted\reports`
or `insufficient_data\reports`.

Stop the Compose dashboard first if it already uses port 8501,
or select another port for the local Streamlit process.

No generated database, report or token belongs in Git.
The repository ignores `artifacts/` and `reports/`.
A path under `$env:TEMP` is outside the repository.

## Report publication and reruns

Every successful report-generating run writes:

- `report.html`
- `report.json`
- `summary.json`

The files are stored under a new directory:

```text
<monitoring-output-directory>/runs/<run-id>/
```

The root `monitoring_state.json` is atomically replaced only after
the complete new package has been written.

Earlier published run directories remain on disk,
but they are not current unless the state file points to them.

An `insufficient_data` state keeps previous run directories
but sets all current report paths to null.

The dashboard does not display an earlier report as the result of a newer
insufficient-data window.

If new report publication fails before the state is switched,
the previously published state remains current.

No report-retention or automatic cleanup policy is configured.

## Focused tests

Run command and dashboard-loader tests:

```powershell
python -m pytest tests/unit/test_run_api_monitoring.py tests/unit/test_dashboard_data.py -v
```

Run the operational loader and Streamlit rendering tests:

```powershell
python -m pytest tests/unit/test_dashboard_operational_data.py tests/unit/test_streamlit_dashboard.py -v
```

The command tests replace the monitoring function.
They verify configuration, argument handling, output and exit codes
without generating real reports.

Dashboard-loader tests cover missing files, invalid data, report publication
and inconsistent drift decisions.

The manual Compose verification exercises actual prediction requests,
persisted events, monitoring output and dashboard rendering.

## Limitations

- Monitoring is started manually; no scheduler is installed.
- Drift does not yet trigger automatic training.
- Prediction events contain features and model outputs, not verified target labels.
  They must not be treated directly as labeled retraining data.
- The reference loader requires the versioned raw dataset and a successful
  DVC status check.
- The deterministic demo directly replays persisted events and is not
  an API request/response test.
- The champion and training-selection panels require an explicitly configured
  MLflow backend.
- The current MLflow experiment does not distinguish an automatic retraining
  job from another training-selection run.
- No dashboard action starts monitoring, training, retraining, DVC pull
  or Registry changes.
