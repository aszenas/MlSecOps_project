# Automated monitoring and retraining

## Flow

The Compose `monitoring` service runs sequential cycles:

1. Read successful API prediction events in the latest monitoring window.
2. Verify the reference data and publish monitoring state and reports.
3. Export pending prediction events to MLflow using existing incremental bookkeeping.
4. If enabled, evaluate drift and verify the labeled v2 training data.
5. Atomically reserve the dataset/backend/model input in SQLite.
6. Call the existing `train_from_dvc(..., dataset_version="v2")` pipeline.
7. Persist the candidate and selection result and identify the automatic run.

Monitoring uses the selected time window. Export handles all pending events,
including older events outside that window. It never deletes source events.
Prediction outputs are not used as target labels.

## Configuration

Set these values in the Git-ignored `.env` file when overrides are needed:

```dotenv
MONITORING_INTERVAL_SECONDS=300
MONITORING_WINDOW_SECONDS=3600
MONITORING_MINIMUM_SAMPLES=100
AUTOMATIC_RETRAINING_ENABLED=false
```

The worker runs immediately, then waits the interval after each cycle ends.
Cycles do not overlap within one worker. The window is a rolling UTC interval;
monitoring windows can overlap.

Retraining defaults to disabled. Set `AUTOMATIC_RETRAINING_ENABLED=true`
when automatic training is desired, then recreate the worker:

```powershell
docker compose up -d --no-build monitoring
```

The worker shares the verified training checkout, prediction database,
MLflow backend and report storage with the existing services.
To refresh committed source and DVC inputs, stop the stack and rerun its
preparation service using the Docker guide. The worker does not fetch a new
Git checkout or download a new data version during each cycle.

## Controlled one-shot verification

Start API, MLflow and the dashboard without the background worker:

```powershell
docker compose down
docker compose build api
docker compose up -d --no-build api streamlit
```

After sufficient prediction traffic exists, run one enabled cycle:

```powershell
docker compose run --rm --no-deps -e AUTOMATIC_RETRAINING_ENABLED=true monitoring python /workspace/repository/scripts/run_monitoring_worker.py --once
```

The one-shot container uses the same persistent volumes and exits afterwards.
Repeat the command with the same training inputs to verify `duplicate_input`.
It must not create another training candidate.

Use `/predict` afterwards and compare its `model_version` with the selected
champion in the training result. The dashboard's refresh button should show
the automatic trigger and attempt ID for a successfully marked run.

The actual Docker result is recorded in the
[verification record](verification_record.md#automated-retraining-verification).

## Results

The cycle includes `monitoring`, `export` and `retraining` outcomes.
A handled export, training or audit failure produces `partial_failure`;
an otherwise successful cycle produces `completed`.

| Retraining status | Meaning |
| --- | --- |
| `disabled` | The automatic training switch is off |
| `insufficient_data` | Monitoring has too few valid samples |
| `no_drift` | The published monitoring decision is false |
| `waiting_for_labeled_data` | The v2 CSV or its DVC pointer is absent |
| `duplicate_input` | This dataset/backend/model combination was already reserved |
| `completed` | Training and selection completed; promotion is not guaranteed |
| `failed` | Verification or training encountered a handled error |

Missing files cause waiting; invalid labels, dirty Git state or mismatched
DVC data cause an error. The training pipeline also checks the expected v2
split and preserves the validation and final-test partitions.

If monitoring itself fails, no new cycle result is published by the wrapper;
the worker reports the error and retries after the configured interval.

## Persistent attempts and recovery

`retraining_attempts` is stored in the same persistent SQLite database as
prediction events. No extra volume is required.

The unique input combines the MLflow URI string, model type, raw DVC hash
and v2 training-file DVC hash. Git commits and rolling monitoring windows are
recorded as context but do not create a new training input by themselves.
A finished attempt remains blocked even when a new window reports drift.

Reservations are committed before training starts. A failed or interrupted
attempt is not automatically retried: first reconcile its SQLite record with
MLflow runs and the champion alias. A process killed during training can leave
an attempt marked `running`; this is not proof that training is still active.
Do not delete the prediction database to clear an attempt. This implementation
does not provide an automatic reconciliation or manual-reset command.

SQLite reservation and MLflow model operations are not one atomic transaction.
The reservation prevents blind duplicate launches; it does not claim distributed
exactly-once execution. Run one worker and do not launch concurrent manual
promotion jobs against the same Registry.

## Model selection and audit

The existing model-selection policy controls promotion. An equal F1,
insufficient recall or another rejected candidate preserves the selected
champion. Prediction resolves the champion on each request.

Completed automatic runs receive these tags:

- `operation=automated_retraining`
- `retraining_attempt_id`
- `retraining_dataset_version=v2`
- `retraining_window_start`
- `retraining_window_end`

The operation marker is written after the context tags. Existing dataset
lineage tags link the run to the verified Git and DVC inputs.
An audit-tagging error is recorded separately from a successful training result;
it does not retroactively mean that no model was trained or promoted.

The dashboard distinguishes automatic and legacy/manual selection audits.
Its training panel shows the latest valid selection audit, not live worker,
waiting, interruption or failure state. Those remain in worker logs and SQLite.

## Verification scope

Unit tests cover decision rules, data validation, persistent reservations,
concurrent claims, worker retries and shutdown, integration switches and audit
failures. The Registry integration tests use actual API events, Evidently
reports, export artifacts and model aliases with controlled training/reference
adapters. Real-data Docker execution is recorded separately.

The Docker run observed successful training with rejection of an equal-F1
candidate. Promotion and training-failure serving behavior are covered by the
isolated Registry tests; do not describe them as observed Docker outcomes.
