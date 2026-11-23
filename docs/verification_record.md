# Data restoration and monitoring verification record

This is a concise evidence record for the versioned submission export and the
Streamlit–Evidently monitoring demonstration. It points to the detailed
procedures in the [dataset release guide](dataset_release.md) and the
[monitoring guide](monitoring.md) instead of duplicating them.

## Submission restoration

The checked Git revision was
`52f7c549b26ba27e1d8c56a726d830d996009d54`. The submission pointer was
[`data/processed/submission.dvc`](../data/processed/submission.dvc) with DVC
directory hash `737d57f421900bc799f3a3997e3a4f06.dir`. The raw pointer used by
the export was [`data/raw.dvc`](../data/raw.dvc), hash
`b740c7a32109e28365b89752a09e6096.dir`.

The clean-clone check used a separate clone-local, initially empty DVC cache
and the configured `dagshub` remote. The targeted restore command was:

```powershell
.\.venv\Scripts\python.exe -m dvc pull data/processed/submission.dvc -r dagshub
```

The restored files were checked with targeted DVC status and Git status. Both
were clean. The restored CSV had 5,000 rows with these partition counts:

| Partition | Rows |
| --- | ---: |
| train | 3,000 |
| validation | 750 |
| future | 750 |
| test | 500 |

The recorded byte-level identifiers were:

| File | SHA-256 |
| --- | --- |
| `submission_preprocessed.csv` | `1e8069e500c1628505aa0a35d712627a39c5b7f9a40e368cfa14de05694646d5` |
| `submission_metadata.json` | `95d075eee37a40995dbf670868b4e3197a62ca46bcd3f06bfe0cab4eac8ef8a8` |

This evidence covers the submission output only. It does not claim that
`training/train_v1.csv` or `training/train_v2.csv` was restored in the clean
clone. The full procedure and split identifiers are in
[`dataset_release.md`](dataset_release.md).

## Streamlit–Evidently verification

The dashboard evidence was inspected in the separate Streamlit branch at
revision `17bd33a5e90fbaacc06fceb49ff533375a571f0a`; that branch was not
merged into `feature/data-release-qa`. Durable repository-relative evidence at
that revision is:

- [`docs/dashboard_monitoring.md`](https://github.com/aszenas/MlSecOps_project/blob/17bd33a5e90fbaacc06fceb49ff533375a571f0a/docs/dashboard_monitoring.md), the reproducible demo and dashboard contract;
- [`scripts/run_persisted_monitoring_demo.py`](https://github.com/aszenas/MlSecOps_project/blob/17bd33a5e90fbaacc06fceb49ff533375a571f0a/scripts/run_persisted_monitoring_demo.py), the isolated persisted-event replay;
- [`app/streamlit_dashboard.py`](https://github.com/aszenas/MlSecOps_project/blob/17bd33a5e90fbaacc06fceb49ff533375a571f0a/app/streamlit_dashboard.py), including the JavaScript-capable `st.iframe` embedding;
- [`tests/integration/test_persisted_monitoring.py`](https://github.com/aszenas/MlSecOps_project/blob/17bd33a5e90fbaacc06fceb49ff533375a571f0a/tests/integration/test_persisted_monitoring.py), the normal, shifted and insufficient-data assertions.

The recorded demonstration results are:

| Scenario | Valid current rows | State | Drift decision | Drifted fields |
| --- | ---: | --- | --- | --- |
| Normal | 3,000 | `ready` | `false` | none |
| Shifted | 3,000 | `ready` | `true` | `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm` |
| Insufficient | 1 | `insufficient_data` | `null` | no current report paths |

The dashboard contract also records the UTC window, sample counters,
reference DVC/split metadata, and the current-state-selected Evidently HTML
report. In `ready`, the report is presented through `st.iframe` and the
loaded HTML is offered for download; `no_run` and `unavailable` do not show a
report. Earlier report directories remain on disk but are not current unless
selected by `monitoring_state.json`.

The demonstration is a direct replay into isolated SQLite event stores using
`save_prediction_event`; it is not an API endpoint-to-endpoint test. No fresh
browser screenshot was captured for this record, and the HTML download was
not previously verified byte-for-byte in a browser. Those are open evidence
items, not successful checks.

## Boundaries and CI status

At the time of the earlier verification above, the dashboard's audit panel
had no separate automatic-retraining marker. New runs can now carry
`operation=automated_retraining`; the new Docker evidence is recorded below.
The champion value still comes from the current Registry alias and is not
inferred from an earlier training result.

Local documentation and repository checks are separate from GitHub Actions.
The current workflow is [`python-qa.yml`](../.github/workflows/python-qa.yml).
No remote run and no uploaded artifact for the workflow revision containing
the report-publication changes was inspected in this record, so GitHub CI is
not marked successful here. The workflow now labels coverage as `partial`
unless all three test groups pass, and reports artifact availability only
when the upload step succeeds. A remote run is still needed to verify those
paths and the resulting artifact in GitHub.

The record is therefore evidence for the submission restore and the recorded
local monitoring scenarios, not a claim that the complete project is
end-to-end verified.

## Automated retraining verification

Date: 2026-11-23. Evidence source: the completed one-shot Docker worker output
provided during development. Real API requests populated the persisted store.

| Item | Observed result |
| --- | --- |
| Monitoring window (UTC) | 2026-11-23T19:31:15.525626Z to 2026-11-23T20:31:15.525626Z |
| Valid current samples | 100 |
| Drift decision | true; all six monitored fields drifted |
| Prediction export | 100 events |
| Export run ID | `cc452b6d6a084958878662c02a10d85e` |
| Training run ID | `a6a0b5bc203244739d3265ea31ef2a95` |
| Attempt ID | `1ea43c80-3844-48e2-bed0-620d510b3266` |
| Candidate Registry version | 3 |
| Selected champion | 2 |
| Promotion | false: `f1_not_improved` |
| Candidate and champion F1 | 0.6538461538461539 |
| Automatic-run tagging | completed |

The candidate was retained in the Registry, while the existing champion was
selected because the validation F1 scores were equal. This is a successful
rejection result, not a failed training run.

A second one-shot container invocation completed with `duplicate_input` for
`b740c7a32109e28365b89752a09e6096.dir:64dc883661cde200d28e59ce1c48af2e`.
Its monitoring window ended at `2026-11-23T20:38:01.833203Z` and contained
101 valid events. The export uploaded only one new event to run
`b3ff97f3d8a341afa9ffa74dffe27e43`. No training call was made by that cycle.
This verifies persistent duplicate protection across separate worker containers.
The individual post-training prediction response was not included in this
record; the additional persisted event alone does not establish its served version.

The requests repeated identical input as synthetic test traffic. The drift
result verifies the integration and does not establish production-data drift.
The emitted dependency warnings did not prevent completion.

This record does not claim a Docker promotion, a directly inspected
post-training prediction response, or successful CI for the new branch
before those checks are observed.
