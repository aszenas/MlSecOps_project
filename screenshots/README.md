# Submission verification screenshots

These screenshots were captured from the local API started from Git revision
`2cec666100807d2171508ceaaf9b36e1da0264be` (`origin/dev`). The API and MLflow
services used isolated local ports and temporary storage; no token, password,
or personal data is included.

## API evidence

- [`api/01-train.png`](./api/01-train.png) shows the real Swagger request and
  the beginning of the successful `/train` response.
- [`api/01-train-details.png`](./api/01-train-details.png) is the readable
  continuation of that response. It shows run ID
  `9c74c2c2302d4cfe8e225579dd27e4a4`, registered version `1`, champion version
  `1`, `promoted: true`, and selection reason `first_champion`.
- [`api/02-predict.png`](./api/02-predict.png) shows the real Swagger
  `/predict` request and response: prediction `0`, probability
  `0.012770727629820311`, and served model version `1`. The served version
  matches the champion returned by `/train`.

The requests were executed against the local Swagger UI at `127.0.0.1:8001`
using the repository's actual schemas. The training request was
`{"dataset_id":"default","model_type":"gradient_boosting","random_state":42}`;
the prediction request used the documented six input fields.

## Open evidence

- [`docker/01-compose-services.png`](./docker/01-compose-services.png) is not
  present: the `origin/dev` Compose configuration starts a placeholder
  Streamlit app, and its `prepare` service requires external DagsHub
  credentials. A truthful Compose service screenshot was therefore not
  produced.
- [`monitoring/01-streamlit-evidently.png`](./monitoring/01-streamlit-evidently.png)
  is not present: the `origin/dev` Compose Streamlit command points to
  `app/placeholder.py`, not the monitoring dashboard, so an Evidently iframe
  screenshot cannot be claimed from this revision.
- No Airflow screenshot was required because no implemented DAG and completed
  Airflow run were available in this environment.

The existing [`mlflow/`](./mlflow/) screenshots are retained unchanged.
Generated monitoring reports and databases remain outside the repository.
