# MLSecOps – Modelltanítás és MLflow-kísérletkövetés (3.1)

**Projekt:** Predictive Maintenance / MLSecOps  
**Fejlesztési branch:** `feature/mlflow-training` (a `dev` branchből)  
**Dokumentálás dátuma:** 2026. november 1.  
**Cél:** A 3.1-es feladat megvalósításának, ellenőrzésének és korlátainak bemutatása, valamint pontos fájl- és függvényszintű átadás Máté **4.1/10 – A `/train` és `/predict` végpontok bekötése** feladatához.

> **Státuszjelölés:** A **megvalósított** funkció a fejlesztés során elkészült; a **lokálisan ellenőrzött** azt jelenti, hogy sikeres futásról vagy tesztről kaptunk eredményt; a **hátralévő** funkció még nem része a 3.1-nek. A dokumentum nem állítja, hogy már sikeres a GitHub Actions ellenőrzés, megtörtént a PR-merge, működik az éles FastAPI, elkészült a Registry-integráció vagy a predikciós események MLflow-naplózása.

## 1. Cél és feladatkör

A korábbi 2.1-es adat-előkészítésre épülve a 3.1-es feladat két bináris gépmeghibásodás-előrejelző modellt tanít: **Gradient Boosting** és **XGBoost**. Elkészült a közös training komponens, az egységes validációs kiértékelés, az MLflow experiment tracking, a teljes preprocessing+modell Pipeline mentése és visszatöltése, a reprodukálhatósági ellenőrzés, valamint az automatizált tesztelés alapja.

**Megvalósult:** két baseline modell; train-only osztálysúlyozás; egyetlen fitted sklearn Pipeline; hat validációs metrika; MLflow paraméter-, metrika- és modellartifact-naplózás; `cloudpickle` mentés; modell-visszatöltés; mockolt unit és izolált SQLite-alapú MLflow integrációs tesztek.

**Nem része a 3.1-nek:** a `/train` és `/predict` HTTP-végpontok valódi bekötése; API request/response validálás; predikciók tartós naplózása; predikciós JSON/CSV artifactok MLflow-ba mentése; modellválasztás a Model Registryből; közös távoli MLflow; automatikus hyperparameter/threshold optimalizálás; driftfigyelés és éles telepítés. Ezeket nem szabad a jelenlegi training tracking képességeiként bemutatni.

## 2. Adatok, szivárgásvédelem és felosztás

- Lokális adatforrás: `data/raw/predictive_maintenance_5000.csv` – 5000 rekord, ebből 163 pozitív (`target=1`), azaz körülbelül **3,26%** meghibásodás. A valódi CSV GitHub Actionsben a külön DVC-megoldásig hiányozhat.
- Hat bemeneti oszlop: `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`.
- A preprocessing további jellemzőt képez: `temperature_difference_k = process_temperature_k - air_temperature_k`; ezt imputálás, skálázás és one-hot encoding követi. A jelenlegi adatállományból így **kilenc transzformált feature** keletkezik.
- A `udi`, `product_id`, `failure_type` és `target` nem lehet modellbemenet. Különösen a `failure_type` okozna célváltozó-szivárgást.
- Reprodukálható, stratifikált kezdeti felosztás `random_state=42` értékkel: **3000 train, 750 validation, 750 future, 500 final test**.
- Az imputert, encodert, scalera és modellt **csak a train adatokon illesztjük**. A validation a modellösszehasonlításé, a future a későbbi drift/retraining szimulációé, a végső test pedig független értékelésre tartandó fenn.
- Az osztálysúlyokat kizárólag `y_train` alapján számoljuk (`calculate_sample_weights`), és a tanításkor adjuk át. XGBoostnál nem alkalmazunk emellett még egy ugyanarra az osztályarányra épített `scale_pos_weight` korrekciót.

## 3. Architektúra és fájltérkép

```text
src/predictive_maintenance/
├── data/
│   ├── loader.py                 # nyers tanító CSV beolvasása és ellenőrzése
│   ├── preprocessing.py          # DataProcessor; feature engineering/preprocessing
│   ├── splitting.py              # DataSplits; create_initial_data_splits
│   ├── imbalance.py              # calculate_sample_weights(y_train)
│   └── persistence.py            # opcionális, megbízható joblib preprocessing artifact
├── models/
│   ├── training.py               # build_model_pipeline, train_model, ModelType
│   ├── evaluation.py             # evaluate_model: 6 metrika
│   └── mlflow_tracking.py        # run_training_experiment; EXPERIMENT_NAME
└── api/
    ├── main.py                   # meglévő API-váz
    ├── routes.py                 # végpontok, Máté feladata
    ├── schemas.py                # Pydantic request/response sémák, Máté feladata
    └── mock_service.py           # jelenlegi mock-szolgáltatás, bekötendő

scripts/
├── run_baseline_experiments.py   # helyi, két modelles baseline futtatás
└── verify_mlflow_model.py        # mentett MLflow-modell ellenőrzése

tests/
├── unit/
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_mlflow_tracking.py
│   └── test_api.py
└── integration/
    ├── test_model_training.py     # valódi CSV-t igénylő integráció
    └── test_mlflow_integration.py # szintetikus adat + ideiglenes SQLite

docs/
├── mlflow_training_branch_documentation_en.md
└── mlflow_training_branch_documentation_hu.md
.github/workflows/python-qa.yml
requirements.txt
```

A felsorolt elérési utak a fejlesztés során egyeztetett struktúrát követik. A végleges PR előtt a tényleges repositoryt, különösen az API-váz aktuális állapotát ellenőrizni kell.

## 4. A 3.1-es modulok műszaki leírása

### 4.1 `models/training.py` – modelltanítás

```python
ModelType = Literal["gradient_boosting", "xgboost"]

build_model_pipeline(preprocessor, model_type="gradient_boosting") -> Pipeline
train_model(X_train, y_train, preprocessor, model_type="gradient_boosting") -> Pipeline
```

A `build_model_pipeline()` egy nem tanított preprocessort klónoz, majd hozzákapcsolja a `GradientBoostingClassifier(random_state=42)` vagy az `XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=1.0, n_jobs=1)` osztályozót.

A `train_model()` elutasítja az üres vagy eltérő hosszúságú adatokat, a hiányzó targetértékeket és az egyosztályos tanítóhalmazt. Az `imbalance.py` modul `calculate_sample_weights()` függvényével a train-adatokból súlyokat képez, majd `model__sample_weight=weights.to_numpy()` paraméterrel illeszti a Pipeline-t.

**Visszatérési érték:** egyetlen **betanított** sklearn Pipeline, `preprocessor` és `model` lépéssel. A nyers, de már snake_case nevű hat input mezővel használható: `pipeline.predict(X)` és `pipeline.predict_proba(X)`. Önmagában **nem naplóz MLflow-ba**.

### 4.2 `models/evaluation.py` – validációs értékelés

```python
evaluate_model(model, X_validation, y_validation, threshold=0.5) -> dict[str, float]
```

Hat mérőszám: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `average_precision`. Az 1-es (failure) osztály valószínűségét a `predict_proba(X)[:, 1]` oszlopból számítja. Ellenőrzi a bináris targetet, a sorok számát, a probability-mátrix alakját és véges értékeit, illetve a `(0,1)` közötti thresholdot. A precision/recall/F1 nem definiált eseteit `zero_division=0` kezeli. Nem vizsgálja automatikusan a final test halmazt.

### 4.3 `models/mlflow_tracking.py` – experiment tracking

```python
EXPERIMENT_NAME = "predictive-maintenance-baselines"

run_training_experiment(
    splits: DataSplits,
    model_type: ModelType,
    tracking_uri: str | None = None,
) -> dict[str, object]
```

A tracking URI forrása elsőként a függvényparaméter, másodikként az `MLFLOW_TRACKING_URI` környezeti változó, végül alapértelmezetten `http://127.0.0.1:5000`. A függvény kiválasztja vagy létrehozza az experimentet, run-t nyit, elkészíti a `DataProcessor().create_preprocessing_pipeline()` objektumot, meghívja a `train_model()` függvényt, majd **csak a validation halmazon** értékel.

Az MLflow-ba kerül:

- Run neve, modell típusa, `random_state=42`, `decision_threshold=0.5`, súlyozási módszer és az egyszerűen naplózható modellparaméterek.
- `train_size`, `validation_size`, `future_size`, `test_size`.
- `validation_accuracy`, `validation_precision`, `validation_recall`, `validation_f1`, `validation_roc_auc`, `validation_average_precision`.
- Projekt-, fejlesztési stádium-, evaluation split- és Git commit-címkék (utóbbi hiba esetén `unknown`).
- Az **egész fitted preprocessing + classifier Pipeline** artifact, `mlflow_sklearn.log_model(..., name="model", serialization_format="cloudpickle", code_paths=["src"])` használatával.

**Visszaadott kulcsok:** `run_id`, `model_type`, `model_uri`, `metrics`. A `models:/m-...` MLflow 3-as logged-model azonosító **nem azonos** a Registryben verziózott modell nevével/verziójával. Predikciós kéréseket és válaszokat ez a függvény **nem naplóz**.

**Lényeges integrációs korlát:** a függvény egy teljes `DataSplits` objektumot vár, és szinkron módon tanít; HTTP-kérést, háttérfeladatot vagy időtúllépést nem kezel.

## 5. Futtatás és ismert eredmények

Aktív Python 3.11-es `.venv` és a szükséges csomagok után a lokális MLflow szerver indítása:

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --workers 1
```

A webfelület: `http://127.0.0.1:5000` → **Model training** → `predictive-maintenance-baselines`. A localhosthoz nem kell MLflow-felhőfiók. A szervert ne tegyük nyilvánosan elérhetővé megfelelő hitelesítés és hozzáférésvédelem nélkül.

Mindkét modell tanítása és naplózása a projekt gyökeréből:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_baseline_experiments.py
```

**2026. november 1-i, lokális validation eredmények** (750 rekord, threshold = 0,5):

| Metrika | Gradient Boosting | XGBoost |
|---|---:|---:|
| Accuracy | 0,9720 | 0,9827 |
| Precision | 0,5484 | 0,7391 |
| Recall | 0,7083 | 0,7083 |
| F1 | 0,6182 | 0,7234 |
| ROC-AUC | 0,9716 | 0,9706 |
| Average precision | 0,7568 | 0,7867 |

Az XGBoost ezen a felosztáson magasabb precisiont és F1-et ért el azonos recall mellett. Ebből nem következtetünk általános fölényre vagy final-test teljesítményre.

A két sikeresen naplózott, **helyi** modell URI-ja:

- Gradient Boosting: `models:/m-e512544fa021420ea31cd9e42d529826`
- XGBoost: `models:/m-fee7392716644fcfa13a3dce16c776d6`

**Figyelem:** ezek azonosítók az eredeti helyi MLflow tárolóban léteznek. Egy Docker-szolgáltatás másik gépen nem tudja őket automatikusan használni; közös szerver, átadás vagy újratanítás szükséges. Az API-ba nem szabad ezeket fix értékként beégetni.

## 6. Tesztek, coverage és CI

### 6.1 Visszatöltés és reprodukálhatóság

A `scripts/verify_mlflow_model.py` a `--model-uri` és `--model-type` alapján betölti az MLflow modellt, újra elkészíti a determinisztikus adatfelosztást és referencia-modellt, majd ellenőrzi a predikált osztályok, probability értékek és kiértékelési metrikák egyezését. A Gradient Boosting 750 rekordos tesztjének `MODEL VERIFICATION PASSED` kimenete rendelkezésre áll; az XGBoostot a fejlesztés során szintén sikeresnek jeleztük, de külön terminálkimenet nem került be a beszélgetésbe. Az ellenőrzés csak azonos adat-/kód-/függőségverzióknál várhat pontos egyezést, és nem általános tuningoltmodell-verifikátor.

```powershell
$env:PYTHONPATH = "src"
python scripts/verify_mlflow_model.py --model-uri "models:/m-e512544fa021420ea31cd9e42d529826" --model-type gradient_boosting
python scripts/verify_mlflow_model.py --model-uri "models:/m-fee7392716644fcfa13a3dce16c776d6" --model-type xgboost
```

### 6.2 Automatizált tesztek

- `tests/unit/test_training.py`: modellválasztás, shape, determinisztikus tanítás, hibás bemenetek.
- `tests/unit/test_evaluation.py`: metrikák, threshold, hibás bemenetek és probability-alakok.
- `tests/unit/test_mlflow_tracking.py`: mockolt MLflow run, paraméterek, metrikák, artifactmentés-hívás, Git SHA, tracking URI.
- `tests/integration/test_model_training.py`: a valódi CSV-n futó modellek (adatfüggő teszt).
- `tests/integration/test_mlflow_integration.py`: mindkét modell tényleges tanítása és MLflow-model round-trip tesztje, **szintetikus adatokkal és ideiglenes SQLite-adatbázissal**, külön szerver nélkül. A URI például `f"sqlite:///{(tmp_path / 'mlflow_test.db').as_posix()}"`. Az MLflow 3.17.2 a régi file-store-t már alapértelmezetten tiltja.

A bejelentett lokális eredmény: **316 vizsgált `src/` statementből 314 lefedett, 99,37%**. A `training.py` és `mlflow_tracking.py` 100%-os, az `evaluation.py` 95,65%-os. A két lefedetlen sor: `data/loader.py:96`, `models/evaluation.py:54`. Ez **lokális teszteredmény**, még nem igazolja a távoli GitHub Actions sikerét.

```powershell
$env:PYTHONPATH = "src"
python -m coverage erase
python -m coverage run -m pytest tests/unit/ -v
python -m coverage run --append -m pytest tests/integration/test_mlflow_integration.py -v
python -m coverage report -m --include="src/*" --fail-under=95
python -m flake8 src tests infra app scripts
python -m pyright
```

A `scripts/` mappát **Flake8 és Pyright ellenőrzi**, de a kötelező minimum 95%-os coverage **csak a `src/` könyvtárra** vonatkozik.

### 6.3 GitHub Actions – elvárt minőségkapuk

A `.github/workflows/python-qa.yml` feladata: függőségek telepítése; unit tesztek; MLflow integrációs tesztek **mindig**; CSV-függő integrációs tesztek a dataset megléte esetén, a duplikált MLflow-futtatás kizárásával; `src/` coverage >=95%; Flake8 `src tests infra app scripts`; Pyright a `scripts/`-re is. A DVC bevezetéséig a hiányzó CSV miatti kihagyást warning és job summary jelölje.

**Ellenőrizendő függőség:** a megosztott `requirements.txt` tartalmazta az `mlflow==3.17.2` csomagot, de **az `xgboost` hiányzott belőle**. A PR/CI előtt a tesztelt XGBoost-verziót is rögzíteni kell. Tiszta Python 3.11-es telepítéssel és GitHub Actions futással igazolandó a dependency-kompatibilitás. A dokumentum nem állítja, hogy a PR már merge-elve lenne.

## 7. Technikai korlátok és biztonság

1. A `cloudpickle` képes Python-kódot végrehajtani betöltéskor, ezért **csak megbízható MLflow-artifactot töltsünk be**. Külső felhasználó ne adhasson tetszőleges modell URI-t.
2. Az MLflow input signature egyes integer oszlopokra és a hiányzó értékekre figyelmeztetett. A modellkiszolgálás előtt következetes dtype- és nullable-kezelés, valamint tesztek szükségesek.
3. A helyi `localhost:5000` MLflow cím más gépről nem érhető el; a közös/Docker-es telepítés külön konfigurációt igényel.
4. A 3.1 nem naplózza a predikciós kéréseket, válaszokat vagy JSON/CSV állományokat MLflow-ba.
5. A training szinkron; HTTP timeout, konkurencia, job státusz és háttérfuttatás API-szintű szervezési feladat.
6. A jelenlegi threshold 0,5; a threshold tuning, drift és retraining még nem része a 3.1-nek.
7. Predikciós auditadatoknál adatszegény naplózás, titokkezelés, hozzáférés-védelem és megőrzési idő szükséges.

---

# 8. Átadás Máténak – 4.1/10: a `/train` és `/predict` működő bekötése

**Máté feladata:** a meglévő API-váz összekötése az elkészült preprocessing és ML-modellekkel. A `/train` végpont valódi tanítást indítson, és **minden training run** paramétereit, metrikáit, modell- és preprocessing artifactjait az MLflow-ba mentse. A `/predict` validált gépadatokat fogadjon, valós predikciót és probabilityt adjon JSON-ban, kezelje a modellverziót és hibákat, továbbá **tartósan naplózza a predikciós kéréseket/eredményeket**, és MLflow-ba **visszakereshető, aggregált JSON/CSV prediction artifactokat** írjon. Swagger/Postman és automatizált tesztek igazolják a működést.

## 8.1 Pontosan mit hol talál?

| Máténak szükséges funkció | Fájl és függvény | Mit ad a 3.1? | Máté feladata |
|---|---|---|---|
| Eredeti tanító CSV beolvasása | `data/loader.py` → `load_raw_data(path)` | A teljes 10 oszlopos nyers CSV ellenőrzése | Megbízható forrás/path kiválasztása, hibakezelés |
| Oszlopnevek normalizálása | `data/preprocessing.py` → `DataProcessor.rename_columns(df)` | Snake_case oszlopok | A tanítás adatútvonalának vezérlése |
| Bemenet és target szétválasztása | `DataProcessor.split_features_target(df)` | Hat feature és bináris címke | Leakage elkerülése |
| Első adatfelosztás | `data/splitting.py` → `create_initial_data_splits(X,y)` | `DataSplits`, 3000/750/750/500 | Későbbi retrainingnél ne ossza újra a fix halmazokat |
| Preprocessing | `DataProcessor.create_preprocessing_pipeline()` | Teljes, nem fitted feature engineering, imputálás, encoding, scaling | `/predict`-nél nem szabad újra fitelni |
| Class weighting | `data/imbalance.py` → `calculate_sample_weights(y_train)` | `train_model()` már meghívja | Ne duplikálja |
| Betanítás | `models/training.py` → `train_model(...)` | Fitted preprocessing+modell Pipeline | HTTP-orchestration, ha közvetlenül hívja |
| Metrikák | `models/evaluation.py` → `evaluate_model(...)` | Hat metrika | Visszaadható response mezők |
| **Tanítás + MLflow** | `models/mlflow_tracking.py` → **`run_training_experiment(splits,model_type,tracking_uri=None)`** | `run_id`, `model_uri`, `model_type`, `metrics`; teljes fitted artifact | **Javasolt `/train` integrációs pont** |
| Helyi futtatás | `scripts/run_baseline_experiments.py` | Kétmodell-es CLI példa | Nem HTTP-végpont |
| Mentett modell visszatöltése | `mlflow.sklearn.load_model(model_uri)`; példa `scripts/verify_mlflow_model.py` | Fitted Pipeline, `predict`, `predict_proba` | Biztonságos modellválasztás, cache, hibakezelés |
| API-váz | `api/main.py`, `api/routes.py` | Meglévő struktúra | Mock helyett valódi hívások |
| Pydantic sémák | `api/schemas.py` | Meglévő API-váz | Hat feature-ből álló request és válaszok |
| API mock és tesztek | `api/mock_service.py`, `tests/unit/test_api.py` | Kezdőpont | Valódi API tesztek bővítése |
| MLflow tesztminták | `tests/unit/test_mlflow_tracking.py`, `tests/integration/test_mlflow_integration.py` | Mock és SQLite tesztminták | Prediction audit és artifact tesztek |

**Kritikus különbség:** a `loader.py` az **eredeti, teljes tanító CSV-t** ellenőrzi. A `/predict` ezzel szemben **csak a hat snake_case bemeneti feature-t** kapja, ezért külön Pydantic validáció kell. A kliensnek nem szabad `Target`, `Failure Type`, `UDI` vagy `Product ID` mezőt küldenie.

## 8.2 `/train` – javasolt működés

1. Pydantic-ban ellenőrizni a `model_type` értékét (`gradient_boosting` vagy `xgboost`) és a megengedett konfigurációt. Külső kliens ne adhasson tetszőleges szerveroldali fájlútvonalat vagy MLflow tracking címet.
2. Az engedélyezett kezdeti adatkészletet `load_raw_data()` segítségével beolvasni, `rename_columns()` és `split_features_target()` segítségével előkészíteni.
3. `create_initial_data_splits()` meghívása kizárólag a kezdeti 5000 rekordos dataset esetén. Későbbi retrainingnél a fix validation/test szétosztást meg kell őrizni.
4. `run_training_experiment(splits, model_type, tracking_uri=configured_uri)` meghívása: ez **már most** menti az MLflow paramétereket, hat metrikát, valamint a fitted preprocessing+modell artifactot.
5. JSON-válasz `status`, `run_id`, `model_uri`, `model_type`, `metrics` mezőkkel. Hosszú tanítás esetén job-azonosító és aszinkron `202 Accepted` folyamat indokolt lehet.
6. Megfelelő hibakezelés: hibás input, hiányzó adatok, MLflow-hiba, hibás modellválasztás, timeout, traininghiba.

**Csak tervezett API-contract; ez a végpont még nincs megvalósítva:**

```json
POST /train
{"model_type": "xgboost"}

200 OK
{
  "status": "completed",
  "model_type": "xgboost",
  "run_id": "<tenyleges-mlflow-run-id>",
  "model_uri": "models:/m-<tenyleges-modellazonosito>",
  "metrics": {"f1": 0.7234, "recall": 0.7083}
}
```

A két metrika csak a korábbi mérésből vett példa; minden API-hívásnak a saját tényleges futási eredményét kell visszaadnia.

## 8.3 `/predict` – javasolt működés

1. A beérkező egy vagy több rekordot Pydantic segítségével ellenőrizni. **Pontosan hat normalizált mező**, megfelelő adattípusok és ésszerű értéktartományok.
2. Megbízható, szerveroldalon konfigurált `model_uri` kiválasztása; később Registry alias/verzió használható. A jelenlegi `models:/m-...` azonosító nem jelenti automatikusan, hogy Registry-verzió készült.
3. `mlflow.sklearn.load_model(model_uri)` segítségével a teljes Pipeline betöltése vagy megbízható cache használata. **Nem szabad külön újra elvégezni a preprocessinget vagy fitelni**, mert a modellartifact tartalmazza.
4. Pandas DataFrame építése a hat oszloppal. `model.predict_proba(X)[:,1]` alapján failure probability; a jelenlegi `0.5` threshold alapján bináris eredmény. Ha később más threshold érvényes, azt explicit kell alkalmazni; a modell `.predict()` önmagában nem veszi figyelembe az új thresholdot.
5. JSON-válasz: `failure_prediction`, `failure_probability`, `model_uri`, megfelelő `model_version`/azonosító és `request_id`.
6. **Újonnan fejlesztendő tartós predikciós audit**: request/batch ID, időpont, modell URI/verzió, engedélyezetten tárolható bemeneti mezők vagy anonimizált változat, probability, kimenet, státusz, hibák. Legyen szabályozott retention, hozzáférés és újrapróbálkozás; titkok ne kerüljenek a logokba.
7. **Újonnan fejlesztendő MLflow prediction artifactok**: az összegyűjtött predikciókat külön, például `operation=predict`, `model_uri`, `batch_id` taggel ellátott prediction runban aggregálni, majd `mlflow.log_dict(payload, "predictions/predictions.json")` vagy `mlflow.log_artifact(csv_path, artifact_path="predictions")` segítségével naplózni. Az MLflow UI-ból ténylegesen legyenek visszakereshetők. **Ezt a jelenlegi 3.1-es tracking függvény még nem végzi.** Nagyobb terhelésnél inkább batch és perzisztens queue/flush mechanizmus ajánlott, mint minden egyes API-hívásra külön MLflow run.

**Szemléltető input és response, nem tényleges predikció:**

```json
POST /predict
{
  "type": "M",
  "air_temperature_k": 298.4,
  "process_temperature_k": 308.3,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 120
}

200 OK
{
  "failure_prediction": 0,
  "failure_probability": 0.12,
  "model_uri": "models:/m-<konfiguralt-modellazonosito>",
  "model_version": "<konfiguralt-verzio-vagy-azonosito>",
  "request_id": "<generalt-azonosito>"
}
```

A `0.12` érték itt **kitalált szemléltető érték**, nem a megadott gépadatokra kiszámolt eredmény. Az MLflow korábban integer input signature és hiányzó értékek kapcsán figyelmeztetett; ezért a dtype és a nullable mezők kezelését explicit tesztekkel kell ellenőrizni.

## 8.4 Máté számára javasolt tesztek

| Terület | Elvárt ellenőrzés |
|---|---|
| `/train` | Swagger/Postman hívás valódi traininget indít; `run_id`, `model_uri`, status; MLflow-ban paraméterek, hat metrika és a teljes Pipeline artifact |
| `/predict` | Hat mezőből álló JSON → valódi probability, thresholdolt 0/1 predikció, modellazonosító |
| Inputvalidáció | Hiányzó/ismeretlen mezők, hibás tartományok/típusok, NaN/Inf, hibás modellválasztás megfelelő HTTP hibával |
| Modellbetöltés | Megbízható modell, cache; hiányzó URI, sérült artifact, nem elérhető MLflow kezelése |
| Tartós prediction audit | Kérés/batch azonosító, időbélyeg, modell, predikció, státusz valóban megmarad és visszakereshető |
| MLflow prediction artifact | Aggregált JSON/CSV külön run/batch alatt; UI-ból és kliensből visszakereshető |
| Tesztizoláció | Szintetikus fixture-ök és ideiglenes SQLite MLflow; a fejlesztő localhostja nem feltétel |
| QA | `src/` coverage >=95%, Flake8, Pyright, kötelező tesztek GitHub Actionsben |

## 8.5 Ajánlott munkamegosztás és átvételi sorrend

**Miután a 3.1 merge-elve lett a `dev`-be:** Máté frissíti az API feature branchét a `dev` alapján; beköti a meglévő training és MLflow függvényeket; megírja a Pydantic és HTTP interfészeket; megoldja a `/predict` modellválasztását és hibakezelését; **külön megvalósítja** a tartós prediction logolást és MLflow JSON/CSV artifactokat; bővíti a unit/API teszteket; végül Swagger/Postman és MLflow UI alapján ellenőrzi az eredményt, és PR-t nyit a `dev` felé.

A tracking URI-t, a közös MLflow-szolgáltatást, a tárolás jogait és a Docker-hálózatot együtt kell egyeztetni; a `localhost:5000` egy másik gépen nem az eredeti MLflow szervert jelenti.

---

## 9. Lezárási ellenőrzőlista

- [x] Két baseline modell, közös fitted preprocessing+classifier Pipeline.
- [x] Hat összehasonlítható validation metrika.
- [x] Két sikeres helyi MLflow baseline run és mentett modell.
- [x] Gradient Boosting modell-visszatöltés és referenciaegyezőség igazolt; XGBoost sikeressége jelezve.
- [x] MLflow unit és ideiglenes SQLite integrációs tesztek; jelzett lokális source coverage: **99,37%**.
- [ ] `requirements.txt`: tesztelt, verziózott `xgboost` hozzáadása és tiszta telepítés ellenőrzése.
- [ ] Végső Flake8/Pyright, teljes tesztkör és CI futtatás a tényleges commiton.
- [ ] Sikeres GitHub Actions, PR-review és merge a `dev`-be.
- [ ] Máté 4.1/10-es valódi FastAPI végpontjai, predikciós auditja és MLflow prediction artifactjai.
- [ ] Későbbi Model Registry és közös MLflow-szolgáltatás bekötése.

**Felelősséghatár:** a 3.1 biztosítja a modellezési, preprocessing, evaluation és MLflow training alapot; a 4.1/10 az API-n keresztüli elérést, a bemenetek/hibák kezelését, a modell kiválasztását, a predikciók tartós naplózását és az MLflow prediction artifactokat valósítja meg.
