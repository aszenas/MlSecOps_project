# MLSecOps – Preprocessing branch: megvalósítás és minőségbiztosítás

**Branch:** `feature/preprocessing-pipeline`  
**Projekt:** Predictive Maintenance / MLSecOps  
**Dokumentált állapot:** 2026. november 1.  
**Hatókör:** a branch adatbetöltési, validációs, preprocessing-, split-, class imbalance-, persistence- és tesztelési feladatai. A modelltraining, FastAPI, MLflow, monitoring, Docker és DVC távoli tároló integrációja még **nem része** ennek az elkészült branchnek.

## 1. Cél és megvalósítási elvek

Az adatfeldolgozási réteg feladata, hogy a nyers AI4I 2020 predictive-maintenance adatállományból egységes, kontrollált, reprodukálható modellbemenetet hozzon létre. Az adatok betöltése, validációja, változókiválasztása, felosztása, a tanításhoz igazított preprocessing és a class imbalance kezelése önálló, tesztelhető komponensekben történik. A későbbi MLflow-alapú tanítás és FastAPI inference ezeket a komponenseket használhatja újra.

A branch fontos alapelvei:

- A **nyers adatokat nem írjuk felül**; a feldolgozás másolatokon dolgozik.
- Az azonosító és a célváltozóra közvetlenül utaló oszlopok nem kerülhetnek a modellbemenetbe.
- A tanulható preprocessing-statisztikákat **kizárólag a kezdeti train adatokon** illesztjük.
- A validation, future és final test részeket külön kezeljük; a végső teszthalmazt nem használjuk traininghez vagy modellválasztáshoz.
- A ritka hibás esetek miatti egyenlőtlenséget a training rekordokhoz rendelt súlyokkal tervezzük kezelni; a validation/test eloszlását nem módosítjuk.
- A produkciós kódhoz unit és integration tesztek, linting, statikus típusellenőrzés és coverage-mérés tartozik.

## 2. Felhasznált adatállomány és célváltozó

**Bemenet:** `data/raw/predictive_maintenance_5000.csv` (5000 megfigyelés, 10 eredeti oszlop). A kezdeti állományban 4837 `Target = 0` és 163 `Target = 1` rekord található, tehát az 1-es osztály aránya 3,26%. A cél a bináris géphiba-előrejelzés (`target`); a modell teljesítménye **nem képezi a branchben elvégzett tesztelés tárgyát**, mert modelltraining még nincs implementálva.

**Nyers, kötelező oszlopok:** `UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Target`, `Failure Type`.

A modellhez **hat eredeti feature** kerül kiválasztásra: `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`. Az `UDI` és `Product ID` rekord-/termékazonosító, a `Failure Type` pedig a hiba kimenetét írja le, ezért leakage-veszélyes. A `target` kizárólag címke (`y`), nem feature. A később számított `temperature_difference_k` a numerikus feature-ök számát hatra emeli (5 eredeti numerikus + 1 számított). A `type` egy-hot kódolásával a jelenlegi adatokon összesen **9 transzformált jellemző** adódik.

## 3. Létrehozott és módosított Python-modulok

### 3.1. `src/predictive_maintenance/data/loader.py`

**Feladat:** `load_raw_data(data_path)` CSV-t olvas a paraméterként kapott útvonalról, előre ellenőrzi a fájl létezését és a CSV-fejléc duplikált oszlopait, majd `validate_raw_data(df)` segítségével validálja a beolvasott táblát.

**Fő ellenőrzések:** nem üres DataFrame; mind a 10 kötelező oszlop megléte; egyedi oszlopnevek; numerikus oszlopok megfelelő típusa; egész értéket igénylő mezők ellenőrzése; szöveges mezők típusa; `Target ∈ {0,1}`, hiányzó target nélkül; `Type ∈ {L,M,H}`. Az adattípusoknál a Pandas dtype-alapú ellenőrzés szerepel, nem automatikus, néma konverzió. A `FileNotFoundError` és `ValueError` hibaüzenetek segítik a hibák visszakövetését.

**Indoklás:** a gépi tanulási pipeline hibás séma esetén ne későbbi, nehezen értelmezhető sklearn-hibával álljon le. A fejléc-ellenőrzés azért előzi meg a `read_csv`-t, mert a Pandas a duplikált CSV-oszlopneveket automatikusan átnevezheti. A numerikus ellenőrzést külön `_validate_numeric_columns()` segédfüggvény végzi, ami csökkenti a `validate_raw_data` komplexitását, miközben a viselkedés változatlan marad.

### 3.2. `src/predictive_maintenance/data/preprocessing.py`

**Feladatok:**

- `rename_columns(df)`: egységes, `snake_case` oszlopneveket hoz létre a `COLUMN_MAPPING` szerint; ellenőrzi az üres bemenetet és az átnevezés utáni duplikált oszlopokat.
- `select_features(df)`: a hat kiválasztott eredeti feature-t rögzített sorrendben adja vissza; kizárja az azonosítókat, a `failure_type`-ot és a `target`-et.
- `split_features_target(df)`: különálló `X` DataFrame-et és `y` Series-t készít, ellenőrzi a target meglétét és bináris értékeit.
- `engineer_temperature_difference(df)` / `add_engineered_features(df)`: hozzáadja a `process_temperature_k - air_temperature_k` különbséget; az eredeti DataFrame változatlan marad.
- `create_imputer()`: külön, még nem illesztett imputáló `ColumnTransformer`-t ad (numerikus: medián, kategorikus: leggyakoribb érték). Ez segédkomponens, **nem** a teljes modellbemenet-előkészítő folyamat.
- `create_preprocessing_pipeline()`: létrehozza a teljes sklearn `Pipeline`-t a feature engineering, imputálás, skálázás és kategorikus kódolás lépéseivel.

**Miért mediános numerikus imputálás?** A medián a szélsőséges mért értékekre kevésbé érzékeny, mint az átlag. A kategóriákhoz a `most_frequent` egyszerű és a tanítóadatokból meghatározható pótlási szabály. A `StandardScaler` összehasonlítható skálára hozza a numerikus jellemzőket, ami későbbi modellek szélesebb köréhez is használható; a választás **nem** állítás arról, hogy Gradient Boosting esetén feltétlenül szükséges a skálázás. A `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` a `type` kategóriákat gépi tanuláshoz használható numerikus oszlopokká alakítja, és inference során egy új kategória nem okoz azonnali hibát. A kimenet Pandas DataFrame az átlátható oszlopnevek miatt.

**Miért Pipeline?** Az imputerek, skálázó és encoder tanult paraméterei együtt mozognak, ezért a `fit` csak a train részen fut, az összes többi halmazon kizárólag `transform`. Ezzel csökkentjük az adatszivárgás kockázatát. A hőmérséklet-feature függvény modul-szinten definiált, így a `joblib`-os mentés/visszatöltés könnyebben megismételhető.

### 3.3. `src/predictive_maintenance/data/splitting.py`

**Feladat:** `create_initial_data_splits(X, y, random_state=42)` a **kezdeti, pontosan 5000 rekordos** adatállományt négy, index szerint diszjunkt halmazra bontja. A `DataSplits` dataclass az összetartozó `X` és `y` részeket tárolja. A függvény ellenőrzi a rekordmennyiséget, az indexek egyezését és egyediségét, valamint a bináris targetet.

| Halmaz | Rekord | Arány | Rendeltetés |
|---|---:|---:|---|
| Train | 3000 | 60% | Kezdeti modell és preprocessing illesztése |
| Validation | 750 | 15% | Modellválasztás és kiértékelés a fejlesztés során |
| Future | 750 | 15% | Későbbi megfigyelések, drift-szimuláció és **címkével rendelkező** adatok felhasználása retraininghez |
| Final test | 500 | 10% | Elkülönített, végső értékelés |

**Miért ezek az arányok?** A kezdeti train halmaz 60%-nyi mintát biztosít, miközben külön marad egy validációs, egy későbbi adatok szerepét szimuláló és egy végső teszthalmaz. A future rész önálló megtartása a beadandó drift/újratanítási bemutatóját támogatja. Ez **projekttervezési döntés**, nem empirikusan optimalizált vagy általánosan minden problémára legjobb felosztás.

**Miért stratifikált?** A pozitív (hiba) osztály ritka (3,26%); egyszerű véletlen felosztás mellett egy kisebb halmazban aránytalanul kevés pozitív minta fordulhatna elő. A `stratify` minden részben a teljes adathoz közeli osztályarányt tart fenn. A `random_state=42` reprodukálható felosztást ad. A tesztek ellenőrzik a méreteket, indexek diszjunktságát, címkeigazítást, arányokat és determinisztikusságot.

**Korlát:** a funkció **csak a kezdeti felosztáshoz** való. A future rekordok későbbi, fokozatos hozzáadásakor nem futtathatjuk újra, mert a validation és final test partícióknak rögzítettnek kell maradniuk. A future adatok címkéi a valós deploymentben nem feltétlenül állnak azonnal rendelkezésre.

### 3.4. `src/predictive_maintenance/data/imbalance.py`

**Feladat:** `validate_binary_target(y)` ellenőrzi a nem üres, kételemű bináris célváltozót; `create_class_distribution(y)` osztályonként számot és százalékot ad; `calculate_sample_weights(y_train)` a `compute_sample_weight(class_weight="balanced")` segítségével **csak a train címkékhez** súlyokat számol, indexet megőrző `pd.Series` formában.

**Miért class/sample weighting?** A 163 pozitív minta az 5000 rekordból mindössze 3,26%, így súlyozás nélkül a tanítás könnyen a gyakori negatív osztályra koncentrálhat. A balanced formula: `w_c = n_samples / (n_classes × n_c)`. A ritkább pozitív osztály mintái így nagyobb súlyt kapnak. Nem történik SMOTE vagy az adathalmaz fizikális túlmintavétele, ezért nem generálunk szintetikus rekordokat, és az eredeti validation/test eloszlás változatlan. A súlyokat **a későbbi modell `.fit(..., sample_weight=...)` hívásában kell ténylegesen alkalmazni**; ezen a branchen csak kiszámítjuk és teszteljük őket. Újratanításnál a súlyokat az aktuális train halmazra újra kell számítani.

### 3.5. `src/predictive_maintenance/data/persistence.py`

**Feladat:** `save_preprocessing_pipeline(pipeline, artifact_path)` csak fitted sklearn `Pipeline`-t fogad el, `.joblib` kiterjesztéssel ment; `load_preprocessing_pipeline(artifact_path)` betölti és ellenőrzi a fájl létezését, a sérült bemenetek egy részét kontrollált kivétellé alakítja, ellenőrzi a Pipeline típusát és illesztett állapotát.

**Miért `joblib`?** Az illesztett sklearn-transzformerek tanult statisztikái (imputer, scaler, encoder) együtt, konzisztensten menthetők és újra felhasználhatók az inference során. A pickle-alapú formátumot **csak megbízható artifactokra** szabad használni; nem biztonságos tetszőleges külső fájlok betöltésére. A unit és integration tesztek `tmp_path` alá mentenek, ezért a branch **nem tartalmaz tartós preprocessing artifactot**. A tervezett teljes modell + preprocessing Pipeline regisztrálása MLflow-ban későbbi feladat; a helyi persistence függvények ennek előkészítő építőelemei.

## 4. Git és repository-szintű QA

A munka a `feature/preprocessing-pipeline` branchen zajlik, amelyet PR és csapattársi review után a `dev` branchbe tervezünk merge-elni. A production kód a `src/` alatt, az automatizált ellenőrzések a `tests/unit/` és `tests/integration/` alatt találhatók. A branch létrehozott/módosított gyökérszintű QA-konfigurációi:

- `.coveragerc`: line coverage, minimum 95%, HTML-riport `htmlcov/` mappába; megadott saját Python-könyvtárak (`src`, `tests`, `infra`, `app`) a jelenlegi repositoryban; `scripts/` nincs, ezért ne szerepeljen.
- `.flake8`: maximum 120 karakteres sorhossz, `max-complexity = 10`, célzott kizárások a virtuális környezet és generált fájlok számára.
- `pyrightconfig.json`: Python 3.11, `standard` mód, saját Python-kód ellenőrzése és `src` import-útvonal támogatása.
- `.github/workflows/python-qa.yml`: PR/push QA-workflow a `dev` és `main` branchhez. A korábban egyeztetett **átmeneti** megoldásban a unit teszt, coverage, Flake8 és Pyright kötelező; az integrációs teszt csak a CSV rendelkezésre állásakor fut. A hiányzó datasetet a workflow figyelmezteti, **nem tekinti sikeres integrációs tesztnek**.

**Fontos:** a workflow megléte önmagában nem blokkolja a merge-et. A `Python QA` checket a GitHub branch ruleset/branch protection beállításban required status checkként kell kijelölni, a szükséges PR-review követelménnyel együtt. A workflow tényleges GitHubon történő sikeres futtatása és a DVC-adatelérés véglegesítése további feladat.

## 5. Automatizált tesztek: létrehozott fájlok és célok

### 5.1. Unit tesztek

| Fájl | Tesztesetek | Ellenőrzött működés |
|---|---:|---|
| `tests/unit/test_loader.py` | 15 | Helyes CSV, hiányzó/üres fájl, kötelező és duplikált oszlopok, numerikus/egész/string típusok, hibás Target és Type |
| `tests/unit/test_splitting.py` | 13 | 3000/750/750/500 méret, összes rekord megőrzése, átfedés tilalma, reprodukálhatóság, stratifikáció, X/y indexegyezés, invalid bemenetek |
| `tests/unit/test_preprocessing.py` | 21 | Átnevezés, feature-választás, target validáció, hiányzó/üres input, hőmérséklet-feature, imputálás, ismeretlen kategória, 9 jellemző, skálázás, train-only statisztika |
| `tests/unit/test_imbalance.py` | 11 | Osztályeloszlás, súlyképlet, pozitív osztály nagyobb súlya, indexillesztés, súlyösszeg, újratanításkor módosuló súlyok, invalid target |
| `tests/unit/test_persistence.py` | 8 | Fitted pipeline mentése/visszatöltése, azonos output, hibás kiterjesztés, unfitted, hiányzó, rossz típusú, sérült artifact |
| **Unit összesen** | **68** | Kis, mesterséges adatokon futó izolált ellenőrzések |

A Step 12 coverage-javítás során a külön felmerült `test_coverage_gaps.py` tesztjeit **beépítettük a megfelelő meglévő test-fájlokba**, ezért a végleges tesztstruktúrában nincs külön `gaps.py`.

### 5.2. Integrációs tesztek

| Fájl | Tesztesetek | Ellenőrzött együttműködés |
|---|---:|---|
| `tests/integration/test_data_pipeline.py` | 3 | Valódi CSV → loader → DataProcessor → split → train-fitted preprocessing; sample weights az X_train indexével; fitted pipeline mentése → visszatöltése → egyező inference output |

Az integrációs tesztek a lokálisan rendelkezésre álló `data/raw/predictive_maintenance_5000.csv` fájlt igénylik. A unit tesztek ettől függetlenül futtathatók.

## 6. Teszt- és QA-eredmények

### 6.1. A teljes tesztkészlet tényleges eredménye

A legutóbbi összeállított forrás- és tesztállomány futtatása:

```text
python -m pytest tests/ -q
71 passed
```

**71/71 teszt sikeres**, ezen belül 68 unit és 3 integrációs teszt. Ez a dokumentált eredmény a tesztelt preprocessing branchre vonatkozik, **nem** a még meg nem valósított teljes MLflow/API/monitoring rendszerre.

### 6.2. Line coverage – ténylegesen mért eredmény

A kijavított forrásmodulokkal és összevont tesztekkel végzett mérésből:

| `src/predictive_maintenance/data/` modul | Stmts | Miss | Coverage |
|---|---:|---:|---:|
| `imbalance.py` | 18 | 0 | 100% |
| `loader.py` | 45 | 1 | 98% |
| `persistence.py` | 25 | 0 | 100% |
| `preprocessing.py` | 58 | 0 | 100% |
| `splitting.py` | 35 | 0 | 100% |
| **SRC összesen** | **181** | **1** | **99%** |

A `src/` + `tests/` együttes mérésnél **641 mérhető utasításból 1 maradt lefedetlen, így az összesített coverage is 99%**. A coverage a *végrehajtott utasításokat* méri (line coverage), nem garantálja minden logikai ág vagy minden jövőbeni adat hibátlan működését. Az eredmény a mért kódbázisra vonatkozik, a repositoryba később kerülő új modulokkal újra kell mérni.

Az ellenőrzéshez használható parancsok (a projekt gyökerében):

```powershell
$env:PYTHONPATH = "src"
python -m coverage erase
python -m coverage run -m pytest tests/
python -m coverage report -m --fail-under=95
python -m coverage report -m --include="src/*" --fail-under=95
python -m coverage html
```

### 6.3. Flake8 és Pyright – eredmények

A legutóbb a fejlesztői PowerShell-környezetben futtatott ellenőrzések:

- **Flake8:** a korábbi 9 figyelmeztetés/hiba után egyetlen `E302` maradt a `preprocessing.py` függvény előtti üres sorok számánál; ezt két üres sorra javítottuk, majd a fejlesztő visszajelezte, hogy elkészült. A **javítás utáni új Flake8-kimenet nincs a dokumentáláshoz csatolva**, ezért külön megerősítés szükséges egy friss futással.
- **Pyright:** `0 errors, 0 warnings, 0 informations` – a fejlesztői környezetben igazolt eredmény.

A teljes, frissített ellenőrzés parancsai:

```powershell
python -m flake8 src tests infra app
python -m pyright
```

A `scripts/` könyvtár nem létezik, ezért kihagytuk a konfigurációból és a futtatási parancsból.

### 6.4. Fontos korlátok és függőségek

1. A pozitív osztály súlyozása jelenleg **kiszámítás**, nem végrehajtott modelltraining.
2. A `future` halmaz a 2020-as datasetből létrehozott félretett rekordok gyűjteménye; nem valós, időrendi, production streaming adat.
3. A pipeline `joblib` mentése csak **tesztekben, ideiglenes könyvtárba** történik; az MLflow artifact tracking és Model Registry később készül el.
4. A GitHub Actions integration tesztje jelenleg **feltételes** a CSV elérhetőségétől függően. A DVC távoli tároló és `dvc pull` CI-integráció után ezt kötelezővé kell tenni.
5. A tesztek nem helyettesítik a teljes Docker Compose, MLflow, FastAPI és monitoring rendszer későbbi integrációs és smoke tesztjeit.

## 7. Függőségek és futtatási útmutató

A gyökérben lévő `requirements.txt` a **jelenleg implementált preprocessing és QA közvetlen függőségeit** sorolja fel: NumPy, Pandas, scikit-learn, joblib, Pytest, Coverage.py, Flake8 és Pyright. A mostani fájl kompatibilitási verziótartományokat használ; **nem teljesen zárolt, byte-for-byte reprodukálható lockfile**. A csapat által ellenőrzött Windows/Python 3.11 `.venv` konkrét verzióit később pontosan rögzíteni kell; az MLflow, FastAPI, Evidently, Streamlit és DVC függőségei csak implementálásukkor kerüljenek bele.

```powershell
# Windows PowerShell; a projekt gyökérmappájában
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"

# A unit tesztekhez nem kell a valódi CSV
python -m pytest tests/unit/ -q

# A teljes tesztcsomaghoz kell a raw CSV
python -m pytest tests/ -q

# QA
python -m coverage erase
python -m coverage run -m pytest tests/
python -m coverage report -m --fail-under=95
python -m coverage report -m --include="src/*" --fail-under=95
python -m flake8 src tests infra app
python -m pyright
```

**Környezetfüggő követelmény:** az integration teszt előtt a CSV-t a megadott relatív útvonalra kell elérhetővé tenni. A későbbi CI-ben ez DVC-alapú visszaállítással oldandó meg; a hiányzó CSV esetén a jelenlegi átmeneti workflow az integrációs tesztet nem futtatja, és ezt külön jelzi.

## 8. Lezárás és további feladatok

A preprocessing branch célzottan elkülöníti és tesztelhetővé teszi a nyers adatvalidációt, a leakage-mentes feature-kiválasztást, a reprodukálható négyes stratifikált felosztást, a train-only illesztett sklearn preprocessinget, a class imbalance mintasúlyozását és a fitted pipeline ideiglenes persistence/inference használatát. **A mért 71/71 teszt és a 99%-os src coverage az aktuális adatfeldolgozási komponensekhez tartozik.**

A PR véglegesítése előtt: ellenőrizzük a `requirements.txt` friss környezetbeli telepítését, ismételjük meg az összes QA-parancsot és nézzük meg a GitHub Actions eredményét. A `dev` branchre történő merge a csapat branching szabályai szerinti review-val és merge committal történjen. A későbbi fejlesztésekben következik a modelltraining és MLflow tracking/registry, REST API, DVC-tároló, monitoring és Docker Compose integrációja.
