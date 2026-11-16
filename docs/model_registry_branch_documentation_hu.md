# MLSecOps – Model Registry, automatikus championválasztás és Data Lineage (3.2)

**Projekt:** Predictive Maintenance / MLSecOps  
**Fejlesztési branch:** `feature/mlflow-model-registry` (a `dev` branchből)  
**Dokumentáció dátuma:** 2026. november 12.  
**Kapcsolódó dokumentumok:** `docs/mlflow_training_branch_documentation_hu.md` (3.1) és a 2.1 preprocessing branch dokumentációja  
**Cél:** A 3.2-es branchen elvégzett fejlesztések és döntések részletes magyarázata, a DVC teljes 2.1 + 3.1 + 3.2 folyamathoz történő bekötésének konkrét terve, az MLflow naplózás pontos leltára és Máté API-fejlesztésének átadása.

> **Fontos státuszjelölés:** **Elkészült** = szerepel a rendelkezésre álló 3.2 kódban és tesztekben. **Lokálisan ellenőrzött** = a fejlesztő visszajelzése szerint a legutóbbi pytest, coverage-küszöb, Flake8 és Pyright sikeres. **Tervezett / integrációra vár** = nincs még automatikusan összekötve. A DVC branch, az aktuális QA-futtató és Máté API-s forrásfájljainak jelenlegi tartalma itt nem volt ellenőrizhető; a felsorolt integrációs fájlneveket merge előtt egyeztetni kell. A sikeres helyi QA nem állítja, hogy a GitHub PR, az élesítés vagy a DVC teljes folyamata már kész.

## 1. Mit fejlesztettünk ezen a branchen, és miért?

A 2.1-es feladatban kialakítottuk a nyers adatok betöltését, oszlopnormalizálását, feature-kiválasztását, preprocessing Pipeline-ját, osztálykiegyensúlyozását és az eredetileg 5000 soros adathalmaz rögzített felosztását. A 3.1-es feladatban elkészült a Gradient Boosting és XGBoost tanítása, egységes validációja, az MLflow experiment tracking és a teljes tanított sklearn Pipeline mentése.

**A 3.1 problémája a 3.2 szempontjából:** a modell naplózása még nem jelenti azt, hogy egy névvel és verzióval rendelkező, később könnyen betölthető **champion modellünk** is van. Több training run közül objektív és auditálható szabály alapján szeretnénk kiválasztani, melyik legyen használatban.

| Fejlesztés | Hol? | Miért kellett? |
|---|---|---|
| Model Registry és verziókezelés | `src/predictive_maintenance/models/model_registry.py` | Minden challenger külön verzióként megmarad; a `champion` alias az aktuálisan kiválasztott modellre mutat. |
| Automatikus modellösszehasonlítás | `models/model_selection.py` | Azonos validation halmazon, azonos prediction threshold mellett F1 és recall alapján döntünk. |
| Kiértékelés / aliasváltás szétválasztása | `model_selection.py`, `models/training_registry.py` | Előbb döntünk és naplózunk, csak utána léptetjük elő a modellt (`apply_promotion=False`). |
| Közös orchestration belépési pont | `models/training_registry.py` | A 3.1-es training, Registry, összehasonlítás, audit és ellenőrzött aliasváltás egyetlen függvényből hívható. |
| Audithoz szükséges metrikák, státuszok | `models/training_registry.py` | Utólag bizonyítható, hogy ki/mi, milyen metrikák és döntési szabály alapján lett champion; a sikertelen váltás is felismerhető. |
| Data Lineage előkészítése | `models/data_lineage.py` | Az MLflow run összekapcsolható a későbbi DVC-adatverzióval anélkül, hogy teljes CSV-t másolnánk MLflow-ba. |
| Unit és MLflow-integrációs tesztek | `tests/unit/` és `tests/integration/` | Nemcsak mockokkal, hanem valódi SQLite-alapú MLflow Registry-vel is bizonyítjuk az alapfolyamatot. |

**Nem készült még el ezen a branchen:** DVC-ből történő automatikus adatletöltés/hashkinyerés és retraining-indítás; `/train` és `/predict` végpont bekötése; predikciós kérések tartós auditnaplózása; több párhuzamos promotion tranzakcióbiztos kezelése; távoli közös MLflow éles üzemeltetése.

### 1.1 Mit csináltunk szakmailag fontosnak?

- **Nem a legutóbb tanított modellt nevezzük ki automatikusan championnak.** A challenger előbb összehasonlításra kerül.
- **Nem csupán a recallt maximalizáljuk.** A teljes pontosság–visszahívás kompromisszumot F1-gyel értékeljük, de a minimális recall küszöböt kötelezően teljesíteni kell.
- **A preprocessort is mentjük.** A betöltött Pipeline képes ugyanazokat az átalakításokat végrehajtani, nem kell Máténak külön betanított encodert/scalert rekonstruálnia.
- **Az elutasított modelleket is megtartjuk.** Ez szükséges az összehasonlításhoz, reprodukálhatósághoz és visszaállításhoz.
- **Az audit az aliasváltás elé került.** Ha a döntés/mérőszám naplózása nem sikerül, ne történjen automatikusan előléptetés.
- **A DVC-integrációt nem blokkoljuk.** Az adatverziós mezők opcionálisak; később valós DVC hashsel tölthetők fel.

## 2. A teljes pipeline jelenlegi és tervezett felépítése

```text
[Beni DVC remote – KÉSŐBB AUTOMATIKUS BEKÖTÉS]
      |
      | dvc pull / opcionálisan dvc repro
      v
[data/raw/predictive_maintenance_5000.csv]
      |
      v
[2.1 data/loader.py: load_raw_data]
      |
      v
[2.1 data/preprocessing.py: rename_columns, split_features_target]
      |
      v
[2.1 data/splitting.py: DataSplits]
      |  X/y train, validation, future, test
      v
[3.2 train_register_and_select]
      |
      +--> [3.1 run_training_experiment]
      |      +--> training.py: tanított preprocessing + modell Pipeline
      |      +--> evaluation.py: 6 validation metrika
      |      +--> MLflow training run és model artifact
      |
      +--> [3.2 register_model -> új Registry-verzió]
      +--> [3.2 opcionális DatasetLineage JSON + tagek]
      +--> [3.2 champion–challenger kiértékelés: DRY RUN]
      +--> [3.2 selection audit]
      +--> [3.2 champion aliasváltás és visszaellenőrzés]
      |
      v
[Máté FastAPI /train és /predict – MÉG BEKÖTENDŐ]
```

### 2.1 Az adatsplit és a szivárgásvédelem

A jelenlegi `create_initial_data_splits()` **pontosan 5000 rekordot vár**. A reprodukálható `random_state=42` mellett a felosztás: **3000 train, 750 validation, 750 future, 500 test**. A validációs halmazon versenyez a challenger és a champion; a future-t drift/retraining demonstrációra tartjuk fenn; a testet végső, független értékelésre.

Hat nyers bemeneti változó: `type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`. A `target` címke, a `failure_type`, valamint a `udi` és `product_id` **nem** lehetnek inferenciabemenetek. A `failure_type` a célváltozót szivárogtatná. Az imputáció, skálázás, one-hot encoding és feature engineering a teljes, betanított Pipeline része; az illesztés kizárólag a train halmazon történik.

**Fontos DVC-következmény:** ha később nem 5000 soros, folyamatosan bővülő adat érkezik, nem szabad változtatás nélkül használni a kezdeti splitter függvényt. Külön retraining/splitpolitika kell, megőrzött validation/test referenciával. A tárolt elemszám önmagában nem bizonyítja, hogy ugyanazokat a rekordokat használtuk; ezért érdemes a split indexeit vagy egy splitmanifestet is verziózni.

## 3. A jelenlegi 3.2 API és döntési logika

### 3.1 Meglévő publikus függvények

```python
from predictive_maintenance.models.model_registry import (
    REGISTERED_MODEL_NAME,  # "predictive-maintenance-model"
    CHAMPION_ALIAS,         # "champion"
    register_model,
    set_champion_version,
    get_champion_version,
    load_champion_model,
)
from predictive_maintenance.models.model_selection import (
    MINIMUM_RECALL,  # 0.70
    evaluate_and_select_champion,
)
from predictive_maintenance.models.data_lineage import DatasetLineage
from predictive_maintenance.models.training_registry import (
    train_register_and_select,
    TrainingRegistryResult,
)
```

A teljes orchestration függvény szignatúrája:

```python
train_register_and_select(
    splits: DataSplits,
    model_type: ModelType,           # "gradient_boosting" vagy "xgboost"
    tracking_uri: str | None = None,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    minimum_recall: float = 0.70,
    threshold: float = 0.5,
    dataset_lineage: DatasetLineage | None = None,
) -> TrainingRegistryResult
```

Visszatérési mezők: `run_id`, `model_uri`, `registered_model_name`, `registered_version`, `selection`. Az utóbbi `SelectionResult`, benne: `promoted`, `reason`, `selected_version`, `challenger_version`, `champion_version_before`, `challenger_metrics`, `champion_metrics`.

**Különbség:** a `selection.promoted=True` a `apply_promotion=False` kiértékelési módban **előléptetési javaslat**, nem annak bizonyítéka, hogy az alias már átállt. A tényleges végrehajtást a `promotion_status=completed` **és** a Registry alias utólagos ellenőrzése igazolja. A training jelenleg szinkron módon fut, a függvény pedig az MLflow processz-szintű URI-beállításait is módosítja.

### 3.2 Mi alapján lesz egy modell champion?

A jelenlegi üzleti/technikai szabályunk:

1. Challenger és champion **ugyanazt** a validation adathalmazt használja, alapértelmezett `threshold=0.5` mellett.
2. **Recall legalább 0,70**. A 0,70 pontosan elfogadható; 0,699 már nem.
3. Ha nincs champion, az első recall-feltételt teljesítő modell válhat championná.
4. Ha van champion, az új modell csak **szigorúan nagyobb F1-score** esetén válthatja le (`challenger_f1 > champion_f1`). Azonos F1-nél a régi champion marad.
5. Minden challenger regisztrált verzióként megmarad, akár elfogadjuk, akár elutasítjuk.

| Régi F1 | Új F1 | Új recall | Döntés |
|---:|---:|---:|---|
| 0,72 | 0,80 | 0,75 | Új champion |
| 0,72 | 0,85 | 0,65 | Régi marad: recall túl alacsony |
| 0,72 | 0,70 | 0,90 | Régi marad: nem jobb F1 |
| 0,72 | 0,72 | 0,80 | Régi marad: azonos F1 |
| 0,72 | 0,78 | 0,70 | Új champion |

**Miért ezt választottuk?** A meghibásodások ritkák, így az accuracy önmagában félrevezető lehet. Az F1 a precision és recall harmonikus közepe, jó kompromisszum a pozitív osztály detektálására. A recall minimum külön biztosítja, hogy a modell ne váljon csak a jó precision miatt championná, miközben túl sok hibát nem vesz észre. A **70% saját projektdöntés/üzleti feltevés**, nem iparági standard vagy igazoltan biztonságos küszöb; kb. 10 tényleges meghibásodásból 3 kimaradása egy reprezentatív mérésben. Valós környezetben ezt szakértővel, költségelemzéssel felül kell vizsgálni. A ROC-AUC és average precision diagnosztikus metrika, de **jelenleg nem feltétel** a promotionhöz.

## 4. Biztonságosabb aliasváltás és audit logging

A `model_selection.py` értékelője `apply_promotion=False` paraméterrel csak döntést ad. Az aliasváltást a `training_registry.py` végzi, **miután** az auditmetrikákat és döntési tageket már naplózta.

```text
Training -> modellartifact mentése -> új Registry-verzió
    -> opcionális lineage artifact és tagek
    -> challenger/champion kiértékelés (alias módosítása nélkül)
    -> döntési tagek és metrikák mentése
    -> ELUTASÍTOTT: promotion_status = not_promoted
    -> JÓVÁHAGYOTT: promotion_status = pending
         -> ellenőrizzük, nem változott-e közben a champion
         -> alias átállítása
         -> Registry alias visszaolvasása
         -> promotion_status = completed
         -> hiba esetén: verification_required és exception
```

Az audit és a promotion külön helperfüggvényekbe került: `_log_selection_audit`, `_record_promotion_status`, `_apply_audited_promotion`. Ez csökkenti a Flake8 által jelzett `C901` komplexitást és jobb unit tesztelhetőséget ad. A függvény elutasítja az üres Registry-nevet/explicit tracking URI-t, valamint a hibás recall/threshold értékeket.

**A jelenlegi megoldás korlátai:**

- Ha a kezdeti lineage- vagy auditnaplózás hibázik, az aliasváltást nem kezdjük meg; a regisztrált challenger **ettől még megmaradhat**.
- Ha az alias már átállt, és utána a `completed` naplózás hibázik, a Registry-ben ténylegesen már új champion lehet. A `verification_required` tehát nem automatikus rollback; az aktuális aliast kell ellenőrizni.
- Két párhuzamos trainer ugyanazt a régi championt láthatja. A promotion előtti újraellenőrzés hasznos, de **nem atomi lock**. Megoldás lehet egyetlen promotion worker, idempotens job és későbbi reconciliation.
- A `training_run_id` a Registry-verziót az MLflow runhoz kapcsolja, de a modell kiszolgálásakor kezelt cache-t Máténak frissítenie kell.

## 5. Pontosan mit naplózunk most az MLflow-ban?

| Terület | Kulcsok / adatok | Tárolás | Státusz |
|---|---|---|---|
| Training | Model típus, hiperparaméterek, training run és teljes fitelt preprocessing+classifier Pipeline | MLflow training run + modellartifact (3.1) | Kész |
| Validation | `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `average_precision` | MLflow run metrics (3.1) | Kész |
| Registry | `predictive-maintenance-model`, numerikus verzió, `champion` alias | Model Registry | Kész |
| Verzió→run kapcsolat | `training_run_id`, `model_type` | Registry model-version tags | Kész |
| Döntés | `selection_status=approved/rejected`, `selection_reason`, `registered_model_name`, `registered_model_version`, `champion_version_before`, `selection_minimum_recall`, `selection_threshold` | Run tags; Registry-verzión csak `selection_status`, `selection_reason` | Kész |
| F1/recall összehasonlítás | `selection_challenger_f1`, `selection_challenger_recall`, és ha van régi champion: `selection_champion_f1`, `selection_champion_recall` | Run metrics | Kész |
| Váltás állapota | `promotion_status=pending/completed/not_promoted/verification_required` | Run tags és challenger model-version tags | Kész |
| Dataset lineage **csak ha átadjuk** | `dataset_name`, `dataset_dvc_hash`, `dataset_git_commit`, `dataset_split_version`; `split_rows` | Run tags + `lineage/dataset_metadata.json` artifact | Opcionálisan kész, DVC automatikus bekötése hátravan |

### 5.1 A tényleges Data Lineage objektum

```python
from predictive_maintenance.models.data_lineage import DatasetLineage

lineage = DatasetLineage(
    dataset_name="predictive_maintenance_5000",
    dvc_hash=None,           # később valódi, ellenőrzött DVC hash
    git_commit=None,         # később a futáskori valódi Git SHA
    split_version="initial-seed-42",  # csak ha tényleg ezt használtuk
)
```

Az MLflow `lineage/dataset_metadata.json` artifactja a fenti mezőkön túl a **tényleges** `DataSplits` objektum `train`, `validation`, `future`, `test` méreteit tartalmazza. A `log_dataset_lineage()` először menti a JSON artifactot (`MlflowClient.log_dict`), csak utána a kereshető tageket. Hibás vagy üres kötelező értéknél a hívás hibára fut. Ha `dataset_lineage=None`, **sem lineage JSON, sem dataset_* tagek nem jönnek létre**. A hash és Git SHA automatikus kinyerése még **nincs** implementálva, és a Data Lineage nem jelenik meg közvetlenül minden Registry-verzió tagnél: az eredeti runhoz a `training_run_id` alapján jutunk el.

### 5.2 Mit kell még naplózni, és hova?

| Szükséges jövőbeli adat | Miért? | Hol / ki? |
|---|---|---|
| Valós DVC objektumhash, Git SHA, adatútvonal, DVC tároló azonosítója | A kód és adatok pontos visszaállíthatósága | CLI / DVC adapter -> MLflow run lineage |
| Splitindex-manifest vagy splitfájlok verziója | Azonos validation kohorsz bizonyítása, nem csak azonos elemszám | DVC és/vagy verziózott manifest + lineage artifact |
| Targeteloszlás, feature schema, preprocessing konfiguráció és hash | Adatok és pipeline-változások felismerése | MLflow artifact / DatasetLineage bővítés |
| Training / registration korai hibák strukturált státusza | A sikertelen workflow-k teljes auditja | Job orchestration, MLflow ahol elérhető |
| Predikciós request ID, időbélyeg, ténylegesen kiszolgált model version, threshold, probability, döntés, hiba | Inferencia auditálása | Máté `/predict` szolgáltatása; MLflow/log store a projektkövetelmény szerint |
| Predikciós JSON/CSV eredményartifactok | Visszakereshető demonstráció és nyomon követés | Máté API feladatában, adatvédelmi szabályokkal |
| Drift/retraining esemény és kiváltó ok | Melyik drift vezetett melyik új championhoz | Monitoring + későbbi orchestration |

**Tárolási stratégia:** maga a nagy CSV és az esetleges előállított splitfájlok a DVC remote-ba kerüljenek; DVC/Git metadata és kis konfigurációk maradjanak Gitben; az MLflow-ban training paraméterek/metrikák, modellartifactok, auditadatok és a datasetet azonosító **hivatkozások** legyenek. A teljes nyers predikciós inputok indiszkiminált naplózása nem ajánlott; legyen jogosultság- és retentionpolitika.

## 6. A DVC teljes bekötésének terve – részletes integrációs szerződés

**EZ A FEJEZET MÉG NEM IMPLEMENTÁLT INTEGRÁCIÓ, HANEM A KÖZÖS FEJLESZTÉS KONKRÉT TERVE.** A DVC `.dvc` metadatával verziózhat adatot; `dvc.yaml` írhat le függőségi gráfot, `dvc.lock` rögzíti a stage-függőségeket, `dvc pull` állítja helyre a távoli adatot, `dvc repro` futtatja a reprodukálható stage-eket. A `dvc repro` önmagában **nem garantálja**, hogy előbb `dvc pull` történik.

### 6.1 Mi kell Benitől / a közös DVC branchből?

1. **A valós DVC setup:** `.dvc/config`, remote neve, típusa, hozzáférés; `.dvc/config.local` és titkok ne legyenek Gitben. Ha például S3/Azure van, a megfelelő DVC extra és CI-secrets kellenek; nem szabad találgatni a remote-ot.
2. **A rögzített adatelérési út:** a helyes kezdeti raw adatállomány `data/raw/predictive_maintenance_5000.csv`. Ellenőrizni kell, hogy ezt `data/raw/predictive_maintenance_5000.csv.dvc`, egy **mappaszintű** `.dvc` kimenet vagy valamelyik `dvc.lock` stage írja le. Nem lehet vakon feltételezni, hogy pontosan melyik fájl tartalmazza a hasht.
3. **Friss klónból működő `dvc pull`:** valós adatfájl visszaállítása, ellenőrzött DVC remote és access; a CSV ne legyen egyszerre Gitben és DVC-ben megduplázva.
4. **Rögzített verziók:** `.dvc` metadata, `.dvc/config`, opcionálisan `dvc.yaml`, `dvc.lock`, `params.yaml` Git commit; az adatcache/remote külön tárolandó.
5. **Splitverzió megállapodás:** csak raw CSV DVC-verziót használunk és determinisztikusan újraosztunk? Vagy a train/validation/future/test split indexei/fájljai is külön DVC-verziót kapnak? **Javaslat: legyen verziózott splitmanifest**, mert a sorok rendezése vagy a splitlogika később megváltozhat.
6. **Új adat / retraining feladatköre:** változó adatszámra új splitpolitika és fagyasztott validation/test referenciák; az aktuális `create_initial_data_splits` kizárólag a jelenlegi kezdeti 5000 sorra készült.
7. **Futtatási modell:** DVC csak **adatverziózást** szolgáltat, és a Python orchestration indítja a traininget? Vagy közös `dvc.yaml` `prepare`/`train` stage-eket építünk? Ezt még külön el kell dönteni.
8. **MLflow környezet:** a trainer és Máté API-ja közös tracking + Registry szervert és artifact store-t lásson. A `localhost:5000` külön gépen/konténerben nem ugyanaz a szolgáltatás.

### 6.2 Pontosan melyik fájlt kell módosítani?

| Fájl | Teendő | Konkrét cél |
|---|---|---|
| `.dvc/config`, `.dvc/config.local`, `.dvcignore` | **Benivel egyeztetni** | DVC remote, cache és credentials; helyi titkok ne kerüljenek commitba. |
| `.dvc` adatmetadata (egyedi CSV vagy mappa), `dvc.lock` | **Valódi fájlok megvizsgálása** | A betöltendő raw CSV azonosítója, DVC hash kinyerési módja. |
| `.gitignore`, `.dvcignore` | **Ellenőrizni** | Raw CSV ne legyen duplán verziózva Gitben; metadata maradjon követhető. |
| `requirements.txt` vagy lockfile | **Felülvizsgálni** | `dvc` + tényleges remote extra, MLflow, XGBoost, Pytest/QA verziók; Python 3.11 kompatibilitás. |
| `src/predictive_maintenance/data/loader.py` | **Újrahasznosítani** | DVC által helyreállított CSV olvasása `load_raw_data(path)` függvénnyel. |
| `src/predictive_maintenance/data/preprocessing.py` | **Újrahasznosítani** | `DataProcessor.rename_columns`, `split_features_target`, preprocessing Pipeline. |
| `src/predictive_maintenance/data/splitting.py` | **Csak új splitpolitika esetén bővíteni** | 5000 soros kezdeti split vs. későbbi retraining; ugyanaz a validation referencia. |
| `src/predictive_maintenance/models/training.py` | **Változatlanul felhasználni** | Modell tanítása, már létező train-only osztálysúlyozás. |
| `src/predictive_maintenance/models/mlflow_tracking.py` | **Alapból felhasználni, később bővíthető** | Training metrics/run + model artifact; opcionálisan MLflow Dataset Input/schema. |
| `src/predictive_maintenance/models/data_lineage.py` | **Szükség szerint bővíteni** | A jelenlegi `DatasetLineage` kapjon valós DVC hash/Git SHA/split version értékeket; opcionálisan adatútvonal és manifesthash. |
| `src/predictive_maintenance/models/training_registry.py` | **Már készen áll a bekötésre** | `dataset_lineage=...` paraméter átadása; a lineage írása megelőzi a championváltást. |
| `src/predictive_maintenance/models/model_selection.py`, `model_registry.py` | **Alapból nincs szükség átírásra** | Megmarad az F1+recall döntési logika; változó validation kohorsz esetén összehasonlítási politika felülvizsgálata. |
| **Új, javasolt** `scripts/run_dvc_training.py` | **Létrehozni** | DVC-restored path -> loader -> DataProcessor -> DataSplits -> Git/DVC metadata -> `train_register_and_select`; értelmezhető CLI eredmény/hiba. |
| **Opcionális** `dvc.yaml`, `params.yaml`, `dvc.lock` | **Benivel közösen kialakítani** | Reprodukálható prepare / adott esetben training stage-ek és paraméterek. |
| `tests/unit/test_data_lineage.py`, `test_training_registry_lineage.py` | **Kiegészíthető** | Hash/Git SHA továbbadás, hibás metadata esetén nincs promotion. |
| **Új, javasolt** `tests/integration/test_dvc_training_integration.py` | **Létrehozni** | Valódi `dvc pull`/kontrollált remote -> teljes 2.1+3.1+3.2 run -> lineage artifact/alias és a hashek ellenőrzése. |
| `tests/integration/test_training_registry_integration.py` | **Mindig futtatni** | Ez most szintetikus adatokkal és izolált SQLite MLflow-val fut, nem igényel DVC-t! |
| **QA Python futtató** (pontos fájlnév itt nem ismert) | **Felülvizsgálni, módosítani** | DVC integration tesztek külön, látható futtatási tierben; ne legyenek elrejtve/állandóan kihagyva. |
| `.github/workflows/python-qa.yml` (valódi útvonalat ellenőrizni) | **CI módosítás** | DVC dependency, hozzáférés titokból, `dvc pull`/DVC test, futások kimenetének látható jelzése, 95% coverage. |
| `src/predictive_maintenance/api/{main,routes,schemas,mock_service}.py` (fájlnevek ellenőrizendők) | **Máté feladata** | `/train`, `/predict`, schema, champion loading, MLflow prediction logging. |
| `docs/` | **Kiegészíteni** | DVC remote setup, tesztparancsok, reproducibility és API handoff. |

### 6.3 Tervezett adapter: DVC -> teljes MLflow pipeline

**A következő PSEUDOKÓD: a DVC-adatok kiolvasásához tartozó függvények még nem léteznek.** Nem szabad őket kész megoldásként meghívni addig, amíg Beni tényleges DVC-konfigurációját meg nem ismertük.

```python
# Példafolyamat; előbb a közös DVC setupot kell ellenőrizni.
raw_path = Path("data/raw/predictive_maintenance_5000.csv")
assert raw_path.is_file()  # `dvc pull` után
raw = load_raw_data(raw_path)
processor = DataProcessor()
normalized = processor.rename_columns(raw)
X, y = processor.split_features_target(normalized)
splits = create_initial_data_splits(X, y, random_state=42)

lineage = DatasetLineage(
    dataset_name="predictive_maintenance_5000",
    dvc_hash=read_verified_dvc_output_hash(...),  # ÚJ ADAPTER
    git_commit=read_git_head_sha(...),             # ÚJ ADAPTER
    split_version=resolve_frozen_split_id(...),   # ÚJ ADAPTER
)
result = train_register_and_select(
    splits=splits,
    model_type="xgboost",
    tracking_uri=server_configured_mlflow_uri,
    dataset_lineage=lineage,
)
```

A `read_verified_dvc_output_hash`, `read_git_head_sha`, `resolve_frozen_split_id` elnevezések **tervezett helperfüggvények**. A DVC-fájlszerkezet ismerete nélkül nem állíthatjuk, hogy egy konkrét `.dvc` fájlban biztosan azonos mezőben van a keresett hash. Ellenőrzött DVC API/parancs vagy tesztelt parser kell. A Git commitot a futáskor `git rev-parse HEAD` paranccsal olvassuk ki. **Ha egy runra DVC-reprodukálhatóságot akarunk garantálni, a hiányzó hash legyen hiba** már a tanítás megkezdése előtt. A nem DVC-s demonstrációs futásnál jelenleg szándékosan opcionális a lineage.

### 6.4 DVC stages: milyen config jöhet szóba?

A DVC pipeline szerkezete (csak példa, **nincs még implementálva**):

```yaml
stages:
  prepare:
    cmd: python scripts/prepare_dataset.py
    deps:
      - data/raw/predictive_maintenance_5000.csv
      - scripts/prepare_dataset.py
      - src/predictive_maintenance/data/preprocessing.py
      - src/predictive_maintenance/data/splitting.py
    outs:
      - data/processed/split_manifest.json
```

A `scripts/prepare_dataset.py` és `data/processed/split_manifest.json` csak javaslat. A `prepare` stage készíthetne rögzített splitindex-manifestet, ellenőrizné az adat és célváltozó struktúráját. A további `train` stage-et **külön** kell kialakítani: a DVC alapvetően deklarált fájlfüggőségekkel és kimenetekkel dolgozik, de az MLflow Registry alias módosítása **külső mellékhatás**. DVC cache miatt az azonos inputú stage el is maradhat; ezért a `dvc repro` nem jelent automatikus „minden alkalommal új modellverzió és championválasztás” garanciát. Ha train stage-et építünk, legyen valós lokális eredményfájl (`training_result.json`) és definiált újrafuttatási/idempotencia-politika. MLflow távoli szervert ne próbáljuk DVC `outs`-ként megadni.

### 6.5 Javasolt pontos végrehajtási sorrend

1. **Közös `dev` branch ellenőrzése:** Beni DVC metadata + remote + data path + `.gitignore` + tényleges QA-futtató beazonosítása. Éles secretet nem commitolunk.
2. **Fresh clone próba:** `dvc pull`, CSV ténylegesen megjelenik; a `load_raw_data()` validációja sikeres. Ne legyen helyspecifikus abszolút Windows path.
3. **Splitmegállapodás:** 5000 sor esetén rögzített validáció/test; új, bővülő adatnál külön splitlogika. Legyen verziózott splitazonosító vagy indexmanifest.
4. **DVC/Git metadata adapter és CLI:** valódi hash kiolvasása; `scripts/run_dvc_training.py` összekötés a meglévő `train_register_and_select()`-tel.
5. **MLflow lineage ellenőrzés:** runon `dataset_*` tagek, `lineage/dataset_metadata.json`, a Registry `training_run_id` hivatkozása. Ellenőrizni kell, hogy mindez tényleg a DVC-ből visszaállított tartalomhoz tartozik.
6. **Teljes integrációs teszt:** első champion, jobb F1 challenger, elutasítás recall miatt, azonos F1, hiba miatti promotion tiltás; valós DVC dataset és izolált MLflow tesztkörnyezet.
7. **QA/CI és dokumentáció:** dedikált DVC teszt tier látható futási eredménnyel; közös MLflow és DVC jogosultságok; visszaállítási runbook.

## 7. QA-futtató és GitHub Actions: mi kell a DVC-hez?

**Jelenlegi visszajelzés:** a legutóbbi kódon a teljes pytest, Flake8, Pyright és legalább 95%-os coverage ellenőrzés zöld. A pontos tesztszámot és százalékot itt nem kaptuk meg, ezért ezt nem találjuk ki.

```powershell
$env:PYTHONPATH = "src"
python -m flake8 src tests infra app scripts
python -m pyright
python -m coverage erase
python -m coverage run -m pytest tests/ -q
python -m coverage report -m --include="src/*" --fail-under=95
```

**A DVC még NEM része a teljesen automatizált QA-nak.** Különítsük el a következő tesztosztályokat:

| Tier | Tesztek | Kell valódi DVC? | Elvárt futtatás |
|---|---|---|---|
| Unit | `tests/unit/` | Nem | Mindig |
| Szintetikus MLflow-integráció | `test_mlflow_integration.py`, `test_model_registry_integration.py`, `test_training_registry_integration.py` | Nem; ideiglenes SQLite | Mindig, DVC nélkül is |
| Régi, valódi CSV-s integráció | Pl. `test_model_training.py`, `test_data_pipeline.py` – tényleges fájlnév ellenőrzendő | Az adatok elérhetősége kell | `dvc pull` után; addig explicit skip/ok, de ne hallgatólagos kihagyás |
| **Új DVC end-to-end** | **Tervezett** `test_dvc_training_integration.py` | Igen, remote vagy kontrollált teszt-remote | Külön helyi/CI job, futás ténye naplózva |
| API-integráció | Máté FastAPI TestClient tesztjei | Alaptesztnek nem; külön DVC job lehet | Mindig |

**QA Python futtató:** a pontos aktuális kódját nem kaptuk meg, ezért nem adok kitalált sorszámot vagy biztos fájlnevet. Ha jelenleg úgy fut, hogy a `pytest` csak unit és néhány integrációs fájlt indít, azt módosítani kell a DVC tesztek explicit beemelésével. Ha `-m "not dvc"` van benne, legyen külön **pozitív** DVC-futtatás is, különben a DVC-integrációs teszt sosem fut le. Ezt a QA runner visszatérési kódja és összefoglalója egyértelműen jelezze. A `pytest.ini` meglévő tartalmát megtartva lehet marker-regisztráció:

```ini
[pytest]
markers =
    dvc: DVC CLI/remote hozzáférést igénylő integrációs tesztek
```

A fenti szakaszt **be kell illeszteni a meglévő `pytest.ini` konfigurációba**, nem szabad a teljes fájlt felülírni. A `tests/integration/test_training_registry_integration.py` **szintetikus**, tehát attól, hogy nincs DVC, még mindig futnia kell. Érdemes helyi/ideiglenes DVC remote-ot használni legalább egy CI-integációhoz; közös remote esetén CI-secrets és minimális jogosultság szükséges. Egy „zöld, DVC nélkül futott pytest” **nem bizonyítja a DVC-integráció sikerét**.

Javasolt későbbi PowerShell futtatás:

```powershell
python -m dvc pull
python -m pytest tests/integration/test_dvc_training_integration.py -m dvc -v
python -m coverage erase
python -m coverage run -m pytest tests/ -m "not dvc" -q
python -m coverage report -m --include="src/*" --fail-under=95
```

Ha a két tier lefedettségét összeadjuk, a második tesztfuttatást `python -m coverage run --append ...` formában kell indítani ugyanazon checkouton. A DVC teszt állapota (futott / kihagyott / hibás), a tesztek száma és az elérhető adatverzió kerüljön külön CI riportba. A `.github/workflows/python-qa.yml` fájlban vizsgáljuk meg a jelenlegi install és tesztszűrés lépéseket; adjuk hozzá a DVC dependencyt, biztonságos hitelesítést, `dvc pull`-t és a dedikált integrációs jobot. A 95%-os coverage a `src/` kódra vonatkozik.

## 8. Máté API-s handoffja – mit kap meg és mit kell bekötnie?

**Fontos átadás:** az új 3.2-es orchestration már egy lépésben tud tanítani, regisztrálni, összehasonlítani, naplózni és championt váltani. Ha Máté automatikus modellkiválasztást szeretne a `/train` végpontból, **ne közvetlenül** a 3.1-es `run_training_experiment()` függvényt hívja, hanem a **`train_register_and_select()`** függvényt. Ez belül újrahasználja a 3.1-et, tehát nem duplikáljuk a traininget és MLflow runokat.

### 8.1 Pontos modul/feladat átadás

| API-funkció | Amit megkap | Amit Máténak kell elkészítenie |
|---|---|---|
| Training adatbetöltés | `data/loader.py -> load_raw_data(path)` | Jogosult, szerveroldalon meghatározott DVC-visszaállított dataset elérése; nem szabad kliens által megadott tetszőleges fájlutat engedni. |
| Adatstruktúra és preprocessing | `data/preprocessing.py -> DataProcessor.rename_columns`, `split_features_target` | A 6 bemeneti változó / címke pontos szétválasztása; API bemenetnél Pydantic validáció. |
| Kezdeti split | `data/splitting.py -> create_initial_data_splits` | Csak az 5000 soros alapdatasettel hívja; bővülő adatra külön retraining pipeline. |
| Training + MLflow + Registry | `models/training_registry.py -> train_register_and_select` | `/train` service bekötése `DataSplits`, `model_type`, megbízható tracking URI, threshold, lineage paraméterekkel. |
| Adatverzió | `models/data_lineage.py -> DatasetLineage` | Beni DVC hashének, Git commitjának és splitverziójának átadása a training hívásnál. |
| Training response | `TrainingRegistryResult`, `SelectionResult` | `run_id`, regisztrált verzió, döntés, **tényleges** promotion státusz és aktuális champion a válaszban. |
| Champion visszatöltése | `models/model_registry.py -> load_champion_model`, `get_champion_version` | Csak megbízható Registry-ből töltsön be modelleket; cache refresh / explicit verziókezelés a kérésekhez. |
| `/predict` | A tanított sklearn Pipeline `predict_proba()` / `predict()` | Pontosan 6 nyers normalizált feature-ből DataFrame; model betöltése; megbeszélt threshold szerinti predikció; **tilos újból fit-elni a preprocessort**. |
| Predikciós naplózás | **Nincs kész 3.2 implementáció** | Request ID, időbélyeg, szolgáltatott modellverzió, threshold, valószínűség, döntés, hiba/státusz; szükség szerint MLflow JSON/CSV artifact, retention/adatvédelem. |
| FastAPI fájlok | `api/main.py`, `routes.py`, `schemas.py`, `mock_service.py` **elérési utak merge-nél ellenőrizendők** | Mock service helyett valódi szolgáltatás; Swagger/Postman, jó válaszkódok, tesztek. |
| API QA | A meglévő SQLite MLflow integration tesztek mintája | FastAPI TestClient: hiányzó champion, érvénytelen input, nincs DVC adat, MLflow unavailable, 429/5xx konvenció szerint, model cache, predikció audit. |

### 8.2 Javasolt `/train` szerződés (még nem létező endpoint)

```json
POST /train
{
  "model_type": "xgboost",
  "dataset_version": "server-side-approved-version"
}
```

Javasolt sikeres válasz (csak szemléltetés):

```json
{
  "run_id": "<valos_run_id>",
  "registered_model_name": "predictive-maintenance-model",
  "registered_version": "<valos_verzio>",
  "selection_reason": "higher_f1",
  "promotion_status": "completed",
  "champion_version": "<Registrybol_kiolvasott_verzio>"
}
```

Máté ne csak a `selection.promoted` értéket vizsgálja, mert ez dry-runban jóváhagyást jelent; a valódi `promotion_status` az MLflow runból olvasható, a champion alias pedig a Registry-ből. Ha a challenger elutasított, a response ne tévessze össze az újonnan létrejött modellverziót az aktuális championnal. Szinkron demó esetén visszaadható HTTP 200; hosszú training vagy container timeout esetén jobb a HTTP `202 Accepted` + job ID + status endpoint. A kliens ne adhasson meg tetszőleges tracking URI-t vagy szerverfájlutat.

### 8.3 Javasolt `/predict` szerződés (még nem létező endpoint)

```json
POST /predict
{
  "type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 120
}
```

```json
{
  "prediction": 0,
  "failure_probability": 0.12,
  "model_version": "<tenylegesen_kiszolgalt_verzio>",
  "threshold": 0.5
}
```

Hat bemeneti mező, nincs `target`, `failure_type`, `udi` vagy `product_id`. A Pydantic sémának helyes típusokat és mértékegységeket kell előírnia; a Python szolgáltatás DataFrame-mé alakítja, amelyet a teljes betanított Pipeline fogad. Amennyiben a predikciós küszöb megváltozik, ne feltételezzük, hogy a sklearn `predict()` ugyanazt használja: `predict_proba()` alapján `probability >= threshold` szabállyal kell osztályozni. **A válaszban szereplő `model_version` a ténylegesen betöltött (esetleg cache-elt) verzió legyen.** Hiányzó champion esetén szerveroldali szolgáltatáshiba (például 503), hibás bemenetnél 422 indokolt; az API-nak kezelnie kell a Registry/MLflow hibákat. A `cloudpickle` miatt kizárólag megbízható modellek töltendők be.

### 8.4 Mit kell konkrétan átadnunk Máténak?

1. Publikus függvények és importútvonalak: `load_raw_data`, `DataProcessor`, `create_initial_data_splits`, `train_register_and_select`, `load_champion_model`, `get_champion_version`.
2. `DataSplits`, `DatasetLineage`, `TrainingRegistryResult`, `SelectionResult` mezői és a championváltás **F1 > és recall >=0,70** szabálya (`threshold=0,5`).
3. Közös MLflow tracking/Registry URI, modellnév `predictive-maintenance-model`, alias `champion`, artifact hozzáférés, változók/secretek.
4. A DVC adatelérési út és hash/Git SHA/splitverzió kiolvasásának Benivel egyeztetett módja.
5. MLflow auditkulcsok és státuszok, különösen `verification_required` jelentése.
6. Egységes `/train` és `/predict` input/output és exception szerződés; hosszabb training esetén async jobstratégia.
7. API unit/integrációs tesztelés, Swagger/Postman bemutató, predikciós auditesemények és JSON/CSV artifact bizonyítéka.

## 9. Lezárási ellenőrzőlista

- [x] 3.2 Model Registry és `champion` alias; model betöltés.
- [x] F1 + legalább 70% recall alapú automatikus döntés; dry-run kiértékelés.
- [x] Training -> Registry -> audit -> ellenőrzött promotion összekötése.
- [x] Opcionális `DatasetLineage` és JSON artifact / run tagek.
- [x] A fejlesztő visszajelzése szerint a legutóbbi pytest, Flake8, Pyright és `src/` >=95% coverage ellenőrzés zöld.
- [ ] PR feltöltés, GitHub Actions ellenőrzések, review és merge a `dev` branchre – jelenleg **nem ellenőrzött**.
- [ ] Beni tényleges DVC remote-ja, tracked data útvonalai és splitverziója – egyeztetendő.
- [ ] DVC adapter + CLI + valódi DVC end-to-end integrációs teszt – **közös integrációs feladat**.
- [ ] QA Python futtató és CI DVC-specifikus teszt tierje – **még módosítandó**.
- [ ] Máté `/train` és `/predict` végpontjai, predikciós audit és API-tesztek – **még bekötendő**.
- [ ] Éles shared MLflow, jogosultságok, concurrent promotion és rollback/reconciliation – későbbi üzemeltetési feladat.

---

**Összegzés:** a **3.2 fejlesztési rész a jelenlegi scope szerint kész és lokálisan ellenőrzött**, ezért dokumentálható és PR-ra előkészíthető. A **teljes DVC -> preprocessing -> 3.1 -> 3.2 -> FastAPI folyamat viszont még nincs bekötve**; az ehhez szükséges fájlokat, szerződéseket, QA/CI teendőket és felelősségi határokat ez a dokumentáció részletesen rögzíti.
