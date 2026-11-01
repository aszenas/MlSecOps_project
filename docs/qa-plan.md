# Tesztelési és minőségbiztosítási terv

## 1. Cél

A QA-folyamat célja annak automatikus ellenőrzése, hogy:

- a preprocessing, modell, monitoring és újratanítás megfelelően működik;
- a `/train` és `/predict` REST API-végpontok működnek;
- a komponensek integráltan is együttműködnek;
- a Python-kód megfelel a coverage-, kódstílus- és típusellenőrzési követelményeknek;
- a Docker image-ek elkészíthetők, és a rendszer Docker Compose-zal elindítható;
- az ellenőrzések lokálisan és a GitHub Actions CI-ban is reprodukálhatók.

## Git workflow

A projekt az alábbi branch-stratégiát alkalmazza:

- `main`: stabil projektverzió;
- `dev`: közös fejlesztési ág;
- `feature/*`: önálló fejlesztési feladatok;
- `fix/*`: hibajavítások.

Minden új `feature/*` vagy `fix/*` branch a friss `dev` branchből indul:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/branch-name
```

A fejlesztési folyamat szabályai:

- közvetlen commit a `main` és `dev` branchekre nem engedélyezett;
- a feature és fix branchek Pull Requestjeinek base ága a `dev`;
- minden Pull Requesthez másik csapattagtól review szükséges;
- merge commitot kell használni;
- squash merge nem használható;
- a feature branch commit historyját meg kell őrizni.

## 2. Eszközök

| Eszköz | Feladat |
|---|---|
| Pytest | Unit-, API-, integrációs és smoke tesztek futtatása |
| Coverage.py | Az `src/` és `tests/` Python-kód lefedettségének mérése |
| Flake8 | PEP8-, kódstílus- és sorhossz-ellenőrzés |
| Pyright | Az `src/` statikus típusellenőrzése |
| pre-commit | Gyors ellenőrzések futtatása commit előtt |
| GitHub Actions | A teljes CI-folyamat automatikus futtatása |
| Docker / Docker Compose | Image build, konfiguráció-validáció és rendszerindítás |

## 3. Vizsgálati hatókör

| Terület | Pytest | Kötelező 95% coverage | Flake8 | Pyright |
|---|---:|---:|---:|---:|
| `src/` | Igen | Igen | Igen | Igen |
| `tests/` | Igen | Igen | Igen | Nem kötelező |
| Streamlit-kód | Ahol értelmezhető | Nem, ha az `src/` könyvtáron kívül van | Igen | Nem kötelező |
| Airflow-kód | Ha megvalósul | Nem, ha az `src/` könyvtáron kívül van | Igen | Nem kötelező |
| Docker és Compose | Build- és smoke teszt | Nem | Nem | Nem |

A Streamlit- és Airflow-kód csak vékony felületi réteg lehet. A preprocessing, modell, monitoring és újratanítás üzleti logikája az `src/` könyvtárba kerüljön. Az `src/` alatt található üzleti logika minden esetben része a kötelező coverage-mérésnek.

## 4. Tesztstratégia

### 4.1 Unit tesztek

A unit tesztek gyorsan, izoláltan és determinisztikusan ellenőrzik az egyes függvényeket és osztályokat.

#### Preprocessing

Ellenőrizendő:

- hiányzó értékek kezelése;
- kötelező oszlopok megléte;
- kategorikus és numerikus változók transzformációja;
- feature-ök és célváltozó szétválasztása;
- output típusa, mérete és oszlopsorrendje;
- üres vagy hibás input kezelése;
- ismeretlen kategóriák kezelése;
- reprodukálható train/test split.

#### Modell

Ellenőrizendő:

- a modell létrehozható és tanítható;
- a predikció elemszáma megfelel az input elemszámának;
- a predikció típusa megfelelő;
- a modell menthető és visszatölthető;
- azonos modell és input reprodukálható eredményt ad;
- a paraméterek és metrikák megfelelő formátumúak;
- a hibás input kontrollált hibát eredményez.

A teszteknek nem egy konkrét magas modelljósági értéket, hanem a tanítási és predikciós folyamat helyes működését kell ellenőrizniük.

#### Monitoring

Ellenőrizendő:

- reference és current dataset validálása;
- hiányzó vagy eltérő oszlopok kezelése;
- drift/no-drift döntési logika;
- üres vagy túl kis current dataset kezelése;
- monitoring eredmény és riport előállítása;
- prediction logok monitoring inputtá alakítása.

#### Újratanítás

Ellenőrizendő:

- retraining trigger igaz és hamis esetben;
- megfelelő adatok kiválasztása;
- candidate modell létrehozása;
- aktuális és candidate modell összehasonlítása;
- rosszabb modell automatikus aktiválásának megakadályozása;
- sikeres modellváltás;
- sikertelen újratanítás kontrollált kezelése.

Az új modell elfogadási szabályát a fejlesztés előtt egyértelműen definiálni kell.

### 4.2 API-tesztek

#### `POST /predict`

Ellenőrizendő:

- érvényes kérés sikeres válasza;
- response schema és prediction típusa;
- prediction request és response naplózása;
- a prediction artifact sikeres mentése MLflow-ba;
- az artifact egyedi predikcióazonosítót vagy időbélyeget tartalmaz;
- MLflow-hiba esetén az API kontrollált viselkedése;
- hiányzó mezők;
- hibás adattípus;
- üres kérés;
- nem elérhető modell;
- belső hiba kontrollált kezelése;
- traceback vagy érzékeny információ nem kerül a válaszba.

#### `POST /train`

Ellenőrizendő:

- érvényes kérés kezelése;
- training folyamat elindítása;
- MLflow run és artifact létrejötte;
- modellregisztráció meghívása;
- hibás vagy hiányzó adatforrás;
- sikertelen training kontrollált kezelése;
- response schema és HTTP-státuszkód.

Az API-tesztekben a külső függőségek mockolhatók. Nem szükséges minden API-teszthez valódi MLflow szervert indítani.

### 4.3 Integrációs tesztek

Az integrációs tesztek a komponensek együttműködését ellenőrzik:

1. preprocessing → modell tanítása → predikció;
2. training → MLflow run → metric, parameter és artifact;
3. modellmentés → modellbetöltés → predikció;
4. `/predict` → request és prediction naplózása;
5. `/predict` → MLflow run vagy artifact → request és prediction visszaolvashatósága;
6. prediction log → monitoring input → drift report;
7. monitoring eredmény → retraining trigger → új modell.

A tesztek lehetőség szerint ideiglenes könyvtárat, kis tesztadatot és lokális tesztkonfigurációt használjanak.

### 4.4 Docker Compose smoke teszt

A smoke teszt ellenőrzi, hogy:

- a Compose-konfiguráció érvényes;
- az image-ek sikeresen buildelhetők;
- a szolgáltatások elindulnak;
- az API health endpoint válaszol;
- legalább egy minimális API-kérés végrehajtható;
- a szolgáltatások teszt után leállíthatók.

Javasolt health endpoint:

```http
GET /health
```

Elvárt válasz:

```json
{
  "status": "ok"
}
```

## 5. Kötelező minőségkapuk

| Ellenőrzés | Hatókör | Elfogadási feltétel |
|---|---|---:|
| Pytest | minden automatizált teszt | minden teszt sikeres |
| Line coverage | `src/` és `tests/` együtt | legalább 95% |
| Flake8 | minden saját Python-kód | belső cél: 0 hiba |
| Maximális sorhossz | minden saját Python-kód | 120 karakter |
| Pyright standard | `src/` | 0 hiba |
| Docker Compose config | Compose-konfiguráció | érvényes |
| Docker build | minden szükséges image | sikeres |
| Smoke test | elindított rendszer | sikeres |

Az oktatói maximum 1% PEP8-hiba helyett a projekt belső minőségkapuja 0 Flake8-hiba.

## 6. Coverage-szabályok

A kötelező coverage-mérés az `src/` és `tests/` könyvtárak Python-kódjára együttesen vonatkozik. A különálló Streamlit- és Airflow-réteg csak külön csapatdöntés alapján kerüljön a közös coverage-scope-ba.

Kötelező minimum:

```text
Line coverage >= 95%
```

A branch coverage mérése ajánlott, de csak külön csapatdöntés alapján legyen kötelező minőségkapu.

Javasolt futtatás:

```bash
coverage erase
coverage run --source=src,tests -m pytest
coverage report --show-missing --fail-under=95
```

Fejlesztői HTML-riport:

```bash
coverage html
```

A CI-ben a coverage-riport artifactként megőrizhető.

## 7. Flake8- és Pyright-szabályok

### Flake8

A Flake8 minden saját Python-kódot vizsgáljon, beleértve:

- `src/`;
- `tests/`;
- Streamlit-kód;
- Airflow-kód, ha készül;
- saját Python utility scriptek.

Javasolt futtatás:

```bash
flake8 src tests dashboard --max-line-length=120
```

A könyvtárlistát a végleges repository-struktúrához kell igazítani.

### Pyright

A Pyright standard módban kizárólag az `src/` könyvtárat vizsgálja.

```bash
pyright src
```

Elvárások:

- 0 Pyright-hiba;
- paraméter- és visszatérési típusok használata;
- indokolatlan `Any` kerülése;
- `type: ignore` csak indoklással alkalmazható.

## 8. Kivételek kezelése

Az alábbi kivételek csak technikailag indokolt esetben használhatók:

- `# noqa: <hibakód>`;
- `# pragma: no cover`;
- `# type: ignore[...]`;
- coverage `omit`;
- Flake8 `per-file-ignores`.

Minden kivételnél:

1. csak a szükséges ellenőrzés nyomható el;
2. konkrét hibakódot kell megadni, ahol lehetséges;
3. a kód mellett rövid indoklás szükséges;
4. Pull Request review szükséges;
5. teljes üzleti logikai modul nem zárható ki;
6. tesztelhető kód nem zárható ki kizárólag a 95% elérése érdekében.

A kivételek keresésére használható:

```bash
grep -RInE "noqa|pragma: no cover|type: ignore|omit" src tests
```

## 9. Pre-commit stratégia

A pre-commit célja a gyors, lokális visszajelzés.

Kötelező ellenőrzések:

- trailing whitespace;
- fájlvégi newline;
- YAML- és TOML-validáció;
- merge conflict markerek;
- túl nagy fájlok;
- Flake8;
- Pyright;
- gyors unit tesztcsomag.

Manuális futtatás:

```bash
pre-commit run --all-files
```

A teljes integrációs tesztcsomag, a legalább 95%-os coverage-ellenőrzés, a Docker build és a Docker Compose smoke teszt minden push és Pull Request esetén kötelezően fusson GitHub Actionsben, de ne fusson minden lokális commit előtt.

## 10. GitHub Actions CI

A CI push és Pull Request esetén fusson.

### Static quality job

- Flake8;
- Pyright standard;
- konfigurációs fájlok ellenőrzése.

### Tests and coverage job

- unit tesztek;
- API-tesztek;
- integrációs tesztek;
- minimum 95% coverage;
- coverage-riport mentése.

### Container verification job

- Docker Compose konfiguráció-validáció;
- Docker image build;
- szolgáltatások indítása;
- health check és smoke teszt;
- szolgáltatások leállítása és takarítása.

Bármely kötelező job hibája esetén a CI sikertelen. A feature branchek Pull Requestjeinek base ága a `dev`. A `dev` branchre csak sikeres CI, review és merge commit útján kerülhet változás. Közvetlen commit a `main` és `dev` branchekre nem engedélyezett.

## 11. Függőségkezelés és reprodukálhatóság

- A használt Python-verziót rögzíteni kell.
- A közvetlen projektfüggőségek pontos verzióval szerepeljenek a választott dependency-fájlban, például a `requirements.txt` fájlban.
- A lokális környezet, a CI és a Docker build ugyanabból a dependency-forrásból telepítsen.
- A dependency-fájl módosítása Pull Request review-hoz kötött.
- A Docker build ugyanazokat a rögzített dependency-verziókat használja.
- A tesztekhez használt random seedet rögzíteni kell, ahol releváns.

## 12. Javasolt helyi parancsok

### Unit tesztek

```bash
python -m pytest tests/unit -v
```

### API-tesztek

```bash
python -m pytest tests/api -v
```

### Integrációs tesztek

```bash
python -m pytest tests/integration -v
```

### Teljes tesztcsomag és coverage

```bash
coverage erase
coverage run --source=src,tests -m pytest
coverage report --show-missing --fail-under=95
```

### Flake8

```bash
flake8 src tests dashboard --max-line-length=120
```

### Pyright

```bash
pyright src
```

### Pre-commit

```bash
pre-commit run --all-files
```

### Docker Compose validáció és build

```bash
docker compose config --quiet
docker compose build
```

### Rendszerindítás és leállítás

```bash
docker compose up -d
docker compose ps
docker compose down --volumes
```

## 13. Repository-elhelyezés és karbantartás

A QA-terv javasolt helye:

```text
docs/qa-plan.md
```

A `README.md` tartalmazzon rövid összefoglalót és hivatkozást erre a dokumentumra.

A QA-tervet módosítani kell, ha:

- új komponens kerül a projektbe;
- változik a tesztstratégia;
- változik valamelyik minőségkapu;
- új kivétel szükséges;
- változik a dependency-kezelés vagy a CI-folyamat.
