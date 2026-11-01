# Végső repository-checklist a beadás előtt

> A checklist célja a repository, az automatizált ellenőrzések, a dokumentáció és a beadási csomag utolsó, tételes átvizsgálása.

## 1. Git és repository

- [ ] A repository elérhető a megadott GitHub-linken.
- [ ] Az oktató olvasási jogosultsággal hozzáfér a repositoryhoz.
- [ ] Privát repository esetén a `github@deirf.hu` felhasználó hozzáférése be van állítva.
- [ ] Ha az oktató nem fér hozzá, a teljes projekt külön ZIP-ben is elkészült, a `.git/` mappával együtt.
- [ ] A `main` branch tiszta, stabil és működőképes.
- [ ] A `dev` branch tartalmazza a review-zott, integrált fejlesztéseket.
- [ ] Nincs nem commitolt módosítás: `git status` tiszta eredményt ad.
- [ ] Nincsenek véletlenül commitolt titkok, tokenek, jelszavak vagy `.env` fájlok.
- [ ] A `.gitignore` kizárja a lokális, generált és érzékeny fájlokat.
- [ ] Minden csoporttag elérte a minimum 50 saját commitot.
- [ ] A commitüzenetek következetesek és értelmezhetők.
- [ ] A feature branchek és Pull Requestek dokumentáltak.
- [ ] A `main` és `dev` branchekre nem került közvetlen commit; minden változás Pull Request és merge útján került be.
- [ ] Minden `feature/*` és `fix/*` branch a friss `dev` branchből indult.
- [ ] A feature és fix branchek Pull Requestjeinek base ága a `dev`.
- [ ] A Pull Requestek merge commit használatával kerültek be.
- [ ] Squash merge nem került alkalmazásra.
- [ ] A feature branchek commit historyja megmaradt.
- [ ] Csoportmunka esetén a review-k, review note-ok és merge-ek visszakövethetők.
- [ ] A review-k száma megfelel a használt feature branchek számának.
- [ ] Nincsenek szükségtelen, nyitva maradt branchek vagy Pull Requestek.

## 2. Repository-struktúra

- [ ] Az alkalmazás üzleti logikája az `src/` alatt található.
- [ ] A tesztek a `tests/` könyvtárban, típus szerint elkülönítve találhatók.
- [ ] A Streamlit-réteg vékony, az üzleti logikát az `src/` moduljaiból hívja.
- [ ] Az Airflow DAG-ok vékonyak, ha az extra funkció megvalósult.
- [ ] A dokumentáció a `README.md` és szükség esetén a `docs/` alatt található.
- [ ] A screenshotok egységes könyvtárban, például `docs/screenshots/` alatt találhatók.
- [ ] A repositoryban nincs felesleges nagy fájl, cache, lokális adatbázis vagy futási maradvány.
- [ ] A nyers vagy nagy dataset nem került szabálytalanul Gitbe.
- [ ] A dataset DVC-vel vagy más megfelelő adatverzió-kezelővel követhető.

Javasolt szerkezet:

```text
project/
├── .github/workflows/
├── data/
├── docs/
│   ├── screenshots/
│   └── qa-plan.md
├── src/
├── tests/
│   ├── unit/
│   ├── api/
│   └── integration/
├── dashboard/
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 3. Dataset és preprocessing

- [ ] A korábban választott dataset azonosítható és dokumentált.
- [ ] A célváltozó egyértelműen meg van nevezve.
- [ ] A preprocessing lépései dokumentáltak.
- [ ] A hiányzó értékek, adattípusok és kategorikus változók kezelése reprodukálható.
- [ ] A preprocessing nem támaszkodik manuális, nem dokumentált lépésekre.
- [ ] Az előfeldolgozás utáni dataset elkészült.
- [ ] Az előfeldolgozott dataset szerepel a Moodle-be feltöltendő ZIP-ben.
- [ ] A training és prediction ugyanazt a preprocessing-logikát használja.
- [ ] A random seedek rögzítettek, ahol relevánsak.

## 4. Modell és újratanítás

- [ ] A modell tanítható a dokumentált folyamat szerint.
- [ ] A modell típusa és a választás indoka szerepel a README-ben.
- [ ] A tanítás paraméterei és metrikái MLflow-ba kerülnek.
- [ ] A modell artifactként mentésre kerül.
- [ ] A modell bekerül az MLflow Model Registrybe.
- [ ] A regisztrált modell verziója visszakövethető.
- [ ] A candidate és az aktuális modell összehasonlításának szabálya dokumentált.
- [ ] Rosszabb candidate modell nem kerül automatikusan aktív használatba.
- [ ] Az újratanítás sikertelen futása kontrolláltan kezelhető.
- [ ] Az újratanítási folyamat automatizálása bemutatható.

## 5. MLflow

- [ ] Az MLflow tracking server elindul a rendszerrel.
- [ ] A training futás paraméterei naplózódnak.
- [ ] A training metrikái naplózódnak.
- [ ] A training artifactjai naplózódnak.
- [ ] A preprocessing vagy pipeline szükséges objektumai mentésre kerülnek.
- [ ] A `/predict` kérések és válaszok naplózódnak.
- [ ] A `/predict` eredménye artifactként is mentésre kerül.
- [ ] A prediction artifact egyedi azonosítóval vagy időbélyeggel visszakereshető.
- [ ] A model registry és a modellverziók megtekinthetők.
- [ ] Az MLflow-adatok Docker újraindítás után sem vesznek el, ha perzisztencia szükséges.
- [ ] Az MLflow dashboardról megfelelő screenshot készült.

## 6. REST API

- [ ] A REST API elindul a Docker Compose rendszerrel.
- [ ] A `POST /train` végpont működik.
- [ ] A `POST /predict` végpont működik.
- [ ] Van egyszerű `GET /health` vagy ezzel egyenértékű health endpoint.
- [ ] A request és response sémák dokumentáltak.
- [ ] A kötelező és opcionális mezők egyértelműek.
- [ ] A hibás adattípus kontrollált klienshibát eredményez.
- [ ] A hiányzó mező kontrollált klienshibát eredményez.
- [ ] Nem elérhető modell esetén kontrollált hiba keletkezik.
- [ ] A válasz nem tartalmaz érzékeny traceback-információt.
- [ ] A Swagger, Postman vagy Python-alapú teszt bizonyítható.
- [ ] A `/train` működéséről screenshot készült.
- [ ] A `/predict` működéséről screenshot készült.

## 7. Monitoring, EvidentlyAI és Streamlit

- [ ] A reference dataset meghatározása dokumentált.
- [ ] A current dataset a naplózott prediction inputokból előállítható.
- [ ] Az Evidently report elkészül.
- [ ] A drift/no-drift eredmény értelmezhető.
- [ ] A hiányzó oszlopok és sémaeltérések kontrolláltan kezelhetők.
- [ ] A Streamlit dashboard elindul a rendszerrel.
- [ ] A Streamlit dashboard megjeleníti az Evidently eredményét.
- [ ] A Streamlit-kód csak vékony megjelenítési réteg.
- [ ] A Streamlit dashboardról, rajta az Evidently reporttal, screenshot készült.
- [ ] Ha Airflow készült, az Airflow dashboardról is készült screenshot.

## 8. Automatikus tesztek

- [ ] A preprocessing unit tesztjei elkészültek.
- [ ] A modell unit tesztjei elkészültek.
- [ ] A monitoring unit tesztjei elkészültek.
- [ ] Az újratanítás unit tesztjei elkészültek.
- [ ] A `/train` API-tesztjei elkészültek.
- [ ] A `/predict` API-tesztjei elkészültek.
- [ ] Pozitív és negatív API-tesztesetek is vannak.
- [ ] A preprocessing → training → prediction integráció tesztelt.
- [ ] A training → MLflow integráció tesztelt.
- [ ] A model save → load → predict folyamat tesztelt.
- [ ] A predict → logging → monitoring folyamat tesztelt.
- [ ] A monitoring → retraining folyamat tesztelt.
- [ ] A tesztek nem függenek indokolatlanul külső hálózattól.
- [ ] A tesztek kis, kontrollált tesztadatokat használnak.
- [ ] A teljes tesztcsomag lokálisan sikeresen lefut.

```bash
python -m pytest -v
```

## 9. Coverage

- [ ] A coverage az `src/` és `tests/` Python-kódját egyaránt méri.
- [ ] A teljes line coverage legalább 95%.
- [ ] A CI `--fail-under=95` minőségkaput alkalmaz.
- [ ] A coverage report megmutatja a hiányzó sorokat.
- [ ] Régi coverage-adat nem torzítja az eredményt.
- [ ] A HTML coverage report elkészíthető.
- [ ] A coverage report CI artifactként elmenthető.
- [ ] A `# pragma: no cover` csak közvetlen indoklással szerepel.
- [ ] Az `omit` csak közvetlen indoklással szerepel.
- [ ] Nincs tesztelhető kód kizárva kizárólag a 95% elérése érdekében.

```bash
coverage erase
coverage run --source=src,tests -m pytest
coverage report --show-missing --fail-under=95
coverage html
```

## 10. Flake8 és PEP8

- [ ] A Flake8 minden saját Python-kódot vizsgál.
- [ ] A maximális sorhossz 120 karakter.
- [ ] A belső minőségkapu 0 Flake8-hiba.
- [ ] A `# noqa` csak konkrét hibakóddal és indoklással szerepel.
- [ ] A `per-file-ignores` használata dokumentált és indokolt.
- [ ] A végső Flake8-futtatás sikeres.

```bash
flake8 src tests dashboard --max-line-length=120
```

> A könyvtárlistát a tényleges repository-struktúrához kell igazítani.

## 11. Pyright és type hinting

- [ ] A Pyright standard módban fut.
- [ ] A Pyright kizárólag az `src/` könyvtárat vizsgálja.
- [ ] A Pyright eredménye 0 hiba.
- [ ] A függvényparaméterek type hintet használnak.
- [ ] A visszatérési értékek type hintet használnak.
- [ ] Az indokolatlan `Any` használat megszűnt.
- [ ] A `# type: ignore[...]` csak közvetlen indoklással szerepel.
- [ ] Ha a strict extra megvalósult, a megadott „szelíd strict” konfiguráció 0 hibával fut.

```bash
pyright src
```

## 12. pre-commit

- [ ] A `.pre-commit-config.yaml` verziókövetett.
- [ ] A pre-commit telepítési és futtatási módja szerepel a README-ben.
- [ ] A trailing whitespace ellenőrzése be van állítva.
- [ ] A fájlvégi newline ellenőrzése be van állítva.
- [ ] A YAML- és TOML-fájlok validálása be van állítva.
- [ ] A merge conflict markerek ellenőrzése be van állítva.
- [ ] A túl nagy fájlok ellenőrzése be van állítva.
- [ ] A Flake8 hook működik.
- [ ] A Pyright hook működik.
- [ ] A gyors unit tesztcsomag hookként fut.
- [ ] A hookok minden verziókövetett fájlra sikeresen lefutnak.
- [ ] A coverage és Docker build automatizált CI-ellenőrzése dokumentált.

```bash
pre-commit run --all-files
```

## 13. GitHub Actions CI

- [ ] A workflow-fájl a `.github/workflows/` könyvtárban található.
- [ ] A CI push eseményre fut.
- [ ] A CI Pull Request eseményre fut.
- [ ] A Python-verzió explicit módon be van állítva.
- [ ] A függőségek telepítése reprodukálható.
- [ ] A Flake8 job sikeres.
- [ ] A Pyright job sikeres.
- [ ] A unit tesztek sikeresek.
- [ ] Az API-tesztek sikeresek.
- [ ] Az integrációs tesztek sikeresek.
- [ ] A coverage legalább 95%.
- [ ] A coverage report mentésre kerül.
- [ ] A Docker Compose konfiguráció-validáció sikeres.
- [ ] A Docker image build sikeres.
- [ ] A szolgáltatások elindulnak.
- [ ] A health check sikeres.
- [ ] A smoke teszt sikeres.
- [ ] A konténerek hiba esetén is leállnak és kitakarításra kerülnek.
- [ ] A legutolsó `dev` branch CI-futás zöld.
- [ ] A `main` branch legutolsó CI-futása zöld, ha már történt release vagy integráció a `main` ágba.
- [ ] A kötelező CI checkek branch protectionnel védik a `dev` branchet.
- [ ] A `main` branch közvetlen módosítása branch protectionnel tiltott, ha ez be van állítva.
- [ ] A feature branchek Pull Requestjeinek base ága a `dev`.
- [ ] Sikertelen kötelező CI check esetén a Pull Request nem merge-elhető a `dev` ágba.

## 14. Docker és Docker Compose

- [ ] A Dockerfile-ok verziókövetettek.
- [ ] A Docker image-ek buildelhetők cache nélkül is.
- [ ] A `docker-compose.yml` vagy `compose.yml` érvényes.
- [ ] Az összes szükséges szolgáltatás szerepel a Compose-fájlban.
- [ ] Az API, MLflow és Streamlit szolgáltatás elindul.
- [ ] Az opcionális Airflow szolgáltatások elindulnak, ha megvalósultak.
- [ ] A portok dokumentáltak és nem ütköznek.
- [ ] A szükséges environment változók dokumentáltak.
- [ ] A szükséges volume-ok és perzisztens tárolók be vannak állítva.
- [ ] Az egész rendszer egyetlen paranccsal elindul.
- [ ] A rendszer tisztán leállítható.
- [ ] A Docker-konténerek működéséről screenshot készült.

```bash
docker compose config --quiet
docker compose build --no-cache
docker compose up -d
docker compose ps
docker compose down --volumes
```

## 15. Függőségek és reprodukálhatóság

- [ ] A használt Python-verzió dokumentált és rögzített.
- [ ] A `requirements.txt` vagy más dependency-fájl verziókövetett.
- [ ] A közvetlen függőségek pontos verzióval szerepelnek.
- [ ] A lokális környezet, CI és Docker ugyanabból a dependency-forrásból telepít.
- [ ] Friss, tiszta virtuális környezetben a telepítés sikeres.
- [ ] A dependency-fájl nem tartalmaz indokolatlan vagy nem használt csomagokat.
- [ ] A dependency-változtatások PR review-n mentek keresztül.

```bash
python -m pip install -r requirements.txt
```

## 16. DVC és adatverziózás

- [ ] A DVC konfiguráció verziókövetett.
- [ ] A szükséges `.dvc` fájlok vagy `dvc.yaml` megtalálhatók.
- [ ] A dataset megfelelő verziója visszaállítható.
- [ ] A DVC használata dokumentált a README-ben.
- [ ] A DVC remote használata dokumentált, ha van.
- [ ] A beadás nem függ olyan privát DVC remote-tól, amelyhez az oktató nem fér hozzá.
- [ ] Az előfeldolgozott dataset külön beadási fájlként is rendelkezésre áll.

## 17. README és dokumentáció

- [ ] A `README.md` a repository gyökerében található.
- [ ] A projekt célja és rövid leírása szerepel.
- [ ] A dataset és célváltozó szerepel.
- [ ] Az architektúra vagy komponenslista szerepel.
- [ ] A telepítési előfeltételek szerepelnek.
- [ ] A `docker compose up` indítási mód szerepel.
- [ ] A szolgáltatások URL-jei és portjai szerepelnek.
- [ ] A `/train` és `/predict` használati példái szerepelnek.
- [ ] A tesztek futtatási módja szerepel.
- [ ] A coverage, Flake8 és Pyright parancsai szerepelnek.
- [ ] A pre-commit telepítése és futtatása szerepel.
- [ ] A DVC használata szerepel.
- [ ] A screenshotok elérési útja szerepel.
- [ ] A QA-tervre mutató hivatkozás működik.
- [ ] A rendszer ismert korlátai és technikai adósságai szerepelnek.
- [ ] Az MI használatának célja és módja röviden dokumentált.
- [ ] A README utasításai friss, tiszta környezetből is követhetők.

Javasolt hivatkozás:

```markdown
## Minőségbiztosítás

A projekt tesztelési stratégiáját, kötelező minőségkapuit, coverage-szabályait és CI-ellenőrzéseit a [Tesztelési és minőségbiztosítási terv](docs/qa-plan.md) tartalmazza.
```

## 18. Screenshotok

- [ ] Az MLflow dashboard screenshotja elkészült.
- [ ] A REST API működésének screenshotja elkészült.
- [ ] A `/train` végpont tesztje látható.
- [ ] A `/predict` végpont tesztje látható.
- [ ] A Docker-konténerek működésének screenshotja elkészült.
- [ ] A Streamlit dashboard screenshotja elkészült.
- [ ] Az EvidentlyAI report látható a Streamlit screenshoton.
- [ ] Az Airflow dashboard screenshotja elkészült, ha az extra megvalósult.
- [ ] A screenshotok olvashatók és nem vágtak le lényeges információt.
- [ ] A screenshotok nem tartalmaznak jelszót, tokent vagy érzékeny adatot.
- [ ] A screenshotok repositorybeli helye szerepel az egyoldalas PDF-ben.

## 19. Biztonság és érzékeny adatok

- [ ] Nincs API-kulcs, jelszó vagy token a forráskódban.
- [ ] Nincs titok a Git historyban.
- [ ] A `.env` nincs commitolva.
- [ ] Van `.env.example`, ha environment változók szükségesek.
- [ ] A logok nem tartalmaznak szükségtelen személyes vagy érzékeny adatot.
- [ ] Az API hibaválaszai nem szivárogtatnak belső elérési utakat vagy stack trace-t.
- [ ] A Docker image nem tartalmaz szükségtelen titkokat vagy lokális fájlokat.
- [ ] A tesztadatok használata jogszerű és dokumentált.

## 20. Beadási PDF

- [ ] Az egyoldalas PDF elkészült.
- [ ] A név vagy nevek szerepelnek.
- [ ] A Neptun-kód vagy kódok szerepelnek.
- [ ] A beadás dátuma szerepel.
- [ ] A screenshotok repositorybeli helye szerepel.
- [ ] A GitHub repository linkje szerepel.
- [ ] Az oktatói hozzáférés biztosított.
- [ ] Az ismert hibák és hiányosságok szerepelnek.
- [ ] A technikai adósságok és megoldási kísérletek röviden szerepelnek.
- [ ] A PDF valóban egyoldalas és olvasható.

## 21. Moodle-be feltöltendő csomag

- [ ] Az előfeldolgozás utáni dataset fájl bekerült a beadási ZIP-be.
- [ ] Az egyoldalas PDF bekerült a beadási ZIP-be.
- [ ] Privát, nem hozzáférhető repository esetén a teljes projekt külön ZIP-je elkészült.
- [ ] A teljes projekt ZIP-je tartalmazza a `.git/` mappát.
- [ ] A ZIP-ek megnyithatók és nem sérültek.
- [ ] A fájlnevek egyértelműek és nem tartalmaznak problémás karaktereket.
- [ ] A ZIP-ben nincs felesleges virtuális környezet, cache vagy Docker-adat.
- [ ] A beadási csomagot egy külön könyvtárba kicsomagolva is ellenőriztétek.

## 22. Végső „clean machine” próba

- [ ] A repository friss klónozása sikeres.
- [ ] A dokumentált Python-verzióval létrehozható a környezet.
- [ ] A függőségek tisztán telepíthetők.
- [ ] A szükséges adatok visszaállíthatók vagy rendelkezésre állnak.
- [ ] A pre-commit hookok sikeresen lefutnak.
- [ ] A teljes tesztcsomag sikeresen lefut.
- [ ] A coverage legalább 95%.
- [ ] A Flake8 eredménye 0 hiba.
- [ ] A Pyright eredménye 0 hiba az `src/` alatt.
- [ ] A Docker image-ek sikeresen buildelhetők.
- [ ] A teljes rendszer `docker compose up` paranccsal elindul.
- [ ] A `/health`, `/train` és `/predict` végpont ellenőrizhető.
- [ ] Az MLflow UI elérhető.
- [ ] A Streamlit dashboard elérhető.
- [ ] Az Evidently report megjelenik.
- [ ] A rendszer tisztán leállítható.

## 23. Végső jóváhagyás

- [ ] A repository aktuális állapotát legalább egy másik csapattag átnézte.
- [ ] Minden kötelező CI check zöld.
- [ ] Minden kötelező screenshot elkészült.
- [ ] A README és a QA-terv a tényleges működést írja le.
- [ ] A beadási PDF és ZIP tartalma egymással konzisztens.
- [ ] A repository linkje működik és az oktató számára hozzáférhető.
- [ ] A csapat minden tagja el tudja magyarázni a saját és a közös implementáció működését.
- [ ] A beadandó a végleges, jóváhagyott és a `main` branchbe integrált verzióból készült, nem egy korábbi feature branchből vagy lokális állapotból.

---

## Ajánlott végső parancssor

A parancsokat a repository gyökeréből futtassátok:

```bash
git status
pre-commit run --all-files
coverage erase
coverage run --source=src,tests -m pytest
coverage report --show-missing --fail-under=95
flake8 src tests dashboard --max-line-length=120
pyright src
docker compose config --quiet
docker compose build --no-cache
docker compose up -d
docker compose ps
docker compose down --volumes
```

> A `dashboard` könyvtárnevet és az egyéb elérési utakat a tényleges repository-struktúrához kell igazítani.
