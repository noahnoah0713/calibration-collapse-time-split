# Beyond AUROC: 시간 분할 검증에서의 보정 붕괴

간독성(DILIrank)·심장독성(DICTrank) 예측 모델이 **시간 분할 검증**에서
판별력은 비교적 유지하면서 **확률 보정이 무너지는 현상**을 정량화하고,
그 원인을 유병률 이동·개념 이동·표본 크기로 분해한 연구의 코드와 결과.

본 저장소만으로 논문의 모든 수치·표·그림을 **처음부터 다시 생성**할 수 있다.

---

## 빠른 재현

```bash
pip install -r requirements.txt
bash rerun_all.sh
```

끝나면 다음이 새로 만들어진다.

| 산출물 | 위치 |
|---|---|
| 실험 결과 보고서 | `RESULTS_REPORT.md` |
| 투고용 LaTeX 표 | `manuscript/tables_main.tex`, `tables_supp.tex` |
| 그림 (PDF/PNG) | `figures/` |
| 모든 중간 결과 | `results/` |

`rerun_all.sh` 는 62개 스크립트를 65번 호출해 순서대로 돌린 뒤
(보고서와 verify9·10 은 셀 수를 머리글에 반영하려고 두 번 돈다), 마지막에 **보고서·표·그림
생성기가 읽는 결과 파일이 전부 만들어졌는지 자동 점검**하고 누락이 있으면 목록을
출력하며 0이 아닌 코드로 종료한다. 중간에 실패한 스크립트가 있으면 그 자리에서
멈춘다.

**주의 두 가지.**

- 이 스크립트는 `results/` 의 파일을 **덮어쓴다.** 논문에 실린 값을 남겨 두려면
  먼저 `cp -r results results_논문값` 로 사본을 떠 둘 것.
- 순열·대조군 반복이 대부분 200~2,000회라서 전체 실행에 **수 시간**이 걸린다.
  일부만 빠르게 보려면 해당 스크립트를 개별로 돌리고 반복 수를 직접 줄일 것
  (`--reps` 를 받는 스크립트가 14개 있다. 예:
  `python src/prevalence_control.py --reps 10`).

---

## 데이터

### 출처와 재배포

| 자료 | 출처 | 저장소 포함 |
|---|---|---|
| DILIrank | FDA Liver Toxicity Knowledge Base (LTKB) 공개 자료 | 가공본만 (`data/processed/`) |
| DICTrank | FDA Drug-Induced Cardiotoxicity Rank 공개 자료 | 가공본만 |
| 승인연도 | Drugs@FDA 벌크 파일 (Products / Submissions / Applications) | 미포함 (42MB) |
| ATC 분류 | KEGG DRUG–ATC 연계 | 매핑 결과만 (`kegg_atc_map.csv`) |

세 자료 모두 미국 FDA 가 공개한 것이고, 이 저장소에는 **원자료가 아니라
분석집합으로 가공한 파일**만 들어 있다. 원자료를 직접 받아 가공본을 다시
만드는 경로는 아래 "원자료에서 시작하기" 에 있다. KEGG 매핑은 조회 결과를
캐시한 것이며 KEGG 자체의 이용 약관을 따른다.


`data/processed/` 만 있으면 논문의 모든 수치가 재현된다. FDA 원자료
(`data/raw/`, 약 42MB)는 저장소에 포함하지 않았다.

`src/` 85개 중 원자료를 읽는 것은 5개다 — `map_approval_year.py`,
`dictrank_pipeline.py`(처리 데이터를 만드는 전처리),
`ev_years.py`, `check_years.py`, `check_labels.py`(원자료 대조 감사).
앞의 둘은 `rerun_all.sh` 가 부르지 않고, 뒤의 셋은 `data/raw/` 가 없으면
**자동으로 건너뛴다**(`run_raw`). 나머지 80개는 `data/processed/` 와
`results/` 만 쓴다.

그 결과 `results/year_evidence.csv` 와 `results/year_corrections.csv`
(§16.1 승인연도 수정 근거표)는 원자료 없이는 다시 만들 수 없다. 두 파일은
저장소에 포함해 두었고, 보고서는 그 사본을 읽는다.

주요 처리 파일

| 파일 | 내용 |
|---|---|
| `dilirank2_with_year.csv` | DILIrank 분석집합. 승인연도·SMILES·ikey14 포함 |
| `dictrank_processed.csv` | DICTrank 분석집합 |
| `*_prefilter.csv` | 구조 필터 적용 **전** 상태 (민감도 분석용) |
| `*_preconflict.csv` | 라벨 충돌 필터 적용 **전** 상태 |
| `kegg_atc_map.csv` | ATC 분류 매핑 |
| `smiles_failed.csv`, `year_unmatched.csv` | 매칭 실패 기록 (투명성 목적) |

### 원자료에서 시작하기

FDA Drugs@FDA 벌크 파일과 두 데이터셋 엑셀을 직접 받아 `data/raw/` 에 둔
뒤 다음 순서로 실행한다. 출처는 참고문헌 [21][22][32] 에 적혀 있다.

```
# DILIrank
python src/fetch_smiles.py            # PubChem 에서 SMILES 수집 (네트워크 필요)
python src/map_approval_year.py       # Drugs@FDA 원자료에서 승인연도 매칭
# DICTrank
python src/dictrank_pipeline.py       # 엑셀 + 원자료 + KEGG ATC 까지 한 번에
# 두 데이터셋 공통 필터
python src/apply_structure_filter.py
python src/apply_label_conflict_filter.py
# 이후는 평소대로
bash rerun_all.sh
```

두 필터는 원본을 `*_prefilter.csv` / `*_preconflict.csv` 로 **최초 1회만**
백업하고 이미 백업이 있으면 덮어쓰지 않는다. 그래서 `rerun_all.sh` 를 몇 번
돌려도 §16 의 민감도 분석이 쓰는 필터 적용 전 데이터가 보존된다.

---

## 재현성 설계

수치가 사람 손을 거치지 않는다.

- **생성기 방식** — 보고서와 표의 모든 숫자는 `src/make_report.py`,
  `src/make_tables.py` 가 `results/` 에서 직접 읽어 쓴다. 옮겨 적는
  단계가 없으므로 전사 오류가 구조적으로 불가능하다.
- **검증 4종**
  - `verify9.py` — 문서의 모든 숫자가 `results/` 의 어느 값에서 왔는지 추적
  - `verify10.py` — 반대 방향. `results/` 에서 기대 문자열을 만들어 문서와 셀 단위 대조
  - `verify11.py` — 모델×표현 격자 32칸이 두 스크립트에서 일치하는지
  - `verify_state.py` — 데이터·코드·결과의 지문(해시) 일치 확인
- **결정성** — 모든 난수는 시드를 고정한 `numpy.random.default_rng` 를
  쓴다. LightGBM 은 스레드 의존성을 제거했고, RandomForest 는 예측 누적을
  단일 스레드로 묶어 비트 단위로 재현된다 (`src/_shared.py`).

### 검증이 실제로 잡아낸 것

검증 장치는 통과하라고 만든 게 아니라 틀린 곳을 찾으라고 만들었다.
실제로 네 가지를 잡았고, 전부 잡히기 전까지는 모든 검사가 통과하고
있었다. 각 항목의 경위는 해당 소스의 주석에 날짜와 함께 적어 두었다.

| 찾은 것 | 어떻게 드러났나 | 영향 |
|---|---|---|
| 로짓 변환의 `float32` 누락 여덟 군데 | 같은 보정 기울기가 §10 과 §10.5 에서 다르게 찍혔다 | 두 절이 같은 양을 다르게 적고 있었다 |
| 재현 경로에 없던 손작업 두 개 | 전체 재실행 결과를 이전 백업과 칸 단위로 비교했다 | 받은 사람이 돌리면 §13.5 와 §33 이 재현되지 않았다 |
| 재개 캐시가 통째로 건너뛴 결과 파일 네 개 | 결과 파일이 그것을 만드는 코드보다 오래됐다 | 두 번의 재실행을 "완료" 로 통과하고 있었다 |
| 결과 파일에 쌓인 중복 행 | 완전히 같은 행을 전수로 셌다 | 다중비교 보정의 검정 개수가 57 대신 73 으로 부풀었다 |

앞의 두 개는 **추측으로 두 번 틀린 뒤에** 찾았다. 원인을 "lbfgs 가 실행
마다 다른 해에 닿는다" 로 적었다가 결정성 검사가 그것을 반증했고,
`src/diff_ladder.py` 가 두 경로를 한 프로세스에서 나란히 돌려 열다섯
단계를 비교한 뒤에야 로짓의 dtype 에서 갈린다는 것이 나왔다.

그래서 검사를 만들 때마다 **일부러 틀린 코드를 심어 잡히는지 먼저
확인한다.** 그 과정에서 쓸모없는 검사를 두 번 걸러냈다. 한 번은 시험
입력의 유병률이 너무 깔끔해서 반올림을 바꿔치기해도 결과가 같았고,
한 번은 파일 시각만 보다가 포맷터가 전체를 건드리자 오경보를 40건
넘게 냈다. 전자는 입력을 바꿨고 후자는 구문 트리 비교로 바꿨다.

### 아직 못 보는 것

- **그림** — `make_figures.py` 가 무엇을 그렸는지 기록을 남기지 않아,
  그림이 표와 다른 값을 그려도 검증되지 않는다.
- **지표 함수의 정확성** — 복제된 정의들이 서로 같은지는 검사하지만,
  그 정의가 애초에 맞는지는 검사하지 않는다. 일관성은 정확성을
  증명하지 못한다. 알려진 답을 넣는 단위 시험이 필요하다.
- **깨끗한 환경에서의 전체 재현** — 받은 사람이 돌리는 것과 같은
  조건으로 처음부터 끝까지 돌려 본 적이 아직 없다.

---

## 디렉터리

```
src/          분석·검증 스크립트 85개 (_shared.py 가 공용 모듈)
data/processed/  분석에 쓰는 처리 완료 데이터
results/      모든 중간·최종 결과 (CSV + 요약 TXT)
figures/      투고용 그림 (PDF 벡터 + PNG 600dpi)
manuscript/   LaTeX 표, 참고문헌, 원고 파일
_backup/      수정 이력 보관 (재현에 불필요)
```

### `results/` 안의 파일

현재 190개(CSV + TXT + JSON)다. 보고서·표·그림 생성기가 직접 읽는 것은
84개이고, 나머지 106개의 내역은 이렇다 (2026-10-20 실측).

| 분류 | 개수 | 내용 |
|---|---|---|
| 분석별 요약 TXT | 56 | 스크립트마다 사람이 읽을 수 있게 남기는 실행 요약 |
| 로그·진행 기록 | 23 | `rerun_all_log.txt`, `*_progress.txt`(긴 분석의 재개 기록) |
| 순열 귀무분포 | 13 | `permutation_null_*`, `era_permutation_null_*`, `*_partial_*` — 검정을 다시 돌리지 않고 확인용 |
| 자료 감사 기록 | 4 | `smiles_audit`, `year_audit`, `label_audit`, `atc_review_L` |
| 그 외 | 10 | `environment.txt`, `prevalence_sweep_reps.csv`(반복별 원자료), `*_pred.csv`(분자별 예측 — 짝지은 비교용) 등 |

이 개수는 손으로 적은 것이므로 분석을 추가하면 어긋난다. 실측은
`ls results/ | wc -l` 과 `python src/verify_state.py` 로 한다.

요약 TXT 는 같은 수치를 CSV 와 중복으로 담고 있으므로 재현에 필수는 아니다.
검토자가 스크립트를 돌리지 않고도 각 분석의 판정을 읽을 수 있도록 남겨 둔 것이다.

`atc_review_L.csv` 는 ATC L 계열(항종양·면역조절) 284건의 수동 검토
기록이다.

### `rerun_all.sh` 가 부르지 않는 스크립트 10개

| 스크립트 | 이유 |
|---|---|
| `_shared.py` | 공용 모듈 (적재·특성화·모델·ECE 등) |
| `fetch_smiles.py`, `fetch_smiles_fix.py` | PubChem 네트워크가 필요한 SMILES 수집 |
| `map_approval_year.py`, `dictrank_pipeline.py` | `data/raw/` 가 필요한 전처리 |
| `record_env.py` | 환경 기록. 돌리면 논문 환경(`results/environment.json`)을 덮어쓰므로 제외 |
| `diagnose.py`, `diagnose_atc.py` | PubChem API 진단용. 결과 파일을 만들지 않는다 |
| (`check_determinism.py` 는 2026-10-20 부터 `rerun_all.sh` 안에 있다 — 격자 모델과 재보정 사다리의 실행 간 동일성을 검사하고 `results/determinism.csv` 를 남긴다) |
| `atc_analysis.py` | `atc_kegg.py` 로 대체된 초기판 (ATC 매핑을 KEGG 기준으로 바꿨다) |
| `baseline.py` | `temporal_split.py` + `baseline_cv.py` 로 대체된 초기판. 출력(`baseline_results.csv`)은 `results/` 에 없다 |

재현에 필요한 스크립트는 전부 `rerun_all.sh` 안에 있다. 위 10개를 돌리지
않아도 보고서·표·그림의 모든 수치가 나온다.

---

## 환경

Python 3.11.14 / macOS 14.3.1 (arm64) / conda 환경 `pharmml`.
전체 패키지 목록은 `results/environment.json`, `results/environment.txt`
에 기록되어 있다. `requirements.txt` 의 주석에 conda 설치가 필요한
패키지를 적어 두었다.

---

## 인용

(논문 게재 후 서지정보와 DOI 를 여기에 추가한다.)

## 라이선스

- **코드**(`src/`, `rerun_all.sh`) — MIT. 전문은 `LICENSE` 에 있다.
- **데이터** — 각 원출처의 조건을 따른다. DILIrank·DICTrank 는 미국
  FDA/NCTR 공개 자료, ATC 분류는 WHO Collaborating Centre for Drug
  Statistics Methodology 의 자료다. 재배포 조건은 각 출처를 확인할 것.
- **결과·문서**(`results/`, `manuscript/`, 보고서) — 논문 게재 후 학술지의
  조건을 따른다.
