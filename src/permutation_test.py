"""
순열 검정 (단일 출처) — 시간분할 성능 저하의 유의성

왜 별도 스크립트인가
--------------------
기존에는 stats_analysis.py(DILIrank)와 dictrank_pipeline.py(DICTrank)가
각자 순열 검정을 200회씩 돌렸다. 세 가지 문제가 있었다.

  1) p = 0/200 이 "0.0000"으로 출력되었다. 순열 p-value는 0이 될 수
     없다. 유한 번 추출한 순열에서 초과 사례가 하나도 없다는 것은
     "p가 0"이 아니라 "p가 1/(N+1)보다 작다"는 뜻이다.
     Phipson & Smyth (2010)의 편향 보정 추정량을 쓴다:

         p = (1 + #{null <= observed}) / (1 + N_valid)

     이 값은 정확히 불편(unbiased)하며 절대 0이 되지 않는다.

  2) dictrank_pipeline.py는 처리된 CSV를 다시 만들기 때문에 구조
     필터(apply_structure_filter.py) 이후의 재실행 목록에서 제외되어
     있다. 그래서 DICTrank 쪽 p값만 필터 이전 값으로 남아 있었다.
     두 데이터셋을 한 스크립트에서 돌려 이 불일치를 없앤다.

  3) 200회는 XGBoost 재학습 비용 때문에 택한 숫자였다. 여기서는
     프로세스 병렬 + 중단 재개를 넣어 1,000회를 현실적인 시간에
     돌린다.

설계
----
  - 귀무분포: 시간분할과 "동일한 검증군 크기"의 무작위 분할을
    N_PERM회 반복해 얻은 AUROC 분포.
  - 난수 seed는 기존과 동일하게 SEED+i 를 쓴다. 따라서 앞의 200회는
    기존 실행과 같은 분할을 재현한다.
  - 각 XGBoost는 n_jobs=1로 두고 프로세스 4개로 병렬화한다.
    (작은 데이터에서는 스레드 병렬보다 프로세스 병렬이 빠르다.)
  - 중간 결과를 results/permutation_partial_<데이터셋>.csv 에
    계속 기록한다. 중단 후 같은 명령을 다시 실행하면 남은 회차부터
    이어서 돈다.

사용법
------
    python src/permutation_test.py            # N_PERM=1000 (기본)
    N_PERM=200 python src/permutation_test.py # 빠른 확인용

출력
----
    results/permutation_test.csv           <- 다른 스크립트가 읽는 단일 출처
    results/permutation_test_summary.txt
    results/permutation_null_dilirank.csv
    results/permutation_null_dictrank.csv

참고문헌
--------
    Phipson B, Smyth GK. Permutation P-values should never be zero:
    calculating exact P-values when permutations are randomly drawn.
    Stat Appl Genet Mol Biol. 2010;9(1):Article39.
"""

import os

# XGBoost(OpenMP)를 여러 프로세스로 돌리기 전에 스레드 수를 1로 고정한다.
#   (1) OpenMP 런타임이 이미 올라온 프로세스를 fork 하면 자식이 교착된다.
#       아래에서 spawn 컨텍스트를 쓰지만 스레드도 함께 묶어 둔다.
#   (2) XGBoost 의 hist 트리법은 deterministic_histogram 기본값 때문에
#       스레드 수와 무관하게 같은 결과를 내지만, 고정해 두면 그 가정에
#       기대지 않아도 된다. 실행 직후 관측 AUROC 를
#       baseline_cv_per_fold.csv 와 대조해 실제로 같은지 확인한다.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import json
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
N_PERM = int(os.environ.get("N_PERM", "1000"))
N_WORKERS = int(os.environ.get("N_WORKERS", str(min(4, os.cpu_count() or 1))))
CHUNK = int(os.environ.get("CHUNK", "20"))  # 이만큼 끝날 때마다 중간 저장

DATASETS = [
    ("DILIrank", "dilirank2_with_year.csv", "간독성"),
    ("DICTrank", "dictrank_processed.csv", "심장독성"),
]

log_lines = []


def log(m=""):
    print(m, flush=True)
    log_lines.append(str(m))


# ------------------------------------------------------------ 공통 함수
def featurize(smiles):
    """ECFP4(2048비트) + RDKit 기술자 6종. 다른 스크립트와 동일."""
    fps, desc = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=np.int8))
        desc.append(
            [
                Descriptors.MolWt(m),
                Descriptors.MolLogP(m),
                Descriptors.TPSA(m),
                Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m),
            ]
        )
    return np.hstack([np.array(fps), np.array(desc)])


def make_model(n_jobs=1):
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=n_jobs,
    )


# ------------------------------------------------- 워커 (프로세스마다 1회 초기화)
_G = {}


def _init(X, y, n_test):
    """프로세스마다 한 번만 실행. 특징행렬을 워커 전역에 얹어 둔다."""
    _G["X"], _G["y"], _G["n_test"] = X, y, n_test


def _one_perm(i):
    """i번째 무작위 분할의 검증군 AUROC. 한쪽 클래스만 나오면 None."""
    X, y, n_test = _G["X"], _G["y"], _G["n_test"]
    r = np.random.default_rng(SEED + i)
    idx = r.permutation(len(y))
    te_i, tr_i = idx[:n_test], idx[n_test:]
    if len(np.unique(y[te_i])) < 2:
        return i, None
    mm = make_model(n_jobs=1).fit(X[tr_i], y[tr_i])
    return i, float(roc_auc_score(y[te_i], mm.predict_proba(X[te_i])[:, 1]))


# ------------------------------------------------------------ 중단 재개
def _fingerprint(df, n_test):
    """재개 캐시가 지금 데이터에서 나온 것인지 판별하는 지문.

    2026-09-21 사고: 라벨충돌 필터를 적용해 데이터가 바뀌었는데
    (DICTrank n=932 -> 919) 필터 이전에 저장된 partial 캐시를 그대로
    이어받아, 관측값은 필터 후 / 귀무분포는 필터 전이 되는 일이
    있었다. 지문이 다르면 캐시를 버리고 처음부터 다시 돈다.
    """
    h = hashlib.sha1("\n".join(sorted(df["smiles"].astype(str))).encode("utf-8")).hexdigest()
    return {
        "n": int(len(df)),
        "n_pos": int(df["y"].sum()),
        "n_test": int(n_test),
        "smiles_sha1": h,
        "cutoff": int(CUTOFF),
        "seed": int(SEED),
    }


def load_partial(path, fp):
    meta = path.with_suffix(".meta.json")
    if not path.exists():
        return {}
    if not meta.exists():
        log(
            f"  [캐시 폐기] {path.name} 에 지문 파일이 없다. "
            f"어느 데이터에서 나온 것인지 확인할 수 없으므로 버린다."
        )
        return {}
    old = json.loads(meta.read_text(encoding="utf-8"))
    if old != fp:
        diff = [k for k in fp if old.get(k) != fp[k]]
        log(
            f"  [캐시 폐기] 데이터가 바뀌었다 (불일치: {', '.join(diff)}). "
            f"귀무분포를 처음부터 다시 계산한다."
        )
        for k in diff:
            log(f"      {k}: 캐시 {old.get(k)} -> 현재 {fp[k]}")
        return {}
    return {
        int(r.i): (None if pd.isna(r.null_auroc) else float(r.null_auroc))
        for r in pd.read_csv(path).itertuples()
    }


def save_partial(path, done, fp):
    pd.DataFrame({"i": sorted(done), "null_auroc": [done[i] for i in sorted(done)]}).to_csv(
        path, index=False
    )
    path.with_suffix(".meta.json").write_text(
        json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8"
    )


# ------------------------------------------------------------ 데이터셋 1개
def run(name, filename, endpoint):
    path = PROC / filename
    if not path.exists():
        log(f"[건너뜀] {name}: {path} 없음")
        return None

    df = pd.read_csv(path).dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)

    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m = df["approval_year"].values < CUTOFF
    te_m = ~tr_m
    n_test = int(te_m.sum())

    log("=" * 70)
    log(f"{name} ({endpoint})  n={len(df)}  학습군 {tr_m.sum()} / 검증군 {n_test}")
    log("=" * 70)

    # 관측값 — 다른 스크립트와 같은 조건(n_jobs=-1)으로 한 번만 학습
    m0 = make_model(n_jobs=1).fit(X[tr_m], y[tr_m])
    auc_t = float(roc_auc_score(y[te_m], m0.predict_proba(X[te_m])[:, 1]))
    log(f"  [관측] 시간분할 AUROC = {auc_t:.4f}")

    # baseline_cv_per_fold.csv 와 교차 확인 (있으면)
    cvp = RES / "baseline_cv_per_fold.csv"
    if cvp.exists():
        cv = pd.read_csv(cvp)
        s = cv[(cv.dataset == name) & (cv.split == "temporal")]
        if len(s):
            ref = float(s.iloc[0]["AUROC"])
            d = abs(ref - auc_t)
            mark = "일치" if d < 5e-4 else f"차이 {d:.4f} <-- 확인 필요"
            log(f"  [대조] baseline_cv_per_fold.csv 의 temporal AUROC " f"{ref:.4f}  ({mark})")

    part = RES / f"permutation_partial_{name.lower()}.csv"
    fp = _fingerprint(df, n_test)
    done = load_partial(part, fp)
    todo = [i for i in range(N_PERM) if i not in done]
    if done:
        log(f"  [재개] 이미 끝난 {len(done)}회를 건너뛰고 {len(todo)}회 진행")
    log(f"  무작위 분할 {N_PERM}회 (프로세스 {N_WORKERS}개, 남은 {len(todo)}회)")

    t0 = time.time()
    if todo:
        # fork 가 아니라 spawn 을 쓴다. fork 는 OpenMP 가 올라온 뒤에
        # 자식을 만들면 교착되고, macOS 기본값도 spawn 이다.
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_init,
            initargs=(X, y, n_test),
        ) as ex:
            for k, (i, a) in enumerate(ex.map(_one_perm, todo, chunksize=1), 1):
                done[i] = a
                if k % CHUNK == 0 or k == len(todo):
                    save_partial(part, done, fp)
                    el = time.time() - t0
                    eta = el / k * (len(todo) - k)
                    log(f"    {k}/{len(todo)}   경과 {el/60:.1f}분 / " f"남은 예상 {eta/60:.1f}분")

    null = np.array([v for v in done.values() if v is not None])
    n_valid = len(null)
    n_skip = len(done) - n_valid
    n_le = int((null <= auc_t).sum())
    p_phipson = (1 + n_le) / (1 + n_valid)
    p_raw = n_le / n_valid if n_valid else np.nan

    log(f"  귀무분포 평균 {null.mean():.4f} (SD {null.std(ddof=1):.4f})")
    log(f"  2.5-97.5 백분위: {np.percentile(null,2.5):.4f} - " f"{np.percentile(null,97.5):.4f}")
    log(f"  ΔAUROC = {null.mean() - auc_t:+.4f}")
    log(
        f"  귀무분포 중 관측값 이하: {n_le}/{n_valid}"
        + (f"  (클래스 부족으로 버린 분할 {n_skip}회)" if n_skip else "")
    )
    log(f"  p (Phipson-Smyth) = {p_phipson:.4f}   <-- 논문에 쓸 값")
    log(f"  p (단순 비율)      = {p_raw:.4f}")
    log(f"  {'*유의*' if p_phipson < 0.05 else '(유의하지 않음)'}")
    log(f"  소요 {(time.time()-t0)/60:.1f}분")
    log()

    pd.DataFrame({"null_auroc": null}).to_csv(
        RES / f"permutation_null_{name.lower()}.csv", index=False
    )
    # 기존 파일명 호환 (make_figures 등이 참조할 수 있음)
    if name == "DILIrank":
        pd.DataFrame({"null_auroc": null}).to_csv(RES / "permutation_null.csv", index=False)

    return {
        "dataset": name,
        "endpoint": endpoint,
        "n": len(df),
        "n_train": int(tr_m.sum()),
        "n_test": n_test,
        "auroc_temporal": auc_t,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_p2_5": float(np.percentile(null, 2.5)),
        "null_p97_5": float(np.percentile(null, 97.5)),
        "delta_auroc": float(null.mean() - auc_t),
        "n_perm_requested": N_PERM,
        "n_perm_valid": n_valid,
        "n_le_observed": n_le,
        "p_permutation": p_phipson,  # 논문 값
        "p_raw_proportion": p_raw,
        "p_estimator": "Phipson-Smyth (1+r)/(1+n)",
    }


def main():
    log("순열 검정 — 시간분할 AUROC vs 동일 크기 무작위 분할")
    log(f"N_PERM={N_PERM}, 프로세스 {N_WORKERS}개, seed {SEED}, 절단 {CUTOFF}년")
    log()

    rows = [r for r in (run(*d) for d in DATASETS) if r]
    if not rows:
        log("실행된 데이터셋 없음.")
        sys.exit(1)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "permutation_test.csv", index=False)

    log("=" * 70)
    log("요약")
    log("=" * 70)
    for r in rows:
        log(
            f"  {r['dataset']:9s} 시간분할 {r['auroc_temporal']:.4f} vs "
            f"무작위 {r['null_mean']:.4f}  "
            f"Δ{r['delta_auroc']:+.4f}  p={r['p_permutation']:.4f}"
        )
    log()
    log("p는 Phipson-Smyth 편향보정 추정량 (1+r)/(1+n) 이므로 0이 되지 않는다.")
    log(f"관측값 이하 사례가 0일 때의 하한은 p = 1/{N_PERM+1} " f"= {1/(N_PERM+1):.4f} 이다.")

    (RES / "permutation_test_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'permutation_test.csv'}")
    log(f"      {RES/'permutation_test_summary.txt'}")


if __name__ == "__main__":
    main()
