"""
시대 분류기 순열검정 — 순열 횟수 증대판 (2026-09-28)

왜 필요한가
-----------
§13(가) 의 시대 분류기(structure -> '승인 2010년 이후인가') 는
shift_quantification.py 에서 순열 200회로 검정됐고, 두 데이터셋 모두
관측 이상 0회가 나와 p = 1/201 = 0.004975 로 **하한에 걸렸다.**
귀무 SD 기준 z 는 5.4(DILIrank) / 11.0(DICTrank) 이므로 실제 증거는
5σ 를 넘는데 인쇄값은 0.0050 이다. §15 에서 문서의 검정 22건을 단일
가족으로 묶는 스트레스 테스트를 할 때, 이 두 값이 하한에 걸려 있으면
BH 순위 앞자리를 차지하면서도 실제보다 약하게 들어간다.

그래서 순열을 2,000회로 올린다. 하한이 1/2001 = 0.0005 로 내려가고,
이는 22건 단일 가족에서 가장 엄한 문턱(α/m = 0.05/22 = 0.00227)보다
4.5배 작다. 즉 이 항목은 어떤 가족 구성에서도 문턱에 걸리지 않는다.

**이 실험은 결론을 바꾸지 않는다.** 이미 유의하고, 올리면 강해지는
방향뿐이다. 바꾸는 것은 '인쇄된 p 가 증거의 강도를 대표하는가' 다.

구현 주의
---------
1. 관측 AUC 를 n_jobs=1 과 n_jobs=-1 두 가지로 계산해 둘 다 찍는다.
   기존 값(shift_quantification.csv)은 n_jobs=-1 로 나온 것이다.
   귀무분포는 워커에서 n_jobs=1 로 돌므로, **관측값도 n_jobs=1 쪽을
   써서 내부 일관성을 맞춘다.** 두 값이 다르면 그 사실을 인쇄한다.
   (XGBoost hist 는 스레드 수와 무관하게 같은 결과를 낸다고 알려져
    있으나, 그 가정에 기대지 않고 실측해 남긴다.)
2. 순열 난수와 fold seed 는 matched_n_control.run_era 와 **동일**하다
   (rng = default_rng(SEED+i), fold seed = SEED 고정). 따라서 앞 100개
   귀무값의 평균·SD 가 matched_n_era_permutation_quick100.csv 와
   일치해야 한다. 이것을 자동 대조 항목으로 넣었다.
3. 중단 재개: results/era_partial_<ds>.csv + .meta.json.
   지문이 다르면(데이터가 바뀌었으면) 캐시를 버린다.

실행
----
    python src/era_permutation.py                 # 2,000회
    N_PERM=500 python src/era_permutation.py      # 짧게 점검
    N_WORKERS=8 python src/era_permutation.py     # 워커 수 지정
"""

import os

# XGBoost(OpenMP) 를 여러 프로세스로 돌리기 전에 스레드 수를 1로 고정한다.
# permutation_test.py 와 같은 이유: fork 교착 회피 + 스레드 수 의존성 제거.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF, N_FOLDS = 42, 2010, 5
N_PERM = int(os.environ.get("N_PERM", "2000"))
N_WORKERS = int(os.environ.get("N_WORKERS", str(min(6, os.cpu_count() or 1))))
CHUNK = int(os.environ.get("CHUNK", "25"))

DATASETS = [("DILIrank", "dilirank2_with_year.csv"), ("DICTrank", "dictrank_processed.csv")]

log_lines = []


def log(m=""):
    print(m, flush=True)
    log_lines.append(str(m))


# ------------------------------------------------------------ 공통 (_shared 와 동일)
def load(path):
    df = pd.read_csv(path).dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def featurize(smiles):
    """ECFP4(2048) + RDKit 기술자 6종 — _shared.featurize 와 동일."""
    fps, desc = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
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


def make_model(n_jobs):
    """_shared.make_model 과 초매개변수 동일. n_jobs 만 인자로 뺐다."""
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


def cv_auc(X, labels, n_jobs=1, fold_seed=SEED):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    oof = np.zeros(len(labels), dtype=float)
    for tri, tei in skf.split(X, labels):
        mm = make_model(n_jobs).fit(X[tri], labels[tri])
        oof[tei] = mm.predict_proba(X[tei])[:, 1]
    return float(roc_auc_score(labels, oof))


# ------------------------------------------------------------ 워커
_G = {}


def _init(X, era):
    _G["X"], _G["era"] = X, era


def _one_perm(i):
    """i번째 순열의 5-fold CV AUC. run_era 와 난수·fold seed 동일."""
    X, era = _G["X"], _G["era"]
    r = np.random.default_rng(SEED + i)
    return i, cv_auc(X, r.permutation(era), n_jobs=1, fold_seed=SEED)


# ------------------------------------------------------------ 재개 캐시
def _fingerprint(df):
    h = hashlib.sha1("\n".join(sorted(df["smiles"].astype(str))).encode("utf-8")).hexdigest()
    return {
        "n": int(len(df)),
        "n_post": int((df["approval_year"].values >= CUTOFF).sum()),
        "smiles_sha1": h,
        "cutoff": int(CUTOFF),
        "seed": int(SEED),
        "n_folds": int(N_FOLDS),
    }


def load_partial(path, fp):
    meta = path.with_suffix(".meta.json")
    if not path.exists():
        return {}
    if not meta.exists():
        log(
            f"  [캐시 폐기] {path.name} 에 지문이 없다 — 어느 데이터에서"
            f" 나온 것인지 확인할 수 없으므로 버린다."
        )
        return {}
    old = json.loads(meta.read_text(encoding="utf-8"))
    if old != fp:
        diff = [k for k in fp if old.get(k) != fp[k]]
        log(f"  [캐시 폐기] 데이터가 바뀌었다 (불일치: {', '.join(diff)}).")
        for k in diff:
            log(f"      {k}: 캐시 {old.get(k)} -> 현재 {fp[k]}")
        return {}
    return {int(r.i): float(r.null_auc) for r in pd.read_csv(path).itertuples()}


def save_partial(path, done, fp):
    pd.DataFrame({"i": sorted(done), "null_auc": [done[i] for i in sorted(done)]}).to_csv(
        path, index=False
    )
    path.with_suffix(".meta.json").write_text(
        json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8"
    )


# ------------------------------------------------------------ 데이터셋 1개
def run(name, filename):
    path = PROC / filename
    if not path.exists():
        log(f"[건너뜀] {name}: {path} 없음")
        return None

    df = load(path)
    X = featurize(df["smiles"].tolist())
    era = (df["approval_year"].values >= CUTOFF).astype(int)

    log("=" * 88)
    log(
        f"{name}   n={len(df)}   시대 양성(2010~) {int(era.sum())} / "
        f"음성 {int((1 - era).sum())}"
    )
    log("=" * 88)

    # --- 관측값: 스레드 수 두 조건에서 각각 계산해 남긴다
    t0 = time.time()
    obs1 = cv_auc(X, era, n_jobs=1)
    obs_all = cv_auc(X, era, n_jobs=-1)
    log(f"  [관측] 5-fold CV AUC   n_jobs=1 {obs1:.12f}")
    log(f"                          n_jobs=-1 {obs_all:.12f}")
    log(
        f"         두 값의 차 {abs(obs1 - obs_all):.2e}"
        f"  ({'동일' if abs(obs1-obs_all) < 1e-12 else '다름 — 아래 대조 참고'})"
    )
    log(f"         (관측값 계산 {time.time()-t0:.0f}초)")

    # --- 기존 기록과 대조
    ref_path = RES / "shift_quantification.csv"
    ref = None
    if ref_path.exists():
        s = pd.read_csv(ref_path)
        s = s[s.dataset == name]
        if len(s):
            ref = float(s.iloc[0]["era_auc_full"])
            d1, da = abs(ref - obs1), abs(ref - obs_all)
            log(f"  [대조] shift_quantification.csv era_auc_full = {ref:.12f}")
            log(f"         n_jobs=1 과의 차 {d1:.2e} / n_jobs=-1 과의 차 {da:.2e}")
            log(f"         -> {'일치' if min(d1, da) < 5e-4 else '불일치 — 확인 필요'}")

    # 귀무분포는 워커에서 n_jobs=1 로 돈다. 내부 일관성을 위해
    # 관측값도 n_jobs=1 쪽을 쓴다.
    obs = obs1
    log(f"  [채택] 검정에 쓰는 관측값 = n_jobs=1 의 {obs:.12f}" f"  (귀무분포와 같은 조건)")

    part = RES / f"era_partial_{name.lower()}.csv"
    fp = _fingerprint(df)
    done = load_partial(part, fp)
    todo = [i for i in range(N_PERM) if i not in done]
    if done:
        log(f"  [재개] 이미 끝난 {len(done)}회를 건너뛴다")
    log(f"  순열 {N_PERM}회 (프로세스 {N_WORKERS}개, 남은 {len(todo)}회)")

    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_init,
            initargs=(X, era),
        ) as ex:
            for k, (i, a) in enumerate(ex.map(_one_perm, todo, chunksize=1), 1):
                done[i] = a
                if k % CHUNK == 0 or k == len(todo):
                    save_partial(part, done, fp)
                    el = time.time() - t0
                    eta = el / k * (len(todo) - k)
                    log(f"    {k}/{len(todo)}   경과 {el/60:.1f}분 / " f"남은 예상 {eta/60:.1f}분")

    null = np.array([done[i] for i in sorted(done) if i < N_PERM])
    n = len(null)
    r_ge = int((null >= obs).sum())
    p = (1 + r_ge) / (1 + n)
    floor = 1 / (n + 1)
    z = (obs - null.mean()) / null.std(ddof=1) if null.std(ddof=1) else np.nan

    log(f"  귀무 평균 {null.mean():.4f} (SD {null.std(ddof=1):.4f})")
    log(
        f"  2.5–97.5 백분위 {np.percentile(null,2.5):.4f}–"
        f"{np.percentile(null,97.5):.4f}   최대 {null.max():.4f}"
    )
    log(f"  관측값 이상 {r_ge}/{n}")
    log(
        f"  p (Phipson–Smyth) = {p:.6f}   하한 1/{n+1} = {floor:.6f}"
        f"   {'← 하한에 걸림' if r_ge == 0 else '← 하한이 아님'}"
    )
    log(f"  귀무 SD 기준 z = {z:.1f}")
    log(
        f"  22건 단일 가족의 가장 엄한 문턱 α/m = 0.05/22 = 0.002273 "
        f"대비: {'통과' if p < 0.05/22 else '미달'}"
    )

    # --- 앞 100개로 quick100 기록과 대조 (난수 재현성 확인)
    q = RES / "matched_n_era_permutation_quick100.csv"
    if q.exists() and n >= 100:
        qq = pd.read_csv(q)
        qq = qq[qq.dataset == name]
        if len(qq):
            first = np.array([done[i] for i in range(100)])
            dm = abs(first.mean() - float(qq.iloc[0]["null_mean"]))
            ds = abs(first.std(ddof=0) - float(qq.iloc[0]["null_sd"]))
            log(
                f"  [대조] 앞 100개 귀무값 vs quick100 기록: "
                f"평균 차 {dm:.2e}, SD 차 {ds:.2e}  "
                f"-> {'일치' if max(dm, ds) < 1e-9 else '불일치'}"
            )

    log(f"  소요 {(time.time()-t0)/60:.1f}분")
    log()

    pd.DataFrame({"null_auc": null}).to_csv(
        RES / f"era_permutation_null_{name.lower()}.csv", index=False
    )

    return dict(
        dataset=name,
        n=len(df),
        n_post=int(era.sum()),
        era_auc=obs,
        era_auc_njobs_all=obs_all,
        era_auc_reference=ref,
        n_perm=n,
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)),
        null_max=float(null.max()),
        null_p025=float(np.percentile(null, 2.5)),
        null_p975=float(np.percentile(null, 97.5)),
        n_ge=r_ge,
        p_phipson_smyth=p,
        p_floor=floor,
        z=z,
    )


def main():
    log("시대 분류기 순열검정 — 횟수 증대판")
    log(f"N_PERM={N_PERM}, 워커 {N_WORKERS}개, seed {SEED}, 절단 {CUTOFF}, " f"{N_FOLDS}-fold CV")
    log(f"python {sys.version.split()[0]}   " f"numpy {np.__version__}   pandas {pd.__version__}")
    log()

    rows = [r for r in (run(n, f) for n, f in DATASETS) if r]
    if not rows:
        log("결과 없음")
        return

    out = pd.DataFrame(rows)
    out.to_csv(RES / "era_permutation.csv", index=False)

    log("=" * 88)
    log("요약")
    log("=" * 88)
    log(
        f"{'데이터셋':10s} {'시대AUC':>9s} {'귀무평균':>9s} {'SD':>7s} "
        f"{'이상':>7s} {'p':>10s} {'하한':>9s} {'z':>6s}"
    )
    log("-" * 88)
    for r in rows:
        log(
            f"{r['dataset']:10s} {r['era_auc']:9.4f} {r['null_mean']:9.4f} "
            f"{r['null_sd']:7.4f} {str(r['n_ge'])+'/'+str(r['n_perm']):>7s} "
            f"{r['p_phipson_smyth']:10.6f} {r['p_floor']:9.6f} {r['z']:6.1f}"
        )
    log("-" * 88)
    log("[읽는 법] '이상' 이 0 이면 p 는 하한값이고 실제 p 는 그보다 작다.")
    log("          0 이 아니면 p 는 하한이 아니며 횟수를 더 올려도")
    log("          그 값 근처로 수렴한다.")
    log()
    log(f"저장: {RES/'era_permutation.csv'}")
    log(f"      {RES/'era_permutation_summary.txt'}")
    log(f"      era_permutation_null_<데이터셋>.csv (귀무분포 전체)")

    (RES / "era_permutation_summary.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
