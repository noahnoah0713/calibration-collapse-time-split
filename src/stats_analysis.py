"""
DILIrank 2.0 시간분할 - 통계 검정 및 층화 분석

1) 순열 검정 결과 인용 (계산은 src/permutation_test.py 가 담당)
2) 유사도 상/하위 부분군 AUROC 차이 (2표본 부트스트랩)

   주의: 이전 판에서 이 검정을 'DeLong 검정'이라 적었으나 잘못이다.
   DeLong 검정은 같은 표본에 대한 두 모델의 상관된 AUROC 를 비교하는
   방법이다. 여기서는 검증군을 유사도 중앙값으로 둘로 나눈 '서로
   겹치지 않는 두 부분군'의 AUROC 를 비교하므로 DeLong 의 가정이
   성립하지 않는다. 실제 코드는 처음부터 2표본 부트스트랩을 썼고,
   보고된 p 값도 부트스트랩 p 다. 명칭만 틀려 있었다.
3) 유사도-성능 관계의 통계적 검정 (Spearman + 로지스틱)
4) 승인 연도구간별 성능 추이
5) 물성별 층화

사용법:
    python src/stats_analysis.py

선행 실행:
    python src/permutation_test.py      <- 순열 검정의 단일 출처

    예전에는 이 스크립트가 순열 검정을 200회 직접 돌렸다. 같은 검정이
    dictrank_pipeline.py 에도 따로 있어 두 데이터셋의 p 값이 서로 다른
    시점의 값이 되는 문제가 있었고, 0/200 이 "p = 0.0000" 으로 출력되는
    문제도 있었다. 계산을 permutation_test.py 한 곳으로 옮기고 여기서는
    그 결과를 읽어 인용만 한다.

출력:
    results/stats_summary.txt
    results/similarity_tests.csv     <- 다중비교 보정이 읽는 파일
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from scipy import stats
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 42
CUTOFF = 2010

rng = np.random.default_rng(SEED)
log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def featurize(smiles):
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


def model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


# ------------------------------------------------------------------ main
def main():
    df = pd.read_csv(PROC / "dilirank2_with_year.csv")
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)

    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m = yr < CUTOFF
    te_m = ~tr_m
    n_test = te_m.sum()

    sim_tests = []

    log("=" * 62)
    log("DILIrank 2.0 시간분할 통계 검정")
    log("=" * 62)
    log(f"전체 {len(df)}개 | 학습군 {tr_m.sum()} | 검증군 {n_test}")
    log()

    # 관측된 시간분할 성능
    m0 = model().fit(X[tr_m], y[tr_m])
    prob_t = m0.predict_proba(X[te_m])[:, 1]
    auc_t = roc_auc_score(y[te_m], prob_t)
    log(f"[관측] 시간분할 AUROC = {auc_t:.4f}")

    # ---------------------------------------------- 1) 순열 검정 (결과 인용)
    log()
    log("--- 1) 순열 검정 (src/permutation_test.py 결과 인용) ---")
    ppath = RES / "permutation_test.csv"
    if not ppath.exists():
        raise SystemExit(
            f"[중단] {ppath} 가 없습니다.\n"
            f"        먼저 python src/permutation_test.py 를 실행하세요.\n"
            f"        (순열 검정은 두 데이터셋을 한 번에 돌리는 그 스크립트가"
            f" 단일 출처입니다.)"
        )
    pt = pd.read_csv(ppath)
    row = pt[pt.dataset == "DILIrank"]
    if not len(row):
        raise SystemExit(f"[중단] {ppath} 에 DILIrank 행이 없습니다.")
    row = row.iloc[0]
    null_mean, null_sd = float(row.null_mean), float(row.null_sd)
    p_perm = float(row.p_permutation)
    if abs(float(row.auroc_temporal) - auc_t) > 5e-4:
        log(
            f"    [주의] permutation_test.csv 의 시간분할 AUROC "
            f"{row.auroc_temporal:.4f} 와 여기서 계산한 {auc_t:.4f} 가 다릅니다. "
            f"두 스크립트가 같은 데이터를 보고 있는지 확인하세요."
        )
    log(f"    무작위 분할 {int(row.n_perm_valid)}회 (유효)")
    log(f"    귀무분포 평균 {null_mean:.4f} (SD {null_sd:.4f})")
    log(f"    2.5-97.5 백분위: {row.null_p2_5:.4f} - {row.null_p97_5:.4f}")
    log(f"    ΔAUROC = {null_mean - auc_t:+.4f}")
    log(
        f"    p (단측, Phipson-Smyth) = {p_perm:.4f}"
        f"   {'*유의*' if p_perm < 0.05 else '(유의하지 않음)'}"
    )

    # ------------------------------------- 2) 유사도 상/하위 (2표본 부트스트랩)
    log()
    log("--- 2) 유사도 상위 vs 하위 부분군 AUROC 차이 " "(2표본 부트스트랩, 2000회) ---")
    tr_fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
        for s in df.loc[tr_m, "smiles"]
    ]
    sim = np.array(
        [
            max(
                DataStructs.BulkTanimotoSimilarity(
                    AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048), tr_fps
                )
            )
            for s in df.loc[te_m, "smiles"]
        ]
    )

    yte = y[te_m]
    med = np.median(sim)
    hi, lo = sim >= med, sim < med
    if len(np.unique(yte[hi])) == 2 and len(np.unique(yte[lo])) == 2:
        a_hi = roc_auc_score(yte[hi], prob_t[hi])
        a_lo = roc_auc_score(yte[lo], prob_t[lo])
        log(f"    유사도 >= {med:.3f} (n={hi.sum()}): AUROC {a_hi:.4f}")
        log(f"    유사도 <  {med:.3f} (n={lo.sum()}): AUROC {a_lo:.4f}")
        log(f"    차이 {a_hi - a_lo:+.4f}")
        # 독립 표본이므로 부트스트랩 검정
        diffs = []
        for _ in range(2000):
            ih = rng.choice(np.where(hi)[0], hi.sum(), replace=True)
            il = rng.choice(np.where(lo)[0], lo.sum(), replace=True)
            if len(np.unique(yte[ih])) == 2 and len(np.unique(yte[il])) == 2:
                diffs.append(
                    roc_auc_score(yte[ih], prob_t[ih]) - roc_auc_score(yte[il], prob_t[il])
                )
        diffs = np.array(diffs)
        p_d = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        log(f"    부트스트랩 p = {p_d:.4f}" f"   {'*유의*' if p_d < 0.05 else '(유의하지 않음)'}")
        sim_tests.append(
            {
                "dataset": "DILIrank",
                "family": "3_exploratory_subgroup",
                "test": "화학공간 유사도 상/하위 AUROC 차이",
                "n_a": int(hi.sum()),
                "n_b": int(lo.sum()),
                "stat": float(a_hi - a_lo),
                "p": float(p_d),
            }
        )

    # ---------------------------------------------- 3) 유사도-정확도 상관
    log()
    log("--- 3) 유사도와 예측 정확성의 관계 ---")
    correct_conf = np.where(yte == 1, prob_t, 1 - prob_t)  # 정답 클래스 확률
    rho, p_rho = stats.spearmanr(sim, correct_conf)
    log(f"    Spearman rho = {rho:+.4f}, p = {p_rho:.4g}" f"   {'*유의*' if p_rho < 0.05 else ''}")
    sim_tests.append(
        {
            "dataset": "DILIrank",
            "family": "3_exploratory_subgroup",
            "test": "유사도-정답확률 Spearman",
            "n_a": int(len(sim)),
            "n_b": 0,
            "stat": float(rho),
            "p": float(p_rho),
        }
    )
    log("    (양수 = 유사할수록 정답 확률이 높음)")

    # ---------------------------------------------- 4) 연도구간별
    log()
    log("--- 4) 승인 연도구간별 성능 ---")
    te_years = yr[te_m]
    for lo_y, hi_y in [(2010, 2013), (2014, 2017), (2018, 2022)]:
        m_ = (te_years >= lo_y) & (te_years <= hi_y)
        if m_.sum() >= 15 and len(np.unique(yte[m_])) == 2:
            log(
                f"    {lo_y}-{hi_y}: n={m_.sum():3d}, 양성 {yte[m_].mean():.0%}, "
                f"AUROC {roc_auc_score(yte[m_], prob_t[m_]):.4f}, "
                f"유사도중앙값 {np.median(sim[m_]):.3f}"
            )
        else:
            log(f"    {lo_y}-{hi_y}: n={m_.sum():3d} (표본 부족)")

    # ---------------------------------------------- 5) 물성별 층화
    log()
    log("--- 5) 물성별 층화 (검증군) ---")
    te_smi = df.loc[te_m, "smiles"].tolist()
    mw = np.array([Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in te_smi])
    logp = np.array([Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in te_smi])
    for nm, v in [("분자량", mw), ("logP", logp)]:
        thr = np.median(v)
        for tag, m_ in [(f"{nm} < {thr:.0f}", v < thr), (f"{nm} >= {thr:.0f}", v >= thr)]:
            if m_.sum() >= 20 and len(np.unique(yte[m_])) == 2:
                log(
                    f"    {tag:18s} n={m_.sum():3d}  "
                    f"AUROC {roc_auc_score(yte[m_], prob_t[m_]):.4f}"
                )

    # ---------------------------------------------- 요약
    log()
    log("=" * 62)
    log("결론")
    log("=" * 62)
    log(f"  무작위 분할 평균 AUROC : {null_mean:.4f}")
    log(f"  시간분할 AUROC         : {auc_t:.4f}")
    log(f"  격차                   : {null_mean - auc_t:+.4f}")
    log(f"  순열검정 p             : {p_perm:.4f}  (permutation_test.py)")
    if p_perm < 0.05:
        log("  -> 시간분할 성능 저하는 통계적으로 유의함")
    else:
        log("  -> 유의성 미확보. 표본 크기 한계 가능성")

    if sim_tests:
        pd.DataFrame(sim_tests).to_csv(RES / "similarity_tests.csv", index=False)

    (RES / "stats_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n저장: {RES / 'stats_summary.txt'}")
    if sim_tests:
        print(f"      {RES / 'similarity_tests.csv'}")


if __name__ == "__main__":
    main()
