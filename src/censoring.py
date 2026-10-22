"""
라벨 검열(label censoring) 보정 분석

가설: 시간분할 성능 저하의 상당 부분은 화학공간 이동이 아니라
      신규 승인약의 라벨이 아직 성숙하지 않아 생긴 편의(bias)다.

근거:
  - DILIrank 1.0 -> 2.0 개정 49건이 전부 음성/모호 -> 양성 방향
  - 승인 시기별 양성률 단조 감소 (65% -> 52% -> 29% -> 14%)

검증 방법:
  실험 1  노출기간 절단: 승인 후 N년 이상 지난 약물만으로 재평가
  실험 2  음성 가중: 노출기간이 짧은 음성에 낮은 가중치 부여
  실험 3  학습군·검증군 동시 절단: 학습군도 같은 기준을 적용해 공정 비교
  실험 4  예측확률 분포 비교: 학습군 양성 / 검증군 양성 / 검증군 음성의
          예측확률 중앙값을 비교

  [주의] 실험 4 가 함께 출력하는 '검증군 양성이 음성보다 높게 예측되는
  비율' 은 정규화된 Mann-Whitney U 통계량이며 AUROC 와 같은 양이다
  (동점을 0.5 대신 0 으로 세는 것만 다르다). 독립적인 검증이 아니므로
  논문에 새로운 증거로 제시하지 말 것. 또한 이 계산에는 검증군 음성이
  전부 들어가므로 'PU 관점' / '양성만 사용' 이라는 서술은 옳지 않다.
  정보가 있는 부분은 중앙값 3개다.

사용법:
    python src/censoring.py

출력:
    results/censoring_results.csv
    results/censoring_summary.txt
"""

import warnings
from pathlib import Path

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
# 라벨 관측 종료 시점은 데이터셋마다 다르다.
#   DILIrank 2.0 : 2025년 공개
#   DICTrank     : 2023년 공개 (Drug Discovery Today 28(11):103770)
# 이전 판은 두 데이터셋 모두 2025를 썼고, 그 결과 DICTrank의 관찰기간이
# 일괄 2년씩 과대 계산되어 절단 기준(>=8/10/12년)이 실제로는
# >=6/8/10년으로 작동했다.
REF_YEARS = {"DILIrank": 2025, "DICTrank": 2023}
N_RAND = 20
rng = np.random.default_rng(SEED)

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def featurize(smiles):
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


def make_model():
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


def weighted_auc(y, p, w):
    """가중 AUROC (양성-음성 쌍별 가중 평균)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    num = den = 0.0
    for i in pos:
        gt = (p[i] > p[neg]).astype(float) + 0.5 * (p[i] == p[neg])
        ww = w[i] * w[neg]
        num += (gt * ww).sum()
        den += ww.sum()
    return num / den if den > 0 else np.nan


def random_baseline(X, y, n_test):
    """동일 크기 무작위 분할 반복 -> 평균 AUROC."""
    vals = []
    for i in range(N_RAND):
        r = np.random.default_rng(SEED + i)
        idx = r.permutation(len(y))
        ti, ri = idx[:n_test], idx[n_test:]
        if len(np.unique(y[ti])) < 2:
            continue
        m = make_model().fit(X[ri], y[ri])
        vals.append(roc_auc_score(y[ti], m.predict_proba(X[ti])[:, 1]))
    return float(np.mean(vals)) if vals else np.nan


def run(name, path, ref_year):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.reset_index(drop=True)
    df["exposure"] = ref_year - df["approval_year"]  # 관찰 기간(년)
    n_future = int((df["exposure"] < 0).sum())
    if n_future:
        log(
            f"  [주의] 라벨 공개연도({ref_year}) 이후 승인으로 기록된 화합물 "
            f"{n_future}건 - 승인연도 매칭 오류 가능성"
        )

    log("=" * 66)
    log(f"{name}")
    log("=" * 66)

    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr

    # ---------------------------------------------- 검열 근거 확인
    log("\n[검열 근거] 노출기간 구간별 양성률")
    for lo, hi in [(0, 7), (8, 11), (12, 15), (16, 25), (26, 100)]:
        m = (df.exposure >= lo) & (df.exposure <= hi)
        if m.sum() >= 10:
            log(
                f"  {lo:2d}-{hi:3d}년 (n={m.sum():3d}, "
                f"승인 {ref_year-hi if hi<100 else 1939}-{ref_year-lo}): "
                f"양성 {df.loc[m,'y'].mean():5.1%}"
            )

    # ---------------------------------------------- 기준선
    m0 = make_model().fit(X[tr], y[tr])
    prob = m0.predict_proba(X[te])[:, 1]
    auc_base = roc_auc_score(y[te], prob)
    auc_rand = random_baseline(X, y, te.sum())
    gap_base = auc_rand - auc_base

    log(f"\n[기준] 무작위 {auc_rand:.4f} | 시간분할 {auc_base:.4f} | " f"격차 {gap_base:+.4f}")

    rows = [
        {
            "exp": "기준(전체)",
            "n_test": int(te.sum()),
            "pos_rate": float(y[te].mean()),
            "auroc": auc_base,
            "auroc_random": auc_rand,
            "gap": gap_base,
        }
    ]

    te_exp = df.loc[te, "exposure"].values
    yte = y[te]

    # ---------------------------------------------- 실험 1: 노출기간 절단
    log("\n--- 실험 1: 검증군 노출기간 절단 ---")
    log("  (라벨이 성숙한 약물만 남기고 재평가)")
    for min_exp in [8, 10, 12, 15]:
        m = te_exp >= min_exp
        if m.sum() < 30 or len(np.unique(yte[m])) < 2:
            log(f"  >={min_exp}년: n={m.sum()} - 표본 부족")
            continue
        a = roc_auc_score(yte[m], prob[m])
        rows.append(
            {
                "exp": f"절단>={min_exp}년",
                "n_test": int(m.sum()),
                "pos_rate": float(yte[m].mean()),
                "auroc": a,
                "auroc_random": auc_rand,
                "gap": auc_rand - a,
            }
        )
        log(
            f"  >={min_exp}년 (승인 <={ref_year-min_exp}): n={m.sum():3d} "
            f"양성{yte[m].mean():5.1%}  AUROC {a:.4f}  "
            f"격차 {auc_rand-a:+.4f}  (기준 대비 {a-auc_base:+.4f})"
        )

    # ---------------------------------------------- 실험 2: 음성 가중
    log("\n--- 실험 2: 노출기간 기반 음성 가중 ---")
    log("  (음성 라벨의 신뢰도 = 관찰기간에 비례, 양성은 항상 1.0)")
    for tau in [5, 8, 10, 15]:
        w = np.where(yte == 1, 1.0, np.clip(te_exp / tau, 0, 1))
        a = weighted_auc(yte, prob, w)
        rows.append(
            {
                "exp": f"가중tau={tau}",
                "n_test": int(te.sum()),
                "pos_rate": float(yte.mean()),
                "auroc": a,
                "auroc_random": auc_rand,
                "gap": auc_rand - a,
            }
        )
        log(
            f"  tau={tau:2d}년: 가중AUROC {a:.4f}  격차 {auc_rand-a:+.4f}  "
            f"(기준 대비 {a-auc_base:+.4f})"
        )

    # ---------------------------------------------- 실험 3: 학습군도 절단
    log("\n--- 실험 3: 학습군·검증군 동시 절단 (공정 비교) ---")
    for min_exp in [10, 15]:
        trm = tr & (df.exposure.values >= min_exp)
        tem = te & (df.exposure.values >= min_exp)
        if tem.sum() < 30 or len(np.unique(y[tem])) < 2:
            log(f"  >={min_exp}년: 검증군 n={tem.sum()} - 표본 부족")
            continue
        mm = make_model().fit(X[trm], y[trm])
        pp = mm.predict_proba(X[tem])[:, 1]
        a = roc_auc_score(y[tem], pp)
        ar = random_baseline(X[trm | tem], y[trm | tem], int(tem.sum()))
        rows.append(
            {
                "exp": f"양측절단>={min_exp}년",
                "n_test": int(tem.sum()),
                "pos_rate": float(y[tem].mean()),
                "auroc": a,
                "auroc_random": ar,
                "gap": ar - a,
            }
        )
        log(
            f"  >={min_exp}년: 학습 n={trm.sum():3d}, 검증 n={tem.sum():3d}  "
            f"무작위 {ar:.4f} | 시간분할 {a:.4f} | 격차 {ar-a:+.4f}"
        )

    # ---------------------------------------------- 실험 4: 양성 순위만
    log("\n--- 실험 4: 양성 예측확률 분포 비교 ---")
    log("  (음성 라벨을 신뢰하지 않고 양성만 사용)")
    tr_pos_prob = m0.predict_proba(X[tr])[:, 1][y[tr] == 1]
    te_pos_prob = prob[yte == 1]
    te_neg_prob = prob[yte == 0]
    log(f"  학습군 양성 예측확률 중앙값: {np.median(tr_pos_prob):.3f}")
    log(f"  검증군 양성 예측확률 중앙값: {np.median(te_pos_prob):.3f} " f"(n={len(te_pos_prob)})")
    log(f"  검증군 음성 예측확률 중앙값: {np.median(te_neg_prob):.3f} " f"(n={len(te_neg_prob)})")
    log(
        f"  -> 검증군 양성이 음성보다 높게 예측되는 비율: "
        f"{np.mean([np.mean(p > te_neg_prob) for p in te_pos_prob]):.3f}"
    )

    out = pd.DataFrame(rows)
    out.insert(0, "dataset", name)

    # ---------------------------------------------- 판정
    log("\n[판정]")
    best = out[out.exp.str.startswith(("절단", "가중"))]
    if len(best):
        b = best.loc[best.gap.idxmin()]
        red = gap_base - b["gap"]
        log(f"  최소 격차 조건: {b['exp']} -> 격차 {b['gap']:+.4f}")
        log(
            f"  기준 격차 {gap_base:+.4f} 대비 {red:+.4f} 감소 " f"({red/gap_base*100:.0f}% 축소)"
            if gap_base != 0
            else ""
        )
        if red > 0.03:
            log("  => 라벨 검열이 성능 저하를 상당 부분 설명함")
        elif red > 0.01:
            log("  => 라벨 검열이 부분적으로 기여함")
        else:
            log("  => 라벨 검열로 설명되지 않음")
    log()
    return out


def main():
    res = []
    for key, nm, p in [
        ("DILIrank", "DILIrank (간독성)", PROC / "dilirank2_with_year.csv"),
        ("DICTrank", "DICTrank (심장독성)", PROC / "dictrank_processed.csv"),
    ]:
        if p.exists():
            ry = REF_YEARS[key]
            log(f"\n[라벨 관측 종료 시점] {key}: {ry}년")
            res.append(run(nm, p, ry))

    if res:
        allr = pd.concat(res, ignore_index=True)
        allr.to_csv(RES / "censoring_results.csv", index=False)
        (RES / "censoring_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
        log(f"저장: {RES/'censoring_results.csv'}")
        log(f"      {RES/'censoring_summary.txt'}")


if __name__ == "__main__":
    main()
