"""
시간분할에서의 확률 보정(calibration) 붕괴 정량화

AUROC는 순위 능력만 측정하므로 유병률에 불변이지만,
실제 배포에서는 임계값으로 판정하므로 보정이 무너지면 쓸 수 없다.

측정 지표:
  - Brier score          : 확률 예측의 전체 오차
  - ECE                  : 예상 보정 오차 (구간별 |실제 - 예측|의 가중평균)
  - MCE                  : 최대 보정 오차
  - Calibration slope    : 로지스틱 재보정 기울기 (1이 이상적)
  - Calibration intercept: 절편 (0이 이상적)
  - 평균 예측확률 vs 실제 유병률
  - 최적 임계값 이동      : Youden J 최대화 임계값

비교 대상: 무작위 분할(반복) vs 시간분할

사용법:
    python src/calibration.py

출력:
    results/calibration_results.csv
    results/calibration_summary.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
N_RAND = 20
N_BINS = 10

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


def ece_mce(y, p, n_bins=N_BINS):
    """예상/최대 보정 오차."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        conf, acc = p[m].mean(), y[m].mean()
        gap = abs(acc - conf)
        ece += m.sum() / len(p) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                "n": int(m.sum()),
                "pred": conf,
                "actual": acc,
                "gap": acc - conf,
            }
        )
    return ece, mce, pd.DataFrame(rows)


def calib_slope_intercept(y, p):
    """로지스틱 재보정: logit(p)를 예측자로 사용. slope=1, intercept=0이 이상적."""
    # 2026-10-21: XGBoost 의 predict_proba 는 float32 를 돌려준다.
    # 캐스팅하지 않으면 로짓과 그 뒤의 로지스틱 적합이 **단정밀도**로
    # 일어난다. 10-20 에 같은 결함을 일곱 군데 고쳤는데 이 함수는
    # 이름이 calib_slope 가 아니어서 [1c]·[1d] 양쪽 이름 목록에
    # 걸리지 않아 빠졌다 — 복제를 **이름**으로 찾은 것이 문제였고,
    # 그래서 로짓 변환을 **본문**으로 훑어 다시 찾았다.
    # 이 입력에서 float32 와 float64 의 차는 기울기 4.6e-10 이었다.
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def best_threshold(y, p):
    """Youden J 최대 임계값."""
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def assess(y, p, tag):
    auc = roc_auc_score(y, p)
    pred = (p >= 0.5).astype(int)
    ece, mce, bins = ece_mce(y, p)
    slope, intercept = calib_slope_intercept(y, p)
    thr = best_threshold(y, p)
    pred_opt = (p >= thr).astype(int)
    return {
        "setting": tag,
        "n": len(y),
        "prevalence": float(y.mean()),
        "mean_pred": float(p.mean()),
        "AUROC": auc,
        "Brier": brier_score_loss(y, p),
        "ECE": ece,
        "MCE": mce,
        "calib_slope": slope,
        "calib_intercept": intercept,
        "BalAcc@0.5": balanced_accuracy_score(y, pred),
        "MCC@0.5": matthews_corrcoef(y, pred),
        "best_thr": thr,
        "BalAcc@best": balanced_accuracy_score(y, pred_opt),
        "MCC@best": matthews_corrcoef(y, pred_opt),
    }, bins


def run(name, path, label_col="y"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", label_col, "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df[label_col] = df[label_col].astype(int)
    df = df.reset_index(drop=True)

    X = featurize(df["smiles"].tolist())
    y = df[label_col].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr

    log("=" * 64)
    log(f"{name}")
    log("=" * 64)
    log(
        f"학습군 N={tr.sum()}, 양성 {y[tr].mean():.1%} | "
        f"검증군 N={te.sum()}, 양성 {y[te].mean():.1%}"
    )

    # 시간분할
    m = make_model().fit(X[tr], y[tr])
    p_t = m.predict_proba(X[te])[:, 1]
    r_t, bins_t = assess(y[te], p_t, "temporal")

    # 무작위 분할 (동일 크기 반복)
    rand_rows = []
    for i in range(N_RAND):
        rg = np.random.default_rng(SEED + i)
        idx = rg.permutation(len(y))
        ti, ri = idx[: te.sum()], idx[te.sum() :]
        if len(np.unique(y[ti])) < 2:
            continue
        mm = make_model().fit(X[ri], y[ri])
        pp = mm.predict_proba(X[ti])[:, 1]
        rand_rows.append(assess(y[ti], pp, "random")[0])
    rdf = pd.DataFrame(rand_rows)
    r_r = {k: rdf[k].mean() for k in rdf.columns if k != "setting"}
    r_r["setting"] = "random"

    # ------------------------------------------------------ 출력
    log("\n[핵심 지표]")
    log(f"{'':22s} {'무작위':>10s} {'시간분할':>10s} {'차이':>10s}")
    for k, fmt, better in [
        ("AUROC", "{:.4f}", "높을수록"),
        ("Brier", "{:.4f}", "낮을수록"),
        ("ECE", "{:.4f}", "낮을수록"),
        ("MCE", "{:.4f}", "낮을수록"),
        ("calib_slope", "{:.4f}", "1에 가까울수록"),
        ("calib_intercept", "{:.4f}", "0에 가까울수록"),
        ("BalAcc@0.5", "{:.4f}", "높을수록"),
        ("MCC@0.5", "{:.4f}", "높을수록"),
    ]:
        log(
            f"  {k:20s} {fmt.format(r_r[k]):>10s} {fmt.format(r_t[k]):>10s} "
            f"{r_t[k]-r_r[k]:+10.4f}   ({better} 좋음)"
        )

    log("\n[확률 치우침]")
    log(f"  실제 유병률      무작위 {r_r['prevalence']:.3f} | " f"시간분할 {r_t['prevalence']:.3f}")
    log(f"  평균 예측확률    무작위 {r_r['mean_pred']:.3f} | " f"시간분할 {r_t['mean_pred']:.3f}")
    log(
        f"  치우침(예측-실제) 무작위 {r_r['mean_pred']-r_r['prevalence']:+.3f} | "
        f"시간분할 {r_t['mean_pred']-r_t['prevalence']:+.3f}"
    )

    log("\n[임계값 이동의 영향]")
    log(f"  최적 임계값(Youden J): {r_t['best_thr']:.3f} (기본 0.5)")
    log(
        f"  BalAcc  0.5에서 {r_t['BalAcc@0.5']:.4f} -> "
        f"최적에서 {r_t['BalAcc@best']:.4f} "
        f"({r_t['BalAcc@best']-r_t['BalAcc@0.5']:+.4f})"
    )
    log(
        f"  MCC     0.5에서 {r_t['MCC@0.5']:.4f} -> "
        f"최적에서 {r_t['MCC@best']:.4f} "
        f"({r_t['MCC@best']-r_t['MCC@0.5']:+.4f})"
    )

    log("\n[신뢰도 구간별 보정 (시간분할)]")
    log(f"  {'구간':12s} {'n':>5s} {'예측':>8s} {'실제':>8s} {'격차':>8s}")
    for _, b in bins_t.iterrows():
        log(
            f"  {b['bin']:12s} {int(b['n']):5d} {b['pred']:8.3f} "
            f"{b['actual']:8.3f} {b['gap']:+8.3f}"
        )
    log()

    out = pd.DataFrame([r_r, r_t])
    out.insert(0, "dataset", name)
    return out


def main():
    results = []
    p1 = PROC / "dilirank2_with_year.csv"
    p2 = PROC / "dictrank_processed.csv"

    if p1.exists():
        results.append(run("DILIrank (간독성)", p1))
    if p2.exists():
        results.append(run("DICTrank (심장독성)", p2))

    if results:
        allr = pd.concat(results, ignore_index=True)
        allr.to_csv(RES / "calibration_results.csv", index=False)

        log("=" * 64)
        log("종합")
        log("=" * 64)
        for ds in allr["dataset"].unique():
            s = allr[allr.dataset == ds]
            r = s[s.setting == "random"].iloc[0]
            t = s[s.setting == "temporal"].iloc[0]
            log(f"\n{ds}")
            log(
                f"  AUROC 저하     {r['AUROC']:.3f} -> {t['AUROC']:.3f} "
                f"({t['AUROC']-r['AUROC']:+.3f})"
            )
            log(
                f"  ECE 악화       {r['ECE']:.3f} -> {t['ECE']:.3f} "
                f"({t['ECE']-r['ECE']:+.3f}, "
                f"{t['ECE']/r['ECE']:.1f}배)"
            )
            log(f"  보정기울기     {r['calib_slope']:.3f} -> " f"{t['calib_slope']:.3f}")
            log(
                f"  MCC@0.5 저하   {r['MCC@0.5']:.3f} -> {t['MCC@0.5']:.3f} "
                f"({t['MCC@0.5']-r['MCC@0.5']:+.3f})"
            )

        (RES / "calibration_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
        log(f"\n저장: {RES/'calibration_results.csv'}")
        log(f"      {RES/'calibration_summary.txt'}")


if __name__ == "__main__":
    main()
