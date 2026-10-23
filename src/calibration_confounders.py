"""
[보완 2] 교란 설명을 ECE 를 결과변수로 재검증

문제:
  논문의 주장은 "시간분할에서 무너지는 것은 판별력보다 보정"이다.
  그런데 교란 설명 세 가지 중 ECE 를 기준으로 검증된 것은 유병률 하나다.
    - 유병률 이동   : prevalence_control.py 가 ECE 로 검증  (O)
    - 라벨 성숙도   : censoring.py 가 AUROC 만 측정         (X)
    - 화학공간 이동 : stats_analysis.py 가 AUROC 만 측정     (X)
  심사자는 "라벨 검열이 ECE 붕괴를 설명하는가"를 직접 묻는다.

이 스크립트가 채우는 것:
  A) 라벨 성숙도 통제를 ECE·reliability 로 반복
       A1 검증군 노출기간 절단 (>=8, >=10, >=12년)
       A2 학습군·검증군 동시 절단 (공정 비교)
     각 조건마다 같은 크기의 무작위 분할 대조군을 새로 뽑아 ECE 배율을 낸다.
     (검증군 크기가 달라지면 무작위 ECE 도 달라지므로 재계산이 필수다.)
  B) 화학공간 유사도 구간별 ECE
       검증군을 최근접 Tanimoto 로 층화해 구간별 ECE·reliability 를 낸다.
       구간별 무작위 대조군은 정의할 수 없으므로 절대값과 추세만 본다.
  C) 승인 연도구간별 ECE (성숙도의 대리 지표)

사용법:
    python src/calibration_confounders.py [--reps N]

출력:
    results/calibration_confounders.csv
    results/calibration_confounders_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
N_REP_DEFAULT = 20
REF_YEAR = {"DILIrank": 2025, "DICTrank": 2023}  # censoring.py 와 동일
log = S.Log()


def rand_baseline(X, y, n_test, n_rep, target_rate=None):
    """같은 크기(선택적으로 같은 양성률)의 무작위 분할 평균 ECE/AUROC/rel/res."""
    out = {k: [] for k in ("ece", "auroc", "reliability", "resolution", "uncertainty", "brier")}
    for i in range(n_rep):
        r = np.random.default_rng(S.SEED + i)
        idx = r.permutation(len(y))
        ti, ri = idx[:n_test], idx[n_test:]
        if target_rate is not None:
            ti = S.subsample_to_rate(y, ti, target_rate, r)
        if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
            continue
        m = S.make_model().fit(X[ri], y[ri])
        p = m.predict_proba(X[ti])[:, 1]
        d = S.decompose(y[ti], p)
        out["ece"].append(S.ece(y[ti], p))
        out["auroc"].append(roc_auc_score(y[ti], p))
        out["reliability"].append(d["reliability"])
        out["resolution"].append(d["resolution"])
        out["uncertainty"].append(d["uncertainty"])
        out["brier"].append(brier_score_loss(y[ti], p))
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}


def nn_similarity(smiles_tr, smiles_te):
    """검증군 각 화합물의 학습군 최근접 Tanimoto."""

    def fp(s):
        return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)

    ftr = [fp(s) for s in smiles_tr]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), ftr)) for s in smiles_te])


def run(name, path, n_rep):
    df = S.load(path)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr, te = S.temporal_masks(df)
    ref = REF_YEAR[name]

    log("=" * 78)
    log(f"{name}   학습 {tr.sum()} / 검증 {te.sum()}   라벨 관측 종료 {ref}년")
    log("=" * 78)

    m0 = S.make_model().fit(X[tr], y[tr])
    p_t = m0.predict_proba(X[te])[:, 1]
    yte, prob = y[te], p_t
    base_r = rand_baseline(X, y, int(te.sum()), n_rep)
    base_pm = rand_baseline(X, y, int(te.sum()), n_rep, target_rate=float(y[te].mean()))
    rows = []

    def emit(tag, group, yy, pp, rnd, extra=None, rnd_pm=None):
        d = S.decompose(yy, pp)
        e = S.ece(yy, pp)
        row = dict(
            dataset=name,
            group=group,
            condition=tag,
            n=len(yy),
            pos_rate=float(yy.mean()),
            ece=e,
            auroc=roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan,
            reliability=d["reliability"],
            resolution=d["resolution"],
            uncertainty=d["uncertainty"],
            brier=brier_score_loss(yy, pp),
            ece_random=rnd.get("ece", np.nan),
            auroc_random=rnd.get("auroc", np.nan),
            rel_random=rnd.get("reliability", np.nan),
            res_random=rnd.get("resolution", np.nan),
        )
        row["ece_ratio"] = (
            row["ece"] / row["ece_random"]
            if row["ece_random"] and np.isfinite(row["ece_random"])
            else np.nan
        )
        # 유병률까지 맞춘 대조군 (크기 + 양성률 동시 일치 => uncertainty 도 일치)
        rp = rnd_pm or {}
        row["ece_random_pm"] = rp.get("ece", np.nan)
        row["auroc_random_pm"] = rp.get("auroc", np.nan)
        row["rel_random_pm"] = rp.get("reliability", np.nan)
        row["res_random_pm"] = rp.get("resolution", np.nan)
        row["ece_ratio_pm"] = (
            row["ece"] / row["ece_random_pm"]
            if row["ece_random_pm"] and np.isfinite(row["ece_random_pm"])
            else np.nan
        )
        dRp = (
            row["reliability"] - row["rel_random_pm"]
            if np.isfinite(row["rel_random_pm"])
            else np.nan
        )
        dSp = (
            row["res_random_pm"] - row["resolution"]
            if np.isfinite(row["res_random_pm"])
            else np.nan
        )
        row["share_reliability_pm"] = (
            100 * dRp / (dRp + dSp) if np.isfinite(dRp) and (dRp + dSp) else np.nan
        )
        dR = row["reliability"] - row["rel_random"] if np.isfinite(row["rel_random"]) else np.nan
        dS = row["res_random"] - row["resolution"] if np.isfinite(row["res_random"]) else np.nan
        row["d_reliability"], row["d_resolution"] = dR, dS
        row["share_reliability"] = 100 * dR / (dR + dS) if np.isfinite(dR) and (dR + dS) else np.nan
        if extra:
            row.update(extra)
        rows.append(row)
        return row

    r0 = emit("T. 시간분할 (기준)", "기준", yte, prob, base_r, rnd_pm=base_pm)
    log(f"\n[기준] 시간분할  n={r0['n']}  양성 {r0['pos_rate']:.1%}")
    log(
        f"   ECE {r0['ece']:.4f}  | 크기맞춤 무작위 {r0['ece_random']:.4f} "
        f"(배율 {r0['ece_ratio']:.2f})  | 크기+유병률맞춤 무작위 "
        f"{r0['ece_random_pm']:.4f} (배율 {r0['ece_ratio_pm']:.2f})"
    )
    log(f"   AUROC {r0['auroc']:.4f}")
    log(
        f"   reliability {r0['reliability']:.4f}  resolution {r0['resolution']:.4f}  "
        f"보정 기여 {r0['share_reliability']:.1f}%"
    )

    # ------------------------------------------------ A1 검증군 노출기간 절단
    log("\n--- A1. 검증군 노출기간 절단 (라벨이 성숙한 약물만) ---")
    log("    ECE 배율이 기준보다 크게 내려가면 라벨 검열이 보정 붕괴에 기여한다는 뜻")
    for min_exp in (8, 10, 12):
        keep = te & (yr <= ref - min_exp)
        n = int(keep.sum())
        if n < 40 or len(np.unique(y[keep])) < 2:
            log(f"    >={min_exp}년: n={n} - 표본 부족")
            continue
        pk = m0.predict_proba(X[keep])[:, 1]
        rate_k = float(y[keep].mean())
        rnd = rand_baseline(X, y, n, n_rep)
        rnd_pm = rand_baseline(X, y, n, n_rep, target_rate=rate_k)
        r = emit(
            f"A1 노출>={min_exp}년",
            "라벨성숙도",
            y[keep],
            pk,
            rnd,
            {"min_exposure": min_exp},
            rnd_pm=rnd_pm,
        )
        log(
            f"    >={min_exp}년: n={n:3d} 양성 {r['pos_rate']:5.1%}  "
            f"ECE {r['ece']:.4f}  AUROC {r['auroc']:.4f}"
        )
        log(
            f"        크기맞춤      무작위 {r['ece_random']:.4f} "
            f"배율 {r['ece_ratio']:.2f}  보정기여 {r['share_reliability']:5.1f}%  "
            f"[기준 {r0['ece_ratio']:.2f} 대비 {r['ece_ratio']-r0['ece_ratio']:+.2f}]"
        )
        log(
            f"        크기+유병률맞춤 무작위 {r['ece_random_pm']:.4f} "
            f"배율 {r['ece_ratio_pm']:.2f}  보정기여 "
            f"{r['share_reliability_pm']:5.1f}%  "
            f"[기준 {r0['ece_ratio_pm']:.2f} 대비 "
            f"{r['ece_ratio_pm']-r0['ece_ratio_pm']:+.2f}]  <-- 유병률 분리된 값"
        )

    # ------------------------------------------------ A2 양측 동시 절단
    log("\n--- A2. 학습군·검증군 동시 절단 (공정 비교) ---")
    for min_exp in (10,):
        trm = tr & (yr <= ref - min_exp)
        tem = te & (yr <= ref - min_exp)
        n = int(tem.sum())
        if n < 40 or trm.sum() < 100 or len(np.unique(y[tem])) < 2:
            log(f"    >={min_exp}년: 표본 부족 (학습 {trm.sum()}, 검증 {n})")
            continue
        mm = S.make_model().fit(X[trm], y[trm])
        pk = mm.predict_proba(X[tem])[:, 1]
        sub = trm | tem
        rate_k = float(y[tem].mean())
        rnd = rand_baseline(X[sub], y[sub], n, n_rep)
        rnd_pm = rand_baseline(X[sub], y[sub], n, n_rep, target_rate=rate_k)
        r = emit(
            f"A2 양측절단>={min_exp}년",
            "라벨성숙도",
            y[tem],
            pk,
            rnd,
            {"min_exposure": min_exp},
            rnd_pm=rnd_pm,
        )
        log(
            f"    >={min_exp}년: 학습 {trm.sum()} / 검증 {n} 양성 "
            f"{r['pos_rate']:5.1%}  ECE {r['ece']:.4f}  AUROC {r['auroc']:.4f}"
        )
        log(
            f"        크기맞춤      무작위 {r['ece_random']:.4f} "
            f"배율 {r['ece_ratio']:.2f}  보정기여 {r['share_reliability']:5.1f}%"
        )
        log(
            f"        크기+유병률맞춤 무작위 {r['ece_random_pm']:.4f} "
            f"배율 {r['ece_ratio_pm']:.2f}  보정기여 "
            f"{r['share_reliability_pm']:5.1f}%  <-- 유병률 분리된 값"
        )

    # ------------------------------------------------ B 유사도 구간별 ECE
    log("\n--- B. 화학공간 유사도 구간별 ECE (구간별 무작위 대조군 없음) ---")
    sim = nn_similarity(df.loc[tr, "smiles"].tolist(), df.loc[te, "smiles"].tolist())
    log(f"    검증군 최근접 Tanimoto 중앙값 {np.median(sim):.4f}  평균 {sim.mean():.4f}")
    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]:
        msk = (sim >= lo) & (sim < hi)
        n = int(msk.sum())
        if n < 20 or len(np.unique(yte[msk])) < 2:
            log(f"    [{lo:.1f}-{hi:.1f}) n={n:3d} - 표본 부족, 계산 안 함")
            continue
        r = emit(f"B 유사도 [{lo:.1f}-{hi:.1f})", "화학공간", yte[msk], prob[msk], {})
        log(
            f"    [{lo:.1f}-{hi:.1f}) n={n:3d} 양성 {r['pos_rate']:5.1%}  "
            f"ECE {r['ece']:.4f}  AUROC {r['auroc']:.4f}  "
            f"reliability {r['reliability']:.4f}"
        )
    hi_m = sim >= np.median(sim)
    for tag, msk in [("B 유사도 상위 50%", hi_m), ("B 유사도 하위 50%", ~hi_m)]:
        if len(np.unique(yte[msk])) < 2:
            continue
        r = emit(tag, "화학공간", yte[msk], prob[msk], {})
        log(
            f"    {tag}: n={r['n']:3d} ECE {r['ece']:.4f}  "
            f"AUROC {r['auroc']:.4f}  reliability {r['reliability']:.4f}"
        )

    # ------------------------------------------------ C 연도구간별 ECE
    log("\n--- C. 승인 연도구간별 ECE (성숙도 대리 지표) ---")
    yte_year = yr[te]
    for a, b in [(2010, 2013), (2014, 2017), (2018, 2026)]:
        msk = (yte_year >= a) & (yte_year <= b)
        n = int(msk.sum())
        if n < 20 or len(np.unique(yte[msk])) < 2:
            log(f"    {a}-{b}: n={n} - 표본 부족")
            continue
        r = emit(f"C {a}-{b}", "연도구간", yte[msk], prob[msk], {})
        log(
            f"    {a}-{b}: n={n:3d} 양성 {r['pos_rate']:5.1%}  "
            f"ECE {r['ece']:.4f}  AUROC {r['auroc']:.4f}  "
            f"reliability {r['reliability']:.4f}"
        )
    log("")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_REP_DEFAULT)
    a = ap.parse_args()
    log("교란 설명의 ECE 기준 재검증")
    log(f"무작위 대조군 {a.reps}회, seed {S.SEED}, 절단 {S.CUTOFF}\n")
    out = []
    for nm, fn in S.DATASETS:
        p = PROC / fn
        if p.exists():
            out.append(run(nm, p, a.reps))
    if not out:
        log("데이터 없음")
        return
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "calibration_confounders.csv", index=False)

    log("=" * 78)
    log("판정 기준")
    log("=" * 78)
    log("A1/A2 — 라벨 성숙도를 통제했을 때 ECE 배율이 1 에 가까워지면")
    log("        라벨 검열이 보정 붕괴의 원인이라는 뜻이다.")
    log("        기준 배율과 비슷하게 남으면 라벨 검열로 설명되지 않는다.")
    log("B/C  — 구간별 무작위 대조군을 정의할 수 없으므로 배율은 없다.")
    log("        절대 ECE 와 reliability 의 추세만 본다. 유사도가 높은 구간에서")
    log("        ECE 가 확연히 낮아지면 화학공간 이동이 보정 붕괴에 기여한다.")
    log("")
    log(
        f"{'데이터셋':9s} {'조건':22s} {'n':>5s} {'양성':>6s} {'ECE':>8s} "
        f"{'배율(크기)':>10s} {'배율(+유병률)':>12s} {'보정기여(+유병률)':>16s}"
    )
    log("-" * 78)

    def g(v, f="{:.2f}"):
        return f.format(v) if np.isfinite(v) else "-"

    for _, r in df.iterrows():
        log(
            f"{r.dataset:9s} {r.condition:22s} {r.n:5d} "
            f"{r.pos_rate:5.1%} {r.ece:8.4f} "
            f"{g(r.ece_ratio):>10s} {g(r.ece_ratio_pm):>12s} "
            f"{g(r.share_reliability_pm, '{:.1f}%'):>16s}"
        )
    log("")
    log("[읽는 법] '배율(+유병률)' 이 성숙도 통제의 순수 효과다. 크기만 맞춘 배율은")
    log("          절단이 유병률도 함께 바꾸기 때문에 두 효과가 섞여 있다.")
    log.save(RES / "calibration_confounders_summary.txt")
    print(f"\n저장: {RES/'calibration_confounders.csv'}")
    print(f"      {RES/'calibration_confounders_summary.txt'}")


if __name__ == "__main__":
    main()
