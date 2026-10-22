# -*- coding: utf-8 -*-
"""처방이 실패했다는 주장에 검정력을 붙인다 — 최소 검출 가능 개선폭.

왜 필요한가 (음성 결과의 급소)
-----------------------------
이 논문의 결론 중 상당 부분이 음성 결과다.

  §20  최신성 가중은 듣지 않는다
  §21  적용범위 필터는 듣지 않는다
  §23  학습 시 클래스 가중은 사후 보정보다 낫지 않다
  §10  학습군 내부 재보정은 일부만 되돌린다

음성 결과를 주장하려면 "효과가 있었다면 볼 수 있었는가" 에 답해야 한다.
검증군이 156개·267개뿐이므로, 작은 개선은 애초에 검출할 수 없다. 지금
저장소에는 그 한계를 적은 수치가 하나도 없다. 그대로 투고하면 심사자가
한 줄로 정리한다 — "효과가 없는 것과 검출하지 못한 것을 구분하지 않았다."

이 스크립트가 하는 것
--------------------
각 처방을 무보정과 같은 검증군에서 짝지어 부트스트랩한다. 예측은
고정하고 검증군만 재표집하므로 두 팔의 상관이 보존된다(짝지은 설계는
독립 비교보다 분산이 작다 — 그래서 검출력이 높다).

  관측 ΔECE          처방 ECE − 무보정 ECE. 음수면 개선이다.
  ΔECE 의 95% 구간    짝지은 부트스트랩 백분위.
  SE(ΔECE)           부트스트랩 표준편차.
  최소 검출 개선폭    MDE80 = 2.80 x SE  (양측 alpha 0.05, 검정력 0.80).
                     이 크기 이상으로 개선했다면 유의하게 잡혔을 것이다.
  메워야 할 간격      무보정 ECE − 무작위 분할 대조 ECE (§8 R1).
                     처방이 '쓸 만하다' 는 말은 이 간격을 상당히 메운다는 뜻이다.
  MDE / 간격          이 연구가 검출할 수 있었던 최소 개선이 간격의 몇 %인가.

판정 문구는 셋 중 하나로만 쓴다.
  (가) 개선이 유의하다              구간이 0 보다 작은 쪽에 있다
  (나) 개선이 없음을 보였다          구간이 0 을 포함하고 그 폭이 간격보다 작다
  (다) 검출하지 못했다 (판정 보류)    구간이 0 을 포함하고 폭이 간격만큼 넓다

중요 — 모든 팔을 §6·§8 과 같은 원본 행 순서에서 다시 계산한다.
§10 은 보정셋을 떼기 위해 승인연도로 정렬한 뒤 적합하므로 수치가 조금
다르다(§10 의 주석 참조). 짝지은 비교는 같은 적합 위에서만 뜻이 있으므로
여기서는 정렬하지 않고, 대신 §10 값과의 차이를 함께 찍는다.

사용법
------
    python src/remedy_power.py                 # 부트스트랩 4000회
    python src/remedy_power.py --boot 500      # 짧게 점검

출력
----
    results/remedy_power.csv
    results/remedy_power_ad.csv
    results/remedy_power_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

CALIB_FRAC = 0.25  # §10 과 같은 보정셋 비율
Z80 = 1.959963985 + 0.841621234  # 양측 0.05 + 검정력 0.80 = 2.8016
TOL = 5e-4


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), S.EPS, 1 - S.EPS)
    return np.log(p / (1 - p))


def platt(p_cal, y_cal, p_te):
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


def isotonic(p_cal, y_cal, p_te):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(p_cal, y_cal)
    return np.clip(ir.predict(p_te), S.EPS, 1 - S.EPS)


def prior_shift(p_te, src, tgt):
    odds = (tgt / (1 - tgt)) / (src / (1 - src))
    return 1.0 / (1.0 + np.exp(-(logit(p_te) + np.log(odds))))


def spw(y_tr, pi_target):
    npos = int((y_tr == 1).sum())
    nneg = int((y_tr == 0).sum())
    if npos == 0 or nneg == 0 or not (0 < pi_target < 1):
        return 1.0
    return (pi_target / (1.0 - pi_target)) / (npos / float(nneg))


def paired_boot(y, p0, p1, n_boot, seed):
    """같은 검증군을 재표집해 ΔECE 분포를 만든다. 예측은 고정한다."""
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(S.ece(y[i], p1[i]) - S.ece(y[i], p0[i]))
    d = np.asarray(d)
    return (
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
        float(d.std(ddof=1)),
        len(d),
    )


def tanimoto_max(fp_te, fp_tr):
    """검증 분자별 학습군 최근접 Tanimoto 유사도 (ECFP4 2048비트 블록)."""
    A = (fp_te > 0).astype(np.float32)
    B = (fp_tr > 0).astype(np.float32)
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = inter / (a + b - inter)
    t = np.nan_to_num(t, nan=0.0)
    return t.max(1)


# 처방이 '쓸 만하다' 의 사전 기준 — 메워야 할 간격의 이 비율 이상을 줄여야 한다.
# 구간 폭을 간격과 비교하는 방식은 틀렸다(2026-10-19 수정). 폭이 좁아도 구간이
# 허용하는 최대 개선이 크면 '개선이 없다' 고 말할 수 없다. 그래서 구간의
# 가장 유리한 끝(lo)을 기준과 비교한다.
USEFUL = 0.25


def verdict(lo, hi, gap):
    if hi < 0:
        return "(가) 개선이 유의하다"
    if lo > 0:
        return "(가') 오히려 유의하게 나빠진다"
    if gap != gap or gap <= 0:
        return "(다) 검출하지 못했다 — 판정 보류"
    # 구간이 허용하는 최대 개선(= -lo)이 쓸 만한 크기에 못 미치면,
    # '쓸 만한 개선은 없다' 를 실제로 보인 것이다.
    if -lo < USEFUL * gap:
        return "(나) 쓸 만한 개선이 없음을 보였다"
    return "(다) 검출하지 못했다 — 판정 보류"


def run(name, fn, n_boot, ref_pc, ref_rec, ref_rw, ref_cw):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rate_tr, rate_te = float(y[tr].mean()), float(y[te].mean())

    # 기준 팔 — §6·§8 과 완전히 같은 적합 (정렬하지 않는다)
    p_A = S.make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ece_A = S.ece(y[te], p_A)

    log("=" * 126)
    log(
        "%s   학습 %d / 검증 %d   무보정 ECE %.4f   부트스트랩 %d회"
        % (name, len(tr), len(te), ece_A, n_boot)
    )
    r0 = ref_pc[(ref_pc.dataset.str.startswith(name)) & (ref_pc.setting.str.startswith("T"))]
    gap = float("nan")
    if len(r0):
        d0 = abs(float(r0.ECE.iloc[0]) - ece_A)
        log(
            "  §8 시간분할 ECE 와의 차 %.2e → %s"
            % (d0, "일치" if d0 < TOL else "**불일치 — 중단**")
        )
        assert d0 < TOL, "§8 재현 실패"
    r1 = ref_pc[(ref_pc.dataset.str.startswith(name)) & (ref_pc.setting.str.startswith("R1"))]
    if len(r1):
        gap = ece_A - float(r1.ECE.iloc[0])
        log(
            "  메워야 할 간격 = 무보정 %.4f − 무작위 대조 %.4f = %.4f"
            % (ece_A, float(r1.ECE.iloc[0]), gap)
        )
    log("=" * 126)

    # 보정셋 — 학습군의 최근 25% (연도 순위로 정하고 행은 재배열하지 않는다)
    rank = np.argsort(np.argsort(yr[tr], kind="stable"))
    n_cal = int(len(tr) * CALIB_FRAC)
    cal = tr[rank >= len(tr) - n_cal]
    fit = tr[rank < len(tr) - n_cal]
    m_fit = S.make_model().fit(X[fit], y[fit])
    p_cal = m_fit.predict_proba(X[cal])[:, 1]
    p_te_fit = m_fit.predict_proba(X[te])[:, 1]

    arms = [
        ("A'. 무보정 (보정셋 제외 학습)", p_te_fit, "참고"),
        ("B. Platt (학습군 내부)", platt(p_cal, y[cal], p_te_fit), "배포 가능"),
        ("C. Isotonic (학습군 내부)", isotonic(p_cal, y[cal], p_te_fit), "배포 가능"),
        ("D. 사전확률 보정", prior_shift(p_A, rate_tr, rate_te), "검증군 양성률 가정"),
    ]

    # 최신성 가중 — τ* 는 §20 의 내부 선택 결과를 그대로 쓴다
    tau_star, tau_lab = None, "없음"
    if ref_rw is not None:
        g = ref_rw[(ref_rw.dataset == name) & (ref_rw.kind == "inner")]
        if len(g):
            b = g.loc[g.ece.idxmin()]
            tau_star = None if pd.isna(b.tau) else float(b.tau)
            tau_lab = str(b.tag)
    if tau_star is not None:
        w = 0.5 ** ((S.CUTOFF - yr[tr]) / tau_star)
        p_R = S.make_model().fit(X[tr], y[tr], sample_weight=w).predict_proba(X[te])[:, 1]
        arms.append(("R. 최신성 가중 τ*=%g (내부 선택)" % tau_star, p_R, "배포 가능"))
    else:
        log(
            "  §20 내부 선택이 '가중 없음' 을 골랐다 (%s) — 최신성 팔은 "
            "무보정과 같으므로 넣지 않는다." % tau_lab
        )

    # 클래스 가중 oracle — §23 W1
    # §23 은 scale_pos_weight 가 아니라 양성에만 주는 sample_weight 로
    # 같은 가중을 건다(class_weighting.fit_w). 두 경로는 목적함수상 같지만
    # XGBoost 내부 적용 지점이 달라 마지막 자리가 어긋난다(실측 0.005 수준).
    # 짝지은 비교는 같은 적합 위에서만 뜻이 있으므로 §23 의 경로를 그대로 쓴다.
    w1 = spw(y[tr], rate_te)
    _sw = np.where(y[tr] == 1, float(w1), 1.0)
    p_W = S.make_model().fit(X[tr], y[tr], sample_weight=_sw).predict_proba(X[te])[:, 1]
    arms.append(("W1. 클래스 가중 oracle (spw=%.4f)" % w1, p_W, "검증군 양성률 가정"))

    log(
        "%-36s %9s %9s %-20s %8s %9s %9s"
        % ("처방", "ECE", "ΔECE", "Δ 95% 구간", "SE", "MDE80", "MDE/간격")
    )
    log("-" * 126)
    rows = []
    for lab, p1, kind in arms:
        e1 = S.ece(y[te], p1)
        lo, hi, se, nb = paired_boot(y[te], p_A, p1, n_boot, S.SEED + 7)
        mde = Z80 * se
        rows.append(
            {
                "dataset": name,
                "remedy": lab,
                "deployable": kind,
                "ece_base": ece_A,
                "ece_remedy": e1,
                "delta": e1 - ece_A,
                "delta_lo": lo,
                "delta_hi": hi,
                "se": se,
                "mde80": mde,
                "gap_to_control": gap,
                "mde_over_gap": mde / gap if gap == gap and gap > 0 else float("nan"),
                "auroc_remedy": float(roc_auc_score(y[te], p1)),
                "n_boot_ok": nb,
                "verdict": verdict(lo, hi, gap),
            }
        )
        log(
            "%-36s %9.4f %+9.4f [%+.4f, %+.4f] %8.4f %9.4f %8.1f%%"
            % (
                lab,
                e1,
                e1 - ece_A,
                lo,
                hi,
                se,
                mde,
                100.0 * mde / gap if gap == gap and gap > 0 else float("nan"),
            )
        )
    log("")
    for r in rows:
        log("  %-36s %s" % (r["remedy"], r["verdict"]))
    log("")

    # §10 과의 차이를 자진 신고한다
    if ref_rec is not None:
        g = ref_rec[ref_rec.dataset.str.startswith(name)]
        log("  §10(승인연도 정렬 후 적합) 과의 ECE 차 — 짝지은 비교를 위해")
        log("  여기서는 정렬하지 않았다:")
        for pref, lab in (("B.", "Platt"), ("C.", "Isotonic"), ("D.", "사전확률 보정")):
            q = g[g.method.str.startswith(pref)]
            w2 = [r for r in rows if r["remedy"].startswith(pref)]
            if len(q) and w2:
                log(
                    "    %-14s §10 %.4f / 여기 %.4f (차 %+.4f)"
                    % (
                        lab,
                        float(q.ECE.iloc[0]),
                        w2[0]["ece_remedy"],
                        w2[0]["ece_remedy"] - float(q.ECE.iloc[0]),
                    )
                )
    if ref_cw is not None:
        q = ref_cw[(ref_cw.dataset == name) & (ref_cw.arm.str.startswith("W1"))]
        w2 = [r for r in rows if r["remedy"].startswith("W1")]
        if len(q) and w2:
            d = w2[0]["ece_remedy"] - float(q.ece.iloc[0])
            log("  §23 W1 과의 차 %+.4f → %s" % (d, "일치" if abs(d) < TOL else "**확인 필요**"))
    if ref_rw is not None and tau_star is not None:
        q = ref_rw[(ref_rw.dataset == name) & (ref_rw.kind == "weight") & (ref_rw.tau == tau_star)]
        w2 = [r for r in rows if r["remedy"].startswith("R.")]
        if len(q) and w2:
            d = w2[0]["ece_remedy"] - float(q.ece.iloc[0])
            log(
                "  §20 τ*=%g 와의 차 %+.4f → %s"
                % (tau_star, d, "일치" if abs(d) < TOL else "**확인 필요**")
            )
    log("")

    # ---- 적용범위 필터는 n 이 달라지므로 따로 ----
    fp = X[:, :2048]
    sim = tanimoto_max(fp[te], fp[tr])
    ad_rows = []
    log("  적용범위 필터 — 유사도 낮은 순으로 버린다. 같은 수를 무작위로")
    log("  버리는 대조와 짝지어 부트스트랩한다 (§21 과 같은 설계).")
    log(
        "%-10s %8s %10s %10s %+10s %-20s %8s"
        % ("버림 비율", "남는 n", "필터 ECE", "무작위 ECE", "Δ", "Δ 95% 구간", "MDE80")
    )
    log("-" * 126)
    for f in (0.1, 0.2, 0.3, 0.4, 0.5):
        n_keep = int(round(len(te) * (1 - f)))
        keep = np.argsort(-sim)[:n_keep]
        e_ad = S.ece(y[te][keep], p_A[keep])
        r = np.random.default_rng(S.SEED + 11)
        e_rd = float(
            np.mean(
                [
                    S.ece(y[te][k], p_A[k])
                    for k in (r.choice(len(te), n_keep, replace=False) for _ in range(200))
                    if len(np.unique(y[te][k])) > 1
                ]
            )
        )
        # 짝지은 부트스트랩: 재표집 안에서 두 설계를 다시 만든다
        rb = np.random.default_rng(S.SEED + 12)
        d = []
        for _ in range(n_boot):
            i = rb.integers(0, len(te), len(te))
            yy, pp, ss = y[te][i], p_A[i], sim[i]
            if len(np.unique(yy)) < 2:
                continue
            k1 = np.argsort(-ss)[:n_keep]
            k2 = rb.choice(len(i), n_keep, replace=False)
            if len(np.unique(yy[k1])) < 2 or len(np.unique(yy[k2])) < 2:
                continue
            d.append(S.ece(yy[k1], pp[k1]) - S.ece(yy[k2], pp[k2]))
        d = np.asarray(d)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        se = float(d.std(ddof=1))
        mde = Z80 * se
        ad_rows.append(
            {
                "dataset": name,
                "drop_frac": f,
                "n_keep": n_keep,
                "ece_ad": e_ad,
                "ece_random_drop": e_rd,
                "delta": e_ad - e_rd,
                "delta_lo": lo,
                "delta_hi": hi,
                "se": se,
                "mde80": mde,
                "n_boot_ok": len(d),
                "verdict": verdict(lo, hi, gap),
            }
        )
        log(
            "%-10.1f %8d %10.4f %10.4f %+10.4f [%+.4f, %+.4f] %8.4f"
            % (f, n_keep, e_ad, e_rd, e_ad - e_rd, lo, hi, mde)
        )
    log("")
    return rows, ad_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    log("처방이 실패했다는 주장에 검정력을 붙인다 — 최소 검출 가능 개선폭")
    log("짝지은 부트스트랩 %d회 (예측 고정, 검증군만 재표집), 시드 %d" % (a.boot, S.SEED))
    log("MDE80 = %.4f x SE  (양측 alpha 0.05, 검정력 0.80)" % Z80)
    log("'쓸 만한 개선' 기준: 메워야 할 간격의 %.0f%% 이상" % (USEFUL * 100))
    log("")

    def rd(n):
        p = os.path.join(RES, n)
        return pd.read_csv(p) if os.path.exists(p) else None

    pc = rd("prevalence_control.csv")
    rec = rd("recalibration_results.csv")
    rw = rd("recency_weighting.csv")
    cw = rd("class_weighting.csv")
    assert pc is not None, "prevalence_control.csv 가 없다 — 먼저 §8 을 돌릴 것"

    rows, ad = [], []
    for name, fn in S.DATASETS:
        x, z = run(name, fn, a.boot, pc, rec, rw, cw)
        rows += x
        ad += z

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RES, "remedy_power.csv"), index=False, encoding="utf-8")
    A = pd.DataFrame(ad)
    A.to_csv(os.path.join(RES, "remedy_power_ad.csv"), index=False, encoding="utf-8")

    log("=" * 126)
    log("판정")
    log("=" * 126)
    log("[읽는 법] '효과가 없다' 와 '검출하지 못했다' 를 구분하는 표다.")
    log(
        "          (나) = 구간이 허용하는 최대 개선조차 간격의 %.0f%% 에 못 미친다."
        % (USEFUL * 100)
    )
    log("          (다) = 구간이 쓸 만한 개선을 아직 배제하지 못한다. 본문에서")
    log("          '개선이 없다' 로 쓰면 안 되고 '구간이 최대 X 까지 허용한다' 로 쓴다.")
    log("")
    for v, g in R.groupby("verdict"):
        log("  %s — %d개" % (v, len(g)))
        for _, r in g.iterrows():
            log("      %-10s %s" % (r.dataset, r.remedy))
    log("")
    _hold = R[R.verdict.str.startswith("(다)")]
    if len(_hold):
        log("  판정 보류 팔이 허용하는 최대 개선:")
        for _, r in _hold.iterrows():
            log(
                "      %-10s %-36s 최대 %.4f (간격의 %.0f%%)"
                % (r.dataset, r.remedy, -r.delta_lo, 100.0 * (-r.delta_lo) / r.gap_to_control)
            )
        log("")
    dep = R[R.deployable == "배포 가능"]
    if len(dep):
        log(
            "  배포 가능한 처방 %d개 중 개선이 유의한 것: %d개"
            % (len(dep), int(dep.verdict.str.startswith("(가)").sum()))
        )
    log("")
    for name, _ in S.DATASETS:
        g = R[R.dataset == name]
        if not len(g):
            continue
        log(
            "  %s — 간격 %.4f, 검출 가능한 최소 개선 %.4f~%.4f "
            "(간격의 %.0f~%.0f%%)"
            % (
                name,
                g.gap_to_control.iloc[0],
                g.mde80.min(),
                g.mde80.max(),
                100 * g.mde_over_gap.min(),
                100 * g.mde_over_gap.max(),
            )
        )
    log("")
    log("저장: %s" % os.path.join(RES, "remedy_power.csv"))
    log("      %s" % os.path.join(RES, "remedy_power_ad.csv"))
    log.save(Path(RES) / "remedy_power_summary.txt")


if __name__ == "__main__":
    main()
