# -*- coding: utf-8 -*-
"""최신성 가중 — 오래된 학습 분자의 비중을 낮추면 보정이 회복되는가.

왜 필요한가
----------
§18 에서 DILIrank 는 **2010년 이후 125개로 학습한 모델(AUROC 0.7290)이
pre-2010 640개로 학습한 모델(0.6476)을 이겼다.** 5배 적은 같은 시대
데이터가 5배 많은 과거 데이터를 이긴 것이다. 그러면 당연한 후속 질문이
남는다 — **오래된 분자의 가중치를 낮추면 되는가.**

§10 은 예측을 **사후에** 고치는 쪽(Platt·isotonic·사전확률 보정)을 다뤘고,
§10.2 는 검증군 라벨 없이 유병률을 추정하려는 시도가 **전부 실패**함을
보였다(EM −17.4%). 이 절은 방향이 다르다 — **학습 단계에서** 고친다.

설계
----
  [1] 최신성 가중 : w_i = 0.5^((2010 − 승인연도_i) / tau)
                   tau = 반감기(년). tau=inf 면 균등가중 = §8 설정.
  [2] 하드 절단   : 2010−K 년 이후 분자만 학습 (K년 창)
                   실무자가 먼저 시도할 단순 대안이다.
  검증군은 두 경우 모두 2010년 이후 전체로 고정한다.

  유효 표본 수 n_eff = (Σw)^2 / Σw^2 를 함께 적는다. 가중은 공짜가
  아니다 — 최근 쪽에 몰수록 실질 표본이 줄어든다. 그 교환을 보여야 한다.

**가장 중요한 부분: tau 를 어떻게 고르는가**
-------------------------------------------
전체 sweep 을 검증군에서 보고 제일 좋은 tau 를 고르면 그건 검증군 누출이다.
그래서 두 번째 팔을 붙인다.

  정직 프로토콜 : pre-2010 을 다시 둘로 가른다.
      내부 학습군 = 2000년 **미만**
      내부 검증군 = 2000~2009
    내부 검증군에서 ECE 가 최소인 tau* 를 고르고, 그 tau* 로 pre-2010
    **전체**를 다시 학습해 2010년 이후에서 평가한다. 미래를 보지 않고
    고를 수 있는 규칙이므로 실제로 배포 가능하다.

**이 프로토콜의 한계를 먼저 적는다.** 내부 이동(1990년대 → 2000년대)은
실제 이동(pre-2010 → post-2010)보다 **훨씬 작다.** 특히 유병률 변화가
실제 −0.3470(DILIrank)에 비해 내부에서는 거의 없다. 그래서 내부에서 고른
tau* 가 실제 이동으로 옮겨간다는 보장이 없다. 옮겨가지 않으면 그것도
답이다 — §10.2 의 맹목 추정량 실패와 같은 결론을 학습 쪽에서도 확인하는
셈이고, "시간 이동은 사후 보정으로도 사전 가중으로도 공짜로는 못 고친다"
가 논문의 정직한 결론이 된다.

사용법
------
    python src/recency_weighting.py

출력
----
    results/recency_weighting.csv
    results/recency_weighting_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

TAUS = [None, 30.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0]  # None = 균등가중
WINDOWS = [30, 20, 15, 10, 5]  # 하드 절단 K년
INNER_CUT = 2000  # 정직 프로토콜 내부 경계
N_BOOT = 2000


def metrics(y, p):
    d = S.decompose(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "ece": S.ece(y, p),
        "slope": S.calib_slope(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
    }


def weights(years, tau, cut=S.CUTOFF):
    """반감기 tau 의 지수 가중. tau 가 None 이면 균등."""
    if tau is None:
        return np.ones(len(years), dtype=float)
    return 0.5 ** ((cut - np.asarray(years, dtype=float)) / float(tau))


def n_eff(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w**2).sum())


def fit_predict(X, y, tr, te, w=None):
    m = S.make_model()
    if w is None:
        m.fit(X[tr], y[tr])
    else:
        m.fit(X[tr], y[tr], sample_weight=w)
    return m.predict_proba(X[te])[:, 1]


def boot_ece(y, p, n_boot=N_BOOT, seed=S.SEED):
    """검증군 재표집으로 ECE 의 불확실성. 예측은 고정이므로 재적합 없다."""
    r = np.random.default_rng(seed)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p[i]))
    if not v:
        return float("nan"), float("nan")
    v = np.sort(np.array(v))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run(name, fn):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    pre = np.where(tr_m)[0]
    post = np.where(te_m)[0]
    y_post = y[post]

    log("=" * 104)
    log(
        "%s   학습 pre-2010 %d개(%d~%d) / 검증 post-2010 %d개"
        % (name, len(pre), yr[pre].min(), yr[pre].max(), len(post))
    )
    log("  검증군은 모든 설정에서 동일하다. 적합은 설정당 1회(결정적)다.")
    log("=" * 104)
    log(
        "%-26s %8s %8s %8s %8s %8s %8s %8s %-18s"
        % ("설정", "학습n", "유효n", "학습률", "격차", "AUROC", "ECE", "기울기", "ECE 95% CI")
    )
    log("-" * 110)

    rows = []

    def record(tag, kind, tau, tr_idx, w, note=""):
        p = fit_predict(X, y, tr_idx, post, w)
        m = metrics(y_post, p)
        lo, hi = boot_ece(y_post, p)
        ne = n_eff(w) if w is not None else float(len(tr_idx))
        wr = float(np.average(y[tr_idx], weights=w)) if w is not None else float(y[tr_idx].mean())
        log(
            "%-26s %8d %8.1f %8.4f %+8.4f %8.4f %8.4f %8.4f [%.4f, %.4f]"
            % (
                tag,
                len(tr_idx),
                ne,
                wr,
                wr - float(y_post.mean()),
                m["auroc"],
                m["ece"],
                m["slope"],
                lo,
                hi,
            )
        )
        rows.append(
            {
                "dataset": name,
                "kind": kind,
                "tag": tag,
                "tau": np.nan if tau is None else tau,
                "n_train": len(tr_idx),
                "n_eff": ne,
                "train_rate_weighted": wr,
                "n_test": len(post),
                "test_rate": float(y_post.mean()),
                "gap": wr - float(y_post.mean()),
                "ece_lo": lo,
                "ece_hi": hi,
                "note": note,
                **m,
            }
        )
        return m

    # --- [1] 최신성 가중 sweep ------------------------------------
    base = None
    for tau in TAUS:
        tag = "가중 균등 (= §8)" if tau is None else "가중 반감기 %g년" % tau
        m = record(tag, "weight", tau, pre, weights(yr[pre], tau))
        if tau is None:
            base = m
    log("")

    # --- [2] 하드 절단 창 ------------------------------------------
    for K in WINDOWS:
        sel = pre[yr[pre] >= S.CUTOFF - K]
        if len(sel) < 60 or len(np.unique(y[sel])) < 2:
            log("%-26s 표본 부족(%d개) — 건너뜀" % ("절단 최근 %d년" % K, len(sel)))
            continue
        record("절단 최근 %d년" % K, "window", float(K), sel, None)
    log("")

    # --- [3] 정직 프로토콜 : 내부 검증군에서 tau* 를 고른다 ----------
    inner_tr = pre[yr[pre] < INNER_CUT]
    inner_te = pre[yr[pre] >= INNER_CUT]
    log("정직 프로토콜 — 미래를 보지 않고 tau 를 고른다")
    log(
        "  내부 학습 %d개(~%d) / 내부 검증 %d개(%d~2009)"
        % (len(inner_tr), INNER_CUT - 1, len(inner_te), yr[inner_te].min() if len(inner_te) else -1)
    )
    log(
        "  내부 유병률 이동 %+.4f  (실제 이동 %+.4f — 내부가 훨씬 작다)"
        % (
            float(y[inner_te].mean()) - float(y[inner_tr].mean()),
            float(y_post.mean()) - float(y[pre].mean()),
        )
    )
    log("")
    log("  %-22s %8s %8s %8s" % ("tau", "내부ECE", "내부AUROC", "내부유효n"))
    log("  " + "-" * 56)
    inner = []
    for tau in TAUS:
        w = weights(yr[inner_tr], tau, cut=INNER_CUT)
        p = fit_predict(X, y, inner_tr, inner_te, w)
        mm = metrics(y[inner_te], p)
        inner.append((tau, mm["ece"], mm["auroc"], n_eff(w)))
        log(
            "  %-22s %8.4f %8.4f %8.1f"
            % ("균등" if tau is None else "반감기 %g년" % tau, mm["ece"], mm["auroc"], n_eff(w))
        )
        rows.append(
            {
                "dataset": name,
                "kind": "inner",
                "tag": "내부 균등" if tau is None else "내부 반감기 %g년" % tau,
                "tau": np.nan if tau is None else tau,
                "n_train": len(inner_tr),
                "n_eff": n_eff(w),
                "train_rate_weighted": float(np.average(y[inner_tr], weights=w)),
                "n_test": len(inner_te),
                "test_rate": float(y[inner_te].mean()),
                "ece_lo": np.nan,
                "ece_hi": np.nan,
                "note": "내부 검증 (tau 선택용)",
                **mm,
            }
        )
    best = min(inner, key=lambda t: t[1])
    tau_star = best[0]
    log("")
    log(
        "  선택: tau* = %s  (내부 ECE %.4f 로 최소)"
        % ("균등" if tau_star is None else "반감기 %g년" % tau_star, best[1])
    )
    log("")
    sel_m = record(
        "정직선택 tau*=%s" % ("균등" if tau_star is None else "%g년" % tau_star),
        "honest",
        np.nan if tau_star is None else tau_star,
        pre,
        weights(yr[pre], tau_star),
        note="내부 검증으로 선택",
    )
    log("")
    return rows, base, sel_m


def main():
    argparse.ArgumentParser().parse_args()
    log("최신성 가중 — 학습 단계에서 시간 이동을 고칠 수 있는가")
    log("시드 %d, 부트스트랩 %d회. 적합은 설정당 1회(결정적)." % (S.SEED, N_BOOT))
    log("")
    rows, B, H = [], {}, {}
    for name, fn in S.DATASETS:
        rr, b, h = run(name, fn)
        rows += rr
        B[name], H[name] = b, h

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "recency_weighting.csv"), index=False, encoding="utf-8")

    log("=" * 104)
    log("판정")
    log("=" * 104)
    log("[읽는 법] 두 줄만 보면 된다 — **균등가중(= §8)** 대 **정직선택 tau***.")
    log("          정직선택이 균등보다 ECE 가 낮아야 '배포 가능한 처방' 이 된다.")
    log("          sweep 최솟값은 검증군을 보고 고른 값이므로 상한선일 뿐이고")
    log("          처방으로 인용하면 안 된다.")
    log("")
    for d in B:
        g = out[(out.dataset == d) & (out.kind == "weight")]
        bestsweep = g.loc[g.ece.idxmin()]
        log("  %s" % d)
        log("    균등가중 (= §8)      ECE %.4f  AUROC %.4f" % (B[d]["ece"], B[d]["auroc"]))
        _dd = H[d]["ece"] - B[d]["ece"]
        _pp = abs(_dd) / B[d]["ece"] * 100
        log(
            "    정직선택 tau*        ECE %.4f  AUROC %.4f  → 균등 대비 %+.4f (%s %.1f%%)"
            % (H[d]["ece"], H[d]["auroc"], _dd, "악화" if _dd > 0 else "개선", _pp)
        )
        log(
            "    (참고) sweep 최솟값  ECE %.4f at %s — **검증군을 보고 고른 상한선**"
            % (bestsweep.ece, bestsweep.tag)
        )
        gw = out[(out.dataset == d) & (out.kind == "window")]
        if len(gw):
            bw = gw.loc[gw.ece.idxmin()]
            log(
                "    (참고) 하드 절단 최선 ECE %.4f at %s (유효n %d)"
                % (bw.ece, bw.tag, int(bw.n_eff))
            )
        log("")
    # ---------------- §19 법칙의 예측 검정 ----------------------------
    _fitp = os.path.join(RES, "prevalence_sweep_fit.csv")
    if os.path.exists(_fitp):
        SWF = {r.dataset: r for _, r in pd.read_csv(_fitp).iterrows()}
        log("=" * 104)
        log("§19 법칙의 예측 검정 — 사후 설명이 아니라 사전 예측이다")
        log("=" * 104)
        log("[논리] §19 는 '보정오차는 학습·검증 유병률 격차의 함수' 라고 했고,")
        log("       절대격차 0.10 당 기울기를 쟀다. 최신성 가중은 학습 양성률을")
        log("       바꾸므로 격차도 바뀐다. 그러면 §19 기울기만으로 ECE 변화를")
        log("       **미리 계산할 수 있다.** 관측값과 맞으면 §19 법칙이 독립")
        log("       실험에서 예측력을 보인 것이다.")
        log("")
        log(
            "  %-10s %-20s %8s %8s %10s %10s %9s"
            % ("데이터셋", "설정", "격차", "|격차|변화", "예측 ΔECE", "관측 ΔECE", "차이")
        )
        log("  " + "-" * 86)
        for d in B:
            if d not in SWF:
                continue
            sl = float(SWF[d]["slope_per_0.10"]) / 0.10  # 격차 1.0 당
            g = out[(out.dataset == d) & (out.kind == "weight")]
            b = g[g.tau.isna()].iloc[0]
            for _, r in g[g.tau.notna()].iterrows():
                dgap = abs(float(r.gap)) - abs(float(b.gap))
                pred = sl * dgap
                obs = float(r.ece) - float(b.ece)
                log(
                    "  %-10s %-20s %+8.4f %+8.4f %+10.4f %+10.4f %+9.4f"
                    % (d, r.tag, float(r.gap), dgap, pred, obs, obs - pred)
                )
        log("")
        log("  [주의] §19 는 학습 크기를 고정한 무작위 분할이었고 여기는 실제")
        log("         시간분할이며 유효 표본 수도 변한다. 그래서 이 비교는")
        log("         **방향과 자릿수** 를 보는 것이지 정밀한 예측 검정이 아니다.")
        log("")
    log("저장: %s" % os.path.join(RES, "recency_weighting.csv"))
    log("      %s" % os.path.join(RES, "recency_weighting_summary.txt"))
    log.save(Path(RES) / "recency_weighting_summary.txt")


if __name__ == "__main__":
    main()
