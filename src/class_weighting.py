# -*- coding: utf-8 -*-
"""학습 시 클래스 가중 — 사후에 고치는 것과 애초에 막는 것.

왜 필요한가
----------
§19 는 보정오차가 학습·검증 유병률 격차의 함수 임을 보였다. 그러면
가장 자연스러운 처방은 학습 단계에서 유효 유병률을 검증군 쪽에 맞추는
것 이다. 지금까지 시험한 것은 둘뿐이다 —

  §10 D  사후 사전확률 보정   : 예측을 다 내고 나서 오즈를 곱해 고친다
  §20    최신성 가중          : 시간축 으로 가중 (실패)

빠진 세 번째가 라벨축 가중 이다. XGBoost 의 scale_pos_weight 로
양성·음성의 유효 비율을 바꿔 학습한다. 같은 정보(검증군 유병률)를 쓰되
사후가 아니라 사전에 적용하는 것이므로, "이미 난 손해를 되돌릴 수
있나" 대 "애초에 안 나게 할 수 있나" 를 가른다.

설계
----
  W0  가중 없음 (scale_pos_weight = 1)            ← §8 설정
  W1  검증군 유병률에 맞춤 (oracle)                ← §10 D 와 같은 정보량
  W2~ 잘못 가정했을 때 (검증군 유병률을 ±0.1, ±0.2 로 오인)
      — 실무에서는 유병률을 모르므로 오지정 민감도 가 본질이다
  P   참고: §10 D 사후 사전확률 보정 (같은 oracle 정보, 사후 적용)

가중 계산: 목표 유병률 pi 를 만들려면
    scale_pos_weight = (pi / (1 − pi)) / (n_pos / n_neg)
이면 양성의 유효 가중합 비율이 pi 가 된다.

이 절의 한계를 먼저 적는다. W1 은 검증군 유병률을 안다고 가정한다.
§10.2 는 그 값을 라벨 없이 추정하려는 시도가 실패함을 보였다. 따라서 W1 은
배포 가능한 처방이 아니라 상한선 이다. 실제로 쓸 수 있는지는 W2~ 의
오지정 민감도가 결정한다 — 유병률을 0.1 틀리게 잡아도 무보정보다 나으면
쓸 만하고, 더 나빠지면 못 쓴다.

사용법
------
    python src/class_weighting.py

출력
----
    results/class_weighting.csv
    results/class_weighting_summary.txt
"""

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

OFFSETS = [-0.20, -0.10, 0.0, +0.10, +0.20]  # 검증군 유병률 오지정 폭
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


def spw(y_tr, pi_target):
    """목표 유병률 pi 를 만드는 scale_pos_weight."""
    npos = int((y_tr == 1).sum())
    nneg = int((y_tr == 0).sum())
    if npos == 0 or nneg == 0 or pi_target <= 0 or pi_target >= 1:
        return 1.0
    return (pi_target / (1.0 - pi_target)) / (npos / float(nneg))


def fit_w(X, y, tr, te, w_pos):
    """양성에만 가중을 준다 (scale_pos_weight 와 동등)."""
    sw = np.where(y[tr] == 1, float(w_pos), 1.0)
    m = S.make_model().fit(X[tr], y[tr], sample_weight=sw)
    return m.predict_proba(X[te])[:, 1]


def prior_shift(p, pi_src, pi_tgt):
    o = np.clip(p, S.EPS, 1 - S.EPS) / np.clip(1 - p, S.EPS, 1 - S.EPS)
    k = (pi_tgt / (1 - pi_tgt)) / (pi_src / (1 - pi_src))
    return (o * k) / (1.0 + o * k)


def boot_ece(y, p, n_boot=N_BOOT, seed=S.SEED):
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p[i]))
    a = np.sort(np.array(v))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def run(name, fn):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    pi_tr = float(y[tr].mean())
    pi_te = float(y[te].mean())

    log("=" * 112)
    log(
        "%s   학습 %d개(양성률 %.4f) / 검증 %d개(양성률 %.4f, 격차 %+.4f)"
        % (name, len(tr), pi_tr, len(te), pi_te, pi_te - pi_tr)
    )
    log("=" * 112)
    log(
        "%-34s %9s %9s %8s %8s %8s %8s %-20s"
        % ("설정", "목표유병률", "가중", "AUROC", "ECE", "기울기", "해상도", "ECE 95%CI")
    )
    log("-" * 112)

    rows = []

    def rec(tag, pi_t, w, p):
        m = metrics(y[te], p)
        lo, hi = boot_ece(y[te], p)
        log(
            "%-34s %9s %9s %8.4f %8.4f %8.4f %8.4f [%.4f, %.4f]"
            % (
                tag,
                "—" if pi_t is None else "%.4f" % pi_t,
                "—" if w is None else "%.3f" % w,
                m["auroc"],
                m["ece"],
                m["slope"],
                m["res"],
                lo,
                hi,
            )
        )
        rows.append(
            {
                "dataset": name,
                "arm": tag,
                "pi_target": np.nan if pi_t is None else pi_t,
                "scale_pos_weight": np.nan if w is None else w,
                "pi_train": pi_tr,
                "pi_test": pi_te,
                "n_train": len(tr),
                "n_test": len(te),
                "ece_lo": lo,
                "ece_hi": hi,
                **m,
            }
        )
        return m

    # W0 — 가중 없음 (= §8)
    base = rec("W0. 가중 없음 (= §8)", None, 1.0, fit_w(X, y, tr, te, 1.0))

    # W1~ — 목표 유병률을 pi_te + offset 으로 두고 가중 학습
    for off in OFFSETS:
        pi_t = float(np.clip(pi_te + off, 0.02, 0.98))
        w = spw(y[tr], pi_t)
        tag = (
            "W1. 검증군 유병률 맞춤 (oracle)"
            if off == 0.0
            else "W%s. 유병률 %+.2f 오지정" % ("2" if off < 0 else "3", off)
        )
        rec(tag, pi_t, w, fit_w(X, y, tr, te, w))

    # P — 사후 사전확률 보정 (같은 oracle 정보, 사후 적용)
    p0 = fit_w(X, y, tr, te, 1.0)
    rec("P. 사후 사전확률 보정 (§10 D)", pi_te, None, prior_shift(p0, pi_tr, pi_te))
    log("")
    return rows, base


def main():
    log("학습 시 클래스 가중 — 사후 보정 대 사전 예방")
    log("시드 %d, 부트스트랩 %d회. 적합은 설정당 1회(결정적)." % (S.SEED, N_BOOT))
    log("")
    rows, B = [], {}
    for name, fn in S.DATASETS:
        rr, b = run(name, fn)
        rows += rr
        B[name] = b
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "class_weighting.csv"), index=False, encoding="utf-8")

    log("=" * 112)
    log("판정")
    log("=" * 112)
    log("[읽는 법] W1 은 검증군 유병률을 그대로 목표로 둔 설정이다. **상한선이")
    log("          아니다** — 아래 표에서 보듯 목표를 검증군 유병률보다 더 낮게")
    log("          잡으면 더 좋아지는 경우가 있다. 즉 손상은 순수한 사전확률")
    log("          이동이 아니며, 가중은 절편만이 아니라 적합 자체를 바꾼다.")
    log("          실제 쓸 수 있는지는 **오지정 행(W2·W3)** 이 정한다. 어느")
    log("          쪽으로 틀리는지가 중요하다 — 과소·과대가 대칭이 아니다.")
    log("")
    for d in out.dataset.unique():
        g = out[out.dataset == d].set_index("arm")
        b = B[d]
        log("  %s   무가중 ECE %.4f / AUROC %.4f" % (d, b["ece"], b["auroc"]))
        for arm in g.index:
            if arm.startswith("W0"):
                continue
            r = g.loc[arm]
            dd = float(r.ece) - b["ece"]
            log(
                "    %-34s ECE %.4f (%+.4f, %s)  AUROC %.4f (%+.4f)"
                % (
                    arm,
                    r.ece,
                    dd,
                    "개선" if dd < 0 else "악화",
                    r.auroc,
                    float(r.auroc) - b["auroc"],
                )
            )
        _mis = g[g.index.str.contains("오지정")]
        _worse = int((_mis.ece > b["ece"]).sum())
        _under = _mis[_mis.index.str.contains("-0")]
        _over = _mis[_mis.index.str.contains("\\+0")]
        log(
            "    → 오지정 %d건 중 %d건이 무가중보다 나쁘다 "
            "(과소지정 %d건 중 %d건 / 과대지정 %d건 중 %d건)."
            % (
                len(_mis),
                _worse,
                len(_under),
                int((_under.ece > b["ece"]).sum()),
                len(_over),
                int((_over.ece > b["ece"]).sum()),
            )
        )
        _w = g[g.index.str.startswith(("W1", "W2", "W3"))]
        _best = _w.ece.idxmin()
        log(
            "    → 가중 중 ECE 최소는 **%s** (%.4f). oracle(W1)이 최소인가: %s"
            % (
                _best,
                float(_w.loc[_best].ece),
                (
                    "예"
                    if str(_best).startswith("W1")
                    else "**아니오** — 목표를 검증군 유병률로 두는 것이 최적이 아니다"
                ),
            )
        )
        if "P. 사후 사전확률 보정 (§10 D)" in g.index:
            _po = float(g.loc["P. 사후 사전확률 보정 (§10 D)"].ece)
            _w1 = float(g.loc["W1. 검증군 유병률 맞춤 (oracle)"].ece)
            log(
                "    → 같은 정보로 사후(%.4f) 대 사전(%.4f): **%s** 가 낫다"
                % (_po, _w1, "사후 보정" if _po < _w1 else "학습 시 가중")
            )
        log("")
    log("저장: %s" % os.path.join(RES, "class_weighting.csv"))
    log("      %s" % os.path.join(RES, "class_weighting_summary.txt"))
    log.save(Path(RES) / "class_weighting_summary.txt")


if __name__ == "__main__":
    main()
