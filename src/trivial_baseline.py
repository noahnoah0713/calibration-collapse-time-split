# -*- coding: utf-8 -*-
"""자명한 기준선 — 학습 모델은 '상수 예측' 을 이기는가.

왜 이 절이 필요한가
------------------
이 논문은 AUROC 가 유지되는 동안 보정이 무너진다고 말한다. 그 주장을
가장 강하게(그리고 가장 위험하게) 검증하는 방법은 하나다 —
**화학 구조를 전혀 쓰지 않는 예측기와 비교하는 것.**

상수 예측기는 모든 분자에 같은 확률을 낸다. 판별력이 정의상 없다
(AUROC 는 의미가 없다). 그런데 보정 지표와 적정 채점규칙(Brier)에서는
학습 모델을 이길 수 있다. 이기면 다음 두 문장이 동시에 참이 된다 —

    모델의 **순위**는 우연보다 낫다 (§5 순열검정이 유의)
    모델의 **확률**은 구조를 안 보는 상수보다 못하다

이 대비가 '판별과 보정은 다른 것을 잰다' 의 가장 날카로운 형태다.

**이 절을 쓰지 않으면 심사자가 먼저 계산한다.** 그 경우 "당신 모델은
상수보다 못하다" 가 심사의견으로 등장하고 논문은 방어 위치에 선다.
먼저 쓰면 같은 사실이 논지의 핵심 근거가 된다.

비교 대상
--------
  M   학습 모델 (XGBoost)
  C0  상수 = 학습군 양성률        ← 실무에서 아무 모델 없이 쓰는 값
  C1  상수 = 0.5                ← 가장 소박한 값
  C2  상수 = 검증군 양성률         ← oracle. ECE 는 정의상 0 이다
  무작위 분할에서도 같은 비교를 붙인다 — **대비가 시간분할에서만 나타나야**
  "시간 이동이 원인" 이라는 주장이 선다.

읽는 법
  적정 채점규칙(Brier)이 결정적이다. ECE 만으로는 "상수는 해상도가 0 이니
  당연하다" 는 반론이 가능하지만, Brier 는 보정과 해상도를 함께 센다.
  Brier 에서도 지면 그 반론이 막힌다.
  Brier 기술점수(BSS) = 1 − Brier(모델)/Brier(C0). 음수면 **학습이 아무
  모델도 없는 것보다 해롭다**는 뜻이다.

주의
  상수 예측기는 **쓸모가 없다** — 해상도가 0 이라 어떤 분자를 고를지
  알려주지 못한다. 이 절의 결론은 "상수를 쓰라" 가 아니라 "시간분할 아래
  이 모델의 **확률값**은 신뢰할 수 없다" 이다. 순위(AUROC)와 확률을
  분리해 쓰라는 처방으로 이어진다.

사용법
------
    python src/trivial_baseline.py

출력
----
    results/trivial_baseline.csv
    results/trivial_baseline_summary.txt
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

N_BOOT = 2000


def brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def full(y, p):
    d = S.decompose(y, p)
    out = {
        "ece": S.ece(y, p),
        "brier": brier(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
        "unc": d["uncertainty"],
    }
    try:
        out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else float("nan")
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["slope"] = S.calib_slope(y, p) if len(np.unique(p)) > 1 else float("nan")
    except Exception:
        out["slope"] = float("nan")
    return out


def boot_diff(y, p_m, p_c, fn, n_boot=N_BOOT, seed=S.SEED):
    """모델 − 상수 의 차에 대한 95% 구간. 같은 재표집을 두 예측에 쓴다."""
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    v = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(fn(y[i], np.asarray(p_m)[i]) - fn(y[i], np.asarray(p_c)[i]))
    if not v:
        return float("nan"), float("nan")
    a = np.sort(np.array(v))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def run(name, fn):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]

    r = np.random.default_rng(S.SEED)
    idx = r.permutation(len(y))
    rte, rtr = idx[: len(te)], idx[len(te) :]

    rows = []
    for split, a, b in (("시간분할", tr, te), ("무작위분할", rtr, rte)):
        m = S.make_model().fit(X[a], y[a])
        p_m = m.predict_proba(X[b])[:, 1]
        yb = y[b]
        pi_tr = float(y[a].mean())
        pi_te = float(yb.mean())

        cands = [
            ("M. 학습 모델 (XGBoost)", p_m),
            ("C0. 상수 = 학습군 양성률", np.full(len(yb), pi_tr)),
            ("C1. 상수 = 0.5", np.full(len(yb), 0.5)),
            ("C2. 상수 = 검증군 양성률 (oracle)", np.full(len(yb), pi_te)),
        ]

        log("=" * 112)
        log(
            "%s / %s   학습 %d개(양성률 %.4f) / 검증 %d개(양성률 %.4f)"
            % (name, split, len(a), pi_tr, len(b), pi_te)
        )
        log("=" * 112)
        log(
            "%-32s %8s %8s %8s %8s %8s %8s %-22s"
            % ("예측기", "AUROC", "ECE", "Brier", "신뢰도", "해상도", "BSS", "ΔBrier 대 M 95%CI")
        )
        log("-" * 118)

        b0 = brier(yb, np.full(len(yb), pi_tr))
        base_m = None
        for tag, pp in cands:
            mm = full(yb, pp)
            bss = 1.0 - mm["brier"] / b0 if b0 else float("nan")
            if tag.startswith("M."):
                base_m = pp
                ci = "—"
            else:
                lo, hi = boot_diff(yb, base_m, pp, brier)
                ci = "[%+.4f, %+.4f]" % (lo, hi)
            log(
                "%-32s %8s %8.4f %8.4f %8.4f %8.4f %8.4f %s"
                % (
                    tag,
                    "—" if mm["auroc"] != mm["auroc"] else "%.4f" % mm["auroc"],
                    mm["ece"],
                    mm["brier"],
                    mm["rel"],
                    mm["res"],
                    bss,
                    ci,
                )
            )
            rows.append(
                {
                    "dataset": name,
                    "split": split,
                    "predictor": tag,
                    "n_train": len(a),
                    "n_test": len(b),
                    "pi_train": pi_tr,
                    "pi_test": pi_te,
                    "bss_vs_C0": bss,
                    **mm,
                }
            )
        log("")
    return rows


def main():
    log("자명한 기준선 — 학습 모델은 상수 예측을 이기는가")
    log("시드 %d, 부트스트랩 %d회. 적합은 분할당 1회(결정적)." % (S.SEED, N_BOOT))
    log("[주의] 상수 예측기는 해상도가 0 이라 **쓸모가 없다.** 이 절의 결론은")
    log("       '상수를 쓰라' 가 아니라 '시간분할 아래 이 모델의 확률값은")
    log("       신뢰할 수 없다' 이다.")
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "trivial_baseline.csv"), index=False, encoding="utf-8")

    log("=" * 112)
    log("판정")
    log("=" * 112)
    log("[읽는 법] Brier 가 결정적이다. ECE 만으로는 '상수는 해상도가 0 이니")
    log("          당연하다' 는 반론이 가능하지만 Brier 는 보정과 해상도를")
    log("          함께 센다. **대비가 시간분할에서만 나타나야** 시간 이동이")
    log("          원인이라는 주장이 선다 — 무작위분할 행을 반드시 함께 읽을 것.")
    log("")
    for d in out.dataset.unique():
        log("  %s" % d)
        for sp in out.split.unique():
            g = out[(out.dataset == d) & (out.split == sp)].set_index("predictor")
            m = g[g.index.str.startswith("M.")].iloc[0]
            lose = [
                i for i in g.index if not i.startswith("M.") and float(g.loc[i].brier) < m.brier
            ]
            log(
                "    %-10s 모델 Brier %.4f / ECE %.4f / AUROC %.4f / BSS %+.4f"
                % (sp, m.brier, m.ece, m.auroc, m.bss_vs_C0)
            )
            if lose:
                log(
                    "      → **모델보다 Brier 가 낮은(더 좋은) 상수 %d종**: %s"
                    % (len(lose), ", ".join(i.split(". ")[0] for i in lose))
                )
            else:
                log("      → 모든 상수보다 모델의 Brier 가 낮다 (정상)")
        log("")
    log("저장: %s" % os.path.join(RES, "trivial_baseline.csv"))
    log("      %s" % os.path.join(RES, "trivial_baseline_summary.txt"))
    log.save(Path(RES) / "trivial_baseline_summary.txt")


if __name__ == "__main__":
    main()
