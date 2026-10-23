"""하이퍼파라미터를 조정해도 보정이 돌아오는가 — 54조합 전수.

왜 필요한가 (가장 흔한 반론)
---------------------------
이 논문의 모든 수치는 `_shared.make_model()` 의 고정 설정 하나에서
나왔다 (n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8). 한 번도 조정한 적이 없다. 그래서 심사자가 당연히
묻는다 — "튜닝하지 않은 모델이 과적합해서 확률이 틀어진 것 아닌가."

§9.1 의 32조합은 모델 종류와 분자 표현을 바꾼 것이고 하이퍼파라미터는
각 모델마다 하나로 고정돼 있다. 즉 이 반론에 대한 답이 저장소에 없다.

이 스크립트가 하는 것
--------------------
  격자        max_depth {3,6,9} x n_estimators {200,400,800}
              x learning_rate {0.02,0.05,0.10} x min_child_weight {1,5}
              = 54조합. 논문의 고정 설정(6,400,0.05,1)이 이 격자 안에 있고,
              그 조합의 예측이 make_model() 과 같은지 먼저 검사한다.

  (A) 정직한 선택   학습군(2010년 이전)만으로 고른다. 학습군을 연도순으로
                    나눠 앞 75%로 적합하고 뒤 25%(학습 시대의 최근 구간)에서
                    log loss / ECE / AUROC 가 가장 좋은 조합을 고른다.
                    검증군은 보지 않는다. 그 조합을 학습군 전체로 다시
                    적합해 검증군에서 평가한다. 실무에서 할 수 있는 최선이다.

  (B) 사후 최선     54조합을 검증군에서 직접 평가해 시간분할 ECE 의
                    최소값을 찾는다. 이것은 검증군을 보고 고르는 것이므로
                    달성 불가능한 하한이고, 그렇게 표시한다. 목적은 하나다
                    — "격자 안에 보정을 되돌리는 조합이 존재하는가" 에 답하는
                    것. 존재하지 않으면 (A)의 실패는 선택 방법의 문제가 아니다.

  (C) 대조군        고정 설정과 (A)에서 고른 조합에 대해 §8 과 같은 무작위
                    분할 대조군을 돌려 ECE 배율을 낸다. 튜닝이 배율을
                    줄이는지 본다.

사용법
------
    python src/hparam_sensitivity.py                    # 격자 54 + 대조군 50회
    python src/hparam_sensitivity.py --ctrl-reps 200    # 대조군을 §8 과 같게
    python src/hparam_sensitivity.py --quick            # 격자를 18조합으로 축소

출력
----
    results/hparam_sensitivity.csv          54조합 x 2데이터셋
    results/hparam_sensitivity_ctrl.csv     배율 대조군
    results/hparam_sensitivity_summary.txt
    results/hparam_progress.txt             진행 상황 (실행 중 확인용)
"""

import argparse
import itertools
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

PROG = os.path.join(RES, "hparam_progress.txt")
INNER_FRAC = 0.25  # 학습군의 뒤 25% 를 조합 선택에 쓴다
FIXED = dict(max_depth=6, n_estimators=400, learning_rate=0.05, min_child_weight=1)
TOL = 1e-9  # 고정 설정 재현 허용 오차


def grid(quick=False):
    md = [3, 6, 9]
    ne = [200, 400] if quick else [200, 400, 800]
    lr = [0.05, 0.10] if quick else [0.02, 0.05, 0.10]
    mc = [1, 5]
    out = [
        dict(max_depth=a, n_estimators=b, learning_rate=c, min_child_weight=d)
        for a, b, c, d in itertools.product(md, ne, lr, mc)
    ]
    # 고정 설정은 어떤 경우에도 격자에 들어 있어야 비교가 가능하다
    if FIXED not in out:
        out.append(dict(FIXED))
    return out


def model(cfg, seed=S.SEED):
    """make_model() 과 나머지 인자를 똑같이 두고 격자 인자만 바꾼다."""
    return XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        **cfg
    )


def tag(cfg):
    return "d%d/n%d/lr%.2f/mcw%d" % (
        cfg["max_depth"],
        cfg["n_estimators"],
        cfg["learning_rate"],
        cfg["min_child_weight"],
    )


def score(y, p):
    return {
        "auroc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, S.EPS, 1 - S.EPS))),
        "ece": S.ece(y, p),
        "slope": S.calib_slope(y, p),
        "brier": float(np.mean((np.asarray(y) - p) ** 2)),
    }


def progress(msg):
    with open(PROG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), msg))


def control_ratio(X, y, te_n, reps, cfg, rate_te, rate_tr, matched):
    """§8 과 같은 설계의 무작위 분할 대조군 ECE 평균."""
    E = []
    for i in range(reps):
        r = np.random.default_rng(S.SEED + i)
        idx = r.permutation(len(y))
        ti, ri = idx[:te_n], idx[te_n:]
        if matched:
            ti = S.subsample_to_rate(y, ti, rate_te, r)
            ri = S.subsample_to_rate(y, ri, rate_tr, r)
        if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
            continue
        if len(ti) < 30 or len(ri) < 100:
            continue
        m = model(cfg).fit(X[ri], y[ri])
        E.append(S.ece(y[ti], m.predict_proba(X[ti])[:, 1]))
    if not E:
        return float("nan"), 0
    return float(np.mean(E)), len(E)


def run(name, fn, G, ctrl_reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rate_te, rate_tr = float(y[te].mean()), float(y[tr].mean())

    # 학습군 내부 시간 분할 — 조합 선택용. 검증군은 전혀 쓰지 않는다.
    order = tr[np.argsort(yr[tr], kind="stable")]
    n_in = int(round(len(order) * INNER_FRAC))
    fit_i, sel_i = order[:-n_in], order[-n_in:]

    log("=" * 122)
    log(
        "%s   학습 %d (내부 적합 %d / 내부 선택 %d, 승인 %d~%d) / 검증 %d"
        % (
            name,
            len(tr),
            len(fit_i),
            len(sel_i),
            int(yr[sel_i].min()),
            int(yr[sel_i].max()),
            len(te),
        )
    )
    log("=" * 122)

    rows = []
    for k, cfg in enumerate(G):
        # (A) 내부 선택용 지표 — 학습군 안에서만
        mi = model(cfg).fit(X[fit_i], y[fit_i])
        s_in = score(y[sel_i], mi.predict_proba(X[sel_i])[:, 1])
        # (B) 사후 평가 — 학습군 전체로 적합해 검증군에서
        mf = model(cfg).fit(X[tr], y[tr])
        p_te = mf.predict_proba(X[te])[:, 1]
        s_te = score(y[te], p_te)
        rows.append(
            dict(
                {"dataset": name, "config": tag(cfg)},
                **cfg,
                **{"inner_%s" % a: b for a, b in s_in.items()},
                **{"temporal_%s" % a: b for a, b in s_te.items()},
                is_fixed=(cfg == FIXED)
            )
        )
        progress(
            "%s %d/%d %s inner_ll=%.4f temporal_ece=%.4f"
            % (name, k + 1, len(G), tag(cfg), s_in["logloss"], s_te["ece"])
        )
        if cfg == FIXED:
            # 논문의 고정 설정을 이 격자가 정확히 재현하는지 확인한다
            p_ref = S.make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            d = float(np.max(np.abs(p_ref - p_te)))
            log(
                "  고정 설정 재현 검사 (make_model 대비 최대 예측차) %.2e → %s"
                % (d, "일치" if d < TOL else "**불일치 — 중단**")
            )
            assert d < TOL, "고정 설정 재현 실패: 최대차 %.3e" % d

    R = pd.DataFrame(rows)

    # ---- (A) 정직한 선택 결과 ----
    pick = {}
    pick["log loss"] = R.loc[R.inner_logloss.idxmin()]
    pick["ECE"] = R.loc[R.inner_ece.idxmin()]
    pick["AUROC"] = R.loc[R.inner_auroc.idxmax()]
    fixed_row = R[R.is_fixed].iloc[0]

    log("")
    log("(A) 학습군만 보고 고른 조합 — 검증군 성능은 선택 후에 본 것이다")
    log(
        "%-10s %-22s %12s %12s %10s %10s"
        % ("선택 기준", "고른 조합", "내부 지표", "검증 ECE", "검증 AUROC", "고정 대비")
    )
    log("-" * 122)
    for crit, r in pick.items():
        inner = {"log loss": r.inner_logloss, "ECE": r.inner_ece, "AUROC": r.inner_auroc}[crit]
        log(
            "%-10s %-22s %12.4f %12.4f %10.4f %+10.4f"
            % (
                crit,
                r.config,
                inner,
                r.temporal_ece,
                r.temporal_auroc,
                r.temporal_ece - fixed_row.temporal_ece,
            )
        )
    log(
        "%-10s %-22s %12s %12.4f %10.4f %10s"
        % ("(고정)", fixed_row.config, "-", fixed_row.temporal_ece, fixed_row.temporal_auroc, "-")
    )

    # ---- (B) 사후 최선 ----
    best = R.loc[R.temporal_ece.idxmin()]
    rank = int((R.temporal_ece < fixed_row.temporal_ece).sum()) + 1
    log("")
    log("(B) 검증군을 보고 고른 사후 최선 — 달성 불가능한 하한")
    log(
        "    최소 검증 ECE %.4f (%s), 최대 %.4f, 중앙값 %.4f"
        % (best.temporal_ece, best.config, R.temporal_ece.max(), R.temporal_ece.median())
    )
    log("    고정 설정은 %d조합 중 %d위 (검증 ECE 작은 순)" % (len(R), rank))
    log("    검증 AUROC 범위 %.4f~%.4f" % (R.temporal_auroc.min(), R.temporal_auroc.max()))

    # ---- (C) 대조군 배율 ----
    ctrl = []
    for lab, cfg_row in (("고정 설정", fixed_row), ("내부 log loss 선택", pick["log loss"])):
        cfg = {k: (int(cfg_row[k]) if k != "learning_rate" else float(cfg_row[k])) for k in FIXED}
        c1, n1 = control_ratio(X, y, len(te), ctrl_reps, cfg, rate_te, rate_tr, matched=False)
        c3, n3 = control_ratio(X, y, len(te), ctrl_reps, cfg, rate_te, rate_tr, matched=True)
        ctrl.append(
            {
                "dataset": name,
                "arm": lab,
                "config": tag(cfg),
                "temporal_ece": float(cfg_row.temporal_ece),
                "ctrl_R1_ece": c1,
                "ctrl_R1_n": n1,
                "ratio_R1": float(cfg_row.temporal_ece) / c1 if c1 > 0 else float("nan"),
                "ctrl_R3_ece": c3,
                "ctrl_R3_n": n3,
                "ratio_R3": float(cfg_row.temporal_ece) / c3 if c3 > 0 else float("nan"),
            }
        )
        progress("%s 대조군 %s 완료 (R1 %d회, R3 %d회)" % (name, lab, n1, n3))
    C = pd.DataFrame(ctrl)
    log("")
    log("(C) 같은 조합으로 돌린 무작위 분할 대조군 %d회" % ctrl_reps)
    log(
        "%-20s %-22s %10s %10s %8s %10s %8s"
        % ("조합", "설정", "검증 ECE", "R1 대조", "R1 배율", "R3 대조", "R3 배율")
    )
    log("-" * 122)
    for _, r in C.iterrows():
        log(
            "%-20s %-22s %10.4f %10.4f %8.2f %10.4f %8.2f"
            % (
                r.arm,
                r.config,
                r.temporal_ece,
                r.ctrl_R1_ece,
                r.ratio_R1,
                r.ctrl_R3_ece,
                r.ratio_R3,
            )
        )
    log("")

    # ---- 판정 ----
    c1_fixed = float(C[C.arm == "고정 설정"].ctrl_R1_ece.iloc[0])
    n_below = int((R.temporal_ece <= c1_fixed).sum())
    log(
        "  격자 %d조합 중 검증 ECE 가 무작위 분할 대조 수준(%.4f) 이하인 것: %d개"
        % (len(R), c1_fixed, n_below)
    )
    if n_below == 0:
        log("  → **격자 안에 보정 붕괴를 되돌리는 조합이 없다.** 사후에 검증군을")
        log("     보고 골라도 마찬가지이므로, 이것은 선택 방법의 문제가 아니다.")
    else:
        log("  → **되돌리는 조합이 있다. 본문에 그 조합과 조건을 명시해야 한다.**")
    sel_ok = all(pick[c].temporal_ece >= c1_fixed for c in pick)
    log(
        "  (A)의 세 선택 기준이 고른 조합도 전부 대조 수준 위인가: %s"
        % ("예" if sel_ok else "아니오")
    )
    log("")
    return rows, ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-reps", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    open(PROG, "w", encoding="utf-8").write(
        "hparam_sensitivity 진행 기록 — %s\n" % time.strftime("%Y-%m-%d %H:%M")
    )
    G = grid(a.quick)
    log("하이퍼파라미터를 조정해도 보정이 돌아오는가 — %d조합 전수" % len(G))
    log(
        "고정 설정 %s 를 기준으로 비교한다. 선택은 학습군 안에서만 한다"
        "(내부 뒤 %.0f%%)." % (tag(FIXED), INNER_FRAC * 100)
    )
    log("대조군 %d회, 시드 %d" % (a.ctrl_reps, S.SEED))
    log("")

    rows, ctrl = [], []
    for name, fn in S.DATASETS:
        x, c = run(name, fn, G, a.ctrl_reps)
        rows += x
        ctrl += c

    pd.DataFrame(rows).to_csv(
        os.path.join(RES, "hparam_sensitivity.csv"), index=False, encoding="utf-8"
    )
    pd.DataFrame(ctrl).to_csv(
        os.path.join(RES, "hparam_sensitivity_ctrl.csv"), index=False, encoding="utf-8"
    )
    log("=" * 122)
    log("저장: %s" % os.path.join(RES, "hparam_sensitivity.csv"))
    log("      %s" % os.path.join(RES, "hparam_sensitivity_ctrl.csv"))
    log.save(Path(RES) / "hparam_sensitivity_summary.txt")


if __name__ == "__main__":
    main()
