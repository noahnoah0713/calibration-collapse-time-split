# -*- coding: utf-8 -*-
"""§10 과 §10.5 의 Platt 이 왜 3.6e-05 어긋나는지 **이분해서** 찾는다.

경위
----
2026-10-20. §10.5(extra_calibrators.py)가 §10 의 B(Platt) ECE 를
0.167939550900045 로 내는데 §10(recalibration.py)은 0.167903612057368 이다.
차 3.6e-05 이고, 보고 자리수 네 자리에는 영향이 없다(둘 다 0.1679).

원인을 두 번 추측했고 **두 번 다 틀렸다.**

  추측 1  logit 의 클리핑 방식 차이 → 식을 맞췄는데 차가 그대로였다.
  추측 2  lbfgs / 다중스레드의 실행 간 비결정성 → check_determinism 의
          [2]·[3]·[5] 가 한 프로세스 안에서도, 프로세스를 달리해도
          (스레드 고정 없이) 차 0 임을 보였다. 반증됐다.

확인된 것들 — 적재·정렬·분할이 같고(인덱스 집합 비교), featurize 의 소스가
같고, make_model 의 파라미터가 같고, ece 일곱 정의가 비트 단위로 같고
(check_determinism [1c]), 사다리 전체가 프로세스를 건너 재현된다.

그래서 추측을 멈추고 **두 경로를 한 프로세스에서 나란히 돌려 처음 갈라지는
지점**을 찾는다. 단계마다 비트 단위로 비교하고, 처음 어긋난 곳에서 멈춰
그 자리를 출력한다.

사용법
------
    python src/diff_ladder.py

출력은 화면과 results/diff_ladder.txt 다. 결과 CSV 를 만들지 않는다 —
진단용이고 보고서가 읽지 않는다.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S
import recalibration as RC

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
CALIB_FRAC = 0.25

lines = []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def cmp(tag, a, b):
    """비트 단위 비교. 같으면 True."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        log("  %-34s *** 모양이 다르다 %s vs %s ***" % (tag, a.shape, b.shape))
        return False
    if a.dtype != b.dtype:
        log("  %-34s dtype 다름 %s vs %s" % (tag, a.dtype, b.dtype))
    same = np.array_equal(a, b)
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub" and a.size:
        mx = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        log("  %-34s %s  최대차 %.3e" % (tag, "동일" if same else "*** 다르다 ***", mx))
    else:
        log("  %-34s %s" % (tag, "동일" if same else "*** 다르다 ***"))
    return same


def path_a(fn):
    """§10 (recalibration.py) 의 경로."""
    import pandas as pd

    df = pd.read_csv(os.path.join(PROC, fn))
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.sort_values("approval_year").reset_index(drop=True)
    X = RC.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr = np.where(yr < RC.CUTOFF)[0]
    te = np.where(yr >= RC.CUTOFF)[0]
    n_cal = int(len(tr) * RC.CALIB_FRAC)
    return dict(df=df, X=X, y=y, yr=yr, tr=tr, te=te, fit=tr[:-n_cal], cal=tr[-n_cal:])


def path_b(fn):
    """§10.5 (extra_calibrators.py) 의 경로."""
    df0 = S.load(os.path.join(PROC, fn))
    d = df0.sort_values("approval_year").reset_index(drop=True)
    X = S.featurize(d["smiles"].tolist())
    y = d["y"].values
    yr = d["approval_year"].values
    tr = np.where(yr < S.CUTOFF)[0]
    te = np.where(yr >= S.CUTOFF)[0]
    n_cal = int(len(tr) * CALIB_FRAC)
    return dict(df=d, X=X, y=y, yr=yr, tr=tr, te=te, fit=tr[:-n_cal], cal=tr[-n_cal:])


def logit_b(p):
    p = np.asarray(p, dtype=float)
    return np.log(np.clip(p, S.EPS, 1 - S.EPS) / np.clip(1 - p, S.EPS, 1 - S.EPS))


def run(name, fn):
    from sklearn.linear_model import LogisticRegression

    log("=" * 96)
    log("%s" % name)
    log("=" * 96)
    A, B = path_a(fn), path_b(fn)

    ok = True
    ok &= cmp(
        "1. SMILES 순서", A["df"]["smiles"].values.astype(str), B["df"]["smiles"].values.astype(str)
    )
    ok &= cmp("2. 라벨 y", A["y"], B["y"])
    ok &= cmp("3. 승인연도", A["yr"], B["yr"])
    ok &= cmp("4. 특성 행렬 X", A["X"], B["X"])
    ok &= cmp("5. 적합셋 인덱스", A["fit"], B["fit"])
    ok &= cmp("6. 보정셋 인덱스", A["cal"], B["cal"])
    ok &= cmp("7. 검증군 인덱스", A["te"], B["te"])
    if not ok:
        log("  → 입력 단계에서 이미 갈라진다. 여기서 멈춘다.")
        log("")
        return

    # 같은 X 를 쓰되 모델 생성자만 두 경로의 것을 쓴다
    pa = {}
    pb = {}
    ma = RC.make_model().fit(A["X"][A["fit"]], A["y"][A["fit"]])
    mb = S.make_model().fit(B["X"][B["fit"]], B["y"][B["fit"]])
    pa["cal"] = ma.predict_proba(A["X"][A["cal"]])[:, 1]
    pb["cal"] = mb.predict_proba(B["X"][B["cal"]])[:, 1]
    pa["te"] = ma.predict_proba(A["X"][A["te"]])[:, 1]
    pb["te"] = mb.predict_proba(B["X"][B["te"]])[:, 1]
    s1 = cmp("8. 적합 모델의 보정셋 예측", pa["cal"], pb["cal"])
    s2 = cmp("9. 적합 모델의 검증군 예측", pa["te"], pb["te"])

    log("  모델 파라미터 동일: %s" % (RC.make_model().get_params() == S.make_model().get_params()))
    if not (s1 and s2):
        log(
            "  → **모델 예측 단계에서 갈라진다.** 입력과 파라미터가 같은데 "
            "예측이 다르면 남는 것은 XGBoost 내부 상태다."
        )
        _d = RC.make_model().get_params()
        _e = S.make_model().get_params()
        for k in sorted(set(_d) | set(_e)):
            if _d.get(k) != _e.get(k):
                log("     파라미터 차이 %s: %r vs %r" % (k, _d.get(k), _e.get(k)))
        log("")
        return

    la = RC.to_logit(pa["cal"])
    lb = logit_b(pb["cal"])
    cmp("10. logit(보정셋 예측)", la, lb)
    la2, lb2 = RC.to_logit(pa["te"]), logit_b(pb["te"])
    cmp("11. logit(검증군 예측)", la2, lb2)

    ya = A["y"][A["cal"]]
    yb = B["y"][B["cal"]]
    lra = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lra.fit(la.reshape(-1, 1), ya)
    lrb = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lrb.fit(lb.reshape(-1, 1), np.asarray(yb))
    log(
        "  12. Platt 계수   A %.17g / %.17g   B %.17g / %.17g"
        % (lra.coef_[0][0], lra.intercept_[0], lrb.coef_[0][0], lrb.intercept_[0])
    )
    cmp(
        "13. Platt 계수 비교",
        [lra.coef_[0][0], lra.intercept_[0]],
        [lrb.coef_[0][0], lrb.intercept_[0]],
    )

    qa = lra.predict_proba(la2.reshape(-1, 1))[:, 1]
    qb = lrb.predict_proba(lb2.reshape(-1, 1))[:, 1]
    cmp("14. Platt 보정 후 확률", qa, qb)
    yte_a, yte_b = A["y"][A["te"]], B["y"][B["te"]]
    ea, eb = RC.ece(yte_a, qa), S.ece(yte_b, qb)
    log("  15. ECE          A %.17g   B %.17g   차 %.3e" % (ea, eb, abs(ea - eb)))
    log("")
    log(
        "  [해석] 8·9 가 동일하고 13 이 다르면 lbfgs 다. 8·9 가 다르면 "
        "모델이다. 전부 같고 15 만 다르면 ece 다."
    )
    log("")


def main():
    log("§10 과 §10.5 의 Platt 불일치를 단계별로 이분한다")
    log("두 경로를 **한 프로세스에서** 나란히 돌린다")
    log("")
    for name, fn in S.DATASETS:
        run(name, fn)
    import io

    io.open(os.path.join(RES, "diff_ladder.txt"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n"
    )
    log("저장: results/diff_ladder.txt")


if __name__ == "__main__":
    main()
