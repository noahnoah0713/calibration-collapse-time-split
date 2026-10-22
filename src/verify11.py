# -*- coding: utf-8 -*-
"""§9.1(robustness.py) 과 §9.5(grid_calibration.py) 의 격자가 실제로 같은
격자인지 결과 파일로 확인한다.

배경 (2026-09-30)
----------------
두 절 모두 "모델 4종 x 표현 4종 = 16조합, 데이터셋 2개 = 32" 이라고 적었으나,
두 스크립트가 같은 이름의 모델을 각자 정의해 하이퍼파라미터가 달랐다
(RandomForest min_samples_leaf, LightGBM num_leaves 대 max_depth,
LogisticRegression max_iter). 그래서 같은 (데이터셋, 표현, 모델) 칸의
auroc_temporal / ece_temporal 이 두 파일에서 어긋났다. 정의는
_shared.grid_models() 한 곳으로 합쳤다.

이 스크립트는 그 수정이 실제로 효과가 있었는지, 문서가 아니라 결과 파일로
증명한다. 두 CSV 의 공통 열(dataset, rep, model, auroc_temporal,
ece_temporal)을 키로 맞춰 값이 같은지 본다. 두 스크립트는 시간분할 예측을
같은 방식으로 만들므로, 같은 정의라면 이 값들은 비트 단위로 같아야 한다.

    python src/verify11.py

종료코드 0 = 두 격자가 일치, 1 = 어긋남.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KEY = ["dataset", "rep", "model"]
VAL = ["auroc_temporal", "ece_temporal"]
TOL = 1e-12

fails = []


def main():
    a = RES / "robustness_models.csv"
    b = RES / "grid_calibration.csv"
    for p in (a, b):
        if not p.exists():
            sys.exit("%s 가 없다. 먼저 해당 스크립트를 돌려야 한다." % p.name)
    ra = pd.read_csv(a)[KEY + VAL]
    rb = pd.read_csv(b)[KEY + VAL]

    print("격자 일치 검증 (§9.1 vs §9.5)")
    print("=" * 70)
    print("robustness_models.csv  %d행" % len(ra))
    print("grid_calibration.csv   %d행" % len(rb))

    # 1) 칸 집합이 같은가
    ka = set(map(tuple, ra[KEY].values))
    kb = set(map(tuple, rb[KEY].values))
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    print(
        "공통 칸 %d개 / robustness 에만 %d개 / grid 에만 %d개"
        % (len(ka & kb), len(only_a), len(only_b))
    )
    for t in only_a:
        fails.append("robustness 에만 있는 칸: %s" % (t,))
    for t in only_b:
        fails.append("grid 에만 있는 칸: %s" % (t,))

    # 2) 중복 칸이 없는가 (있으면 병합이 뜻을 잃는다)
    for nm, r in (("robustness_models.csv", ra), ("grid_calibration.csv", rb)):
        d = r[r.duplicated(KEY, keep=False)]
        if len(d):
            fails.append("%s 에 같은 칸이 %d행 중복" % (nm, len(d)))

    # 3) 값이 같은가
    m = ra.merge(rb, on=KEY, suffixes=("_rb", "_gc"))
    print()
    print(
        "%-10s %-12s %-14s %12s %12s %10s"
        % ("데이터셋", "표현", "모델", "AUROC차", "ECE차", "판정")
    )
    print("-" * 74)
    worst = {v: 0.0 for v in VAL}
    for _, r in m.sort_values(KEY).iterrows():
        d = {v: abs(r[v + "_rb"] - r[v + "_gc"]) for v in VAL}
        for v in VAL:
            worst[v] = max(worst[v], d[v])
        ok = all(d[v] <= TOL for v in VAL)
        print(
            "%-10s %-12s %-14s %12.3e %12.3e %10s"
            % (
                r["dataset"],
                r["rep"],
                r["model"],
                d["auroc_temporal"],
                d["ece_temporal"],
                "일치" if ok else "*** 불일치 ***",
            )
        )
        if not ok:
            fails.append(
                "%s / %s / %s 가 두 파일에서 다르다 "
                "(AUROC차 %.3e, ECE차 %.3e)"
                % (r["dataset"], r["rep"], r["model"], d["auroc_temporal"], d["ece_temporal"])
            )
    print()
    print("최대차: AUROC %.3e, ECE %.3e" % (worst["auroc_temporal"], worst["ece_temporal"]))
    print()
    print("실패 %d건" % len(fails))
    for f in fails:
        print("  X", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
