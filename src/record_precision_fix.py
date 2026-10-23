"""2026-10-20 정밀도 수정의 전·후 값을 기록한다.

왜 이 파일이 있는가
-----------------
단정밀도 로짓 문제를 고치면(§10.5 참조) 수정 전 값은 어느 결과 파일에도
남지 않는다. 그런데 보고서는 "기울기가 0.8969 와 0.8962 로 갈려 있었다" 를
적어야 한다 — 그것이 수정의 근거다. 숫자를 산문에 손으로 박으면 verify9 가
"근거 불명" 으로 잡고, 잡는 게 맞다. 그래서 전·후 값을 데이터로 남겨
보고서가 읽게 한다.

비교 대상은 `_backup/precision_20261020/` 의 수정 전 CSV 와 현재
`results/` 의 수정 후 CSV 다. 백업을 지우면 이 스크립트는 돌지 않으므로,
백업은 재현 자료의 일부다.

사용법
------
    python src/record_precision_fix.py

출력
----
    results/precision_fix_20261020.csv
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BK = os.path.join(ROOT, "_backup", "precision_20261020")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "precision_fix_20261020.csv")

if not os.path.isdir(BK):
    # 2026-10-22: 백업이 없으면 실패가 아니라 건너뛴다. rerun_all.sh 의
    # run() 은 0 이 아닌 종료코드를 받으면 전체를 중단시키므로, 저장소를
    # 받은 사람이 이 폴더 없이 돌리면 파이프라인이 통째로 멈춘다.
    # results/precision_fix_20261020.csv 는 저장소에 포함돼 있고 보고서는
    # 그 사본을 읽으므로, 건너뛰어도 문서 생성에는 지장이 없다.
    print("건너뜀 — %s 가 없다." % BK)
    print("  이 스크립트는 2026-10-20 정밀도 수정 **전** 결과와 현재를")
    print("  비교해 기록을 만든다. 수정 전 스냅샷이 없으면 비교할 것이")
    print("  없다. results/precision_fix_20261020.csv 의 기존 사본을 쓴다.")
    raise SystemExit(0)


KEYS = {
    "recalibration_results.csv": ["dataset", "method"],
    "prevalence_control.csv": ["dataset", "setting"],
    "robustness_models.csv": None,
    "recalibration_sensitivity.csv": None,
    "tiebreak_sensitivity.csv": None,
    "grid_calibration.csv": None,
}


def main():
    if not os.path.isdir(BK):
        raise SystemExit("백업이 없다: %s — 이 기록은 백업이 있어야 만들 수 있다" % BK)
    rows = []
    for fn, keys in KEYS.items():
        pa, pb = os.path.join(BK, fn), os.path.join(RES, fn)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        a, b = pd.read_csv(pa), pd.read_csv(pb)
        if len(a) != len(b) or list(a.columns) != list(b.columns):
            rows.append(
                dict(
                    file=fn,
                    row="(구조)",
                    column="-",
                    before=float("nan"),
                    after=float("nan"),
                    abs_diff=float("nan"),
                    changes_4dp=True,
                    note="행수나 열이 달라 셀 대조를 하지 못했다",
                )
            )
            continue
        num = [c for c in a.columns if a[c].dtype.kind in "fi"]
        for i in range(len(a)):
            lab = " / ".join(str(a[k].iloc[i]) for k in keys) if keys else "행 %d" % i
            for c in num:
                x, y = float(a[c].iloc[i]), float(b[c].iloc[i])
                if x == y or (x != x and y != y):
                    continue
                rows.append(
                    dict(
                        file=fn,
                        row=lab,
                        column=c,
                        before=x,
                        after=y,
                        abs_diff=abs(x - y),
                        changes_4dp=(("%.4f" % x) != ("%.4f" % y)),
                        note="",
                    )
                )
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    n4 = int(d.changes_4dp.sum()) if len(d) else 0
    print("정밀도 수정 전·후 대조")
    print("  바뀐 칸 %d개 / 그중 소수 4자리에서 바뀌는 칸 %d개" % (len(d), n4))
    if n4:
        print()
        print("%-30s %-34s %-12s %14s %14s" % ("파일", "행", "열", "전", "후"))
        for _, r in d[d.changes_4dp].iterrows():
            print(
                "%-30s %-34s %-12s %14.8f %14.8f"
                % (r.file, str(r.row)[:34], r.column, r.before, r.after)
            )
    print()
    print("저장: %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
