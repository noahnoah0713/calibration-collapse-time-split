"""재실행 전후 결과 파일을 칸 단위로 비교한다 (2026-10-21).

왜 필요한가
-----------
verify9·verify10 은 "문서와 결과 파일이 맞는가" 를 본다. "어제 값과
달라졌는가" 는 아무도 안 본다. 재실행이 결과를 다시 만들면 어떤 값이
몇째 자리에서 움직였는지, 그리고 움직인 값이 본문 문장에 박혀 있는지를
확인해야 한다. 소수 4자리에서 바뀐 칸만 보고서 수치에 영향을 준다.

쓰기
----
    python src/diff_results.py _backup/precision_20261020
    python src/diff_results.py <백업디렉터리> --save   # _backup 에 CSV 로
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def isnum(c):
    return pd.api.types.is_numeric_dtype(c)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    old_dir = os.path.join(ROOT, sys.argv[1])
    save = "--save" in sys.argv
    if not os.path.isdir(old_dir):
        print("없는 디렉터리: %s" % old_dir)
        raise SystemExit(2)

    rows = []
    only_new, only_old, shape_diff, unreadable = [], [], [], []
    news = {f for f in os.listdir(RES) if f.endswith(".csv")}
    olds = {f for f in os.listdir(old_dir) if f.endswith(".csv")}
    only_new = sorted(news - olds)
    only_old = sorted(olds - news)

    for f in sorted(news & olds):
        try:
            a = pd.read_csv(os.path.join(old_dir, f))
            b = pd.read_csv(os.path.join(RES, f))
        except Exception as e:
            unreadable.append((f, str(e)[:60]))
            continue
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            shape_diff.append((f, a.shape, b.shape))
            continue
        key = [c for c in a.columns if not isnum(a[c])]
        for c in a.columns:
            if not isnum(a[c]):
                continue
            for i in range(len(a)):
                x, y = a[c][i], b[c][i]
                if pd.isna(x) and pd.isna(y):
                    continue
                if pd.isna(x) or pd.isna(y) or x != y:
                    xf = float(x) if not pd.isna(x) else float("nan")
                    yf = float(y) if not pd.isna(y) else float("nan")
                    rows.append(
                        {
                            "file": f,
                            "row": i,
                            "key": " / ".join(str(a[q][i]) for q in key)[:60],
                            "column": c,
                            "before": xf,
                            "after": yf,
                            "abs_diff": abs(yf - xf),
                            "changes_4dp": ("%.4f" % xf) != ("%.4f" % yf),
                        }
                    )

    d = pd.DataFrame(rows)
    print("=" * 78)
    print("결과 파일 전후 비교   기준: %s" % sys.argv[1])
    print("=" * 78)
    print(
        "공통 CSV %d개 / 새로 생긴 것 %d개 / 사라진 것 %d개"
        % (len(news & olds), len(only_new), len(only_old))
    )
    if only_new:
        print("  새로 생긴 것:", ", ".join(only_new))
    if only_old:
        print("  사라진 것:", ", ".join(only_old))
    if shape_diff:
        print("  모양이 달라진 것 %d개:" % len(shape_diff))
        for f, s1, s2 in shape_diff:
            print("    %-44s %s -> %s" % (f, s1, s2))
    if unreadable:
        print("  읽지 못한 것:", unreadable)
    if d.empty:
        print("\n바뀐 칸 없음 — 모든 공통 파일이 비트 단위로 같다.")
        return
    n4 = int(d.changes_4dp.sum())
    print("\n바뀐 칸 %d개   소수 4자리에서 바뀐 칸 %d개" % (len(d), n4))
    print("\n파일별 (4자리 변동이 있는 것부터)")
    g = (
        d.groupby("file")
        .agg(바뀐칸=("column", "size"), 네자리=("changes_4dp", "sum"), 최대차=("abs_diff", "max"))
        .sort_values(["네자리", "최대차"], ascending=False)
    )
    for f, r in g.iterrows():
        print(
            "  %-44s 바뀐칸 %4d  4자리 %4d  최대차 %.2e"
            % (f, int(r.바뀐칸), int(r.네자리), r.최대차)
        )
    if n4:
        print("\n소수 4자리에서 바뀐 칸 (큰 것부터 40개)")
        for _, r in d[d.changes_4dp].sort_values("abs_diff", ascending=False).head(40).iterrows():
            print("  %-34s %-46s %-14s %.6f -> %.6f" % (r.file, r.key, r.column, r.before, r.after))
    if save:
        out = os.path.join(ROOT, "_backup", "diff_%s.csv" % sys.argv[1].replace("/", "_"))
        d.to_csv(out, index=False)
        print("\n기록: %s" % out)


if __name__ == "__main__":
    main()
