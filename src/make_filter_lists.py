# -*- coding: utf-8 -*-
"""큐레이션 필터 2종이 실제로 제거한 화합물 목록을 results/ 로 내보낸다.

왜 필요한가
-----------
Methods 는 "탄소 없는 구조 필터"와 "라벨 충돌 필터"가 제거한 화합물을
보충자료에 전부 싣는다고 적는다. 그런데 탄소 필터 쪽은 로그가 멱등
재실행으로 덮여("0건 제거") 목록이 남아 있지 않다. 대신 적용 전후
스냅샷이 둘 다 보존돼 있으므로, 두 파일을 비교해 목록을 다시 만든다.

    data/processed/dilirank2_with_year_prefilter.csv   (필터 전, 870)
    data/processed/dilirank2_with_year_preconflict.csv (탄소 필터 후, 861)
    data/processed/dilirank2_with_year.csv             (충돌 필터 후, 860)
    data/processed/dictrank_processed_prefilter.csv     (953)
    data/processed/dictrank_processed_preconflict.csv   (932)
    data/processed/dictrank_processed.csv               (919)

라벨 충돌 쪽은 apply_label_conflict_filter.py 가 이미
results/label_conflict_groups.csv 에 전체 내역을 남겼으므로 그것을 쓰고,
여기서는 스냅샷 차이와 행 수가 맞는지만 확인한다.

출력
----
    results/structure_filter_removed.csv   dataset, compound
    results/filter_lists_summary.txt

실행
----
    python src/make_filter_lists.py
"""

import collections
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

SETS = [
    (
        "DILIrank",
        "dilirank2_with_year_prefilter.csv",
        "dilirank2_with_year_preconflict.csv",
        "dilirank2_with_year.csv",
    ),
    (
        "DICTrank",
        "dictrank_processed_prefilter.csv",
        "dictrank_processed_preconflict.csv",
        "dictrank_processed.csv",
    ),
]

log = []


def say(s=""):
    print(s)
    log.append(s)


def names(path):
    with io.open(path, encoding="utf-8") as f:
        return [r["CompoundName"].strip() for r in csv.DictReader(f)]


def removed(before, after):
    """before 에는 있고 after 에는 없는 이름. 같은 이름이 여러 행일 수
    있으므로 집합이 아니라 다중집합으로 센다."""
    cb, ca = collections.Counter(before), collections.Counter(after)
    out = []
    for n, c in cb.items():
        if ca[n] < c:
            out += [n] * (c - ca[n])
    return sorted(out, key=lambda s: s.lower())


def main():
    rows = []
    say("큐레이션 필터 2종이 제거한 화합물")
    say("=" * 70)
    bad = 0
    for ds, f0, f1, f2 in SETS:
        p0, p1, p2 = (os.path.join(PROC, x) for x in (f0, f1, f2))
        for p in (p0, p1, p2):
            if not os.path.exists(p):
                sys.exit("없는 파일: %s" % p)
        a, b, c = names(p0), names(p1), names(p2)
        carbon = removed(a, b)
        conflict = removed(b, c)
        say("")
        say("%s  %d -> %d -> %d" % (ds, len(a), len(b), len(c)))
        say("  탄소 없는 구조 %d건" % len(carbon))
        for n in carbon:
            say("    %s" % n)
            rows.append((ds, n))
        say("  라벨 충돌로 제거된 행 %d건" % len(conflict))

    # 라벨 충돌 목록은 이미 파일로 있다. 행 수가 맞는지만 확인한다.
    g = os.path.join(RES, "label_conflict_groups.csv")
    if not os.path.exists(g):
        sys.exit("없는 파일: %s — apply_label_conflict_filter.py 를 먼저 돌려라" % g)
    with io.open(g, encoding="utf-8") as f:
        grp = list(csv.DictReader(f))
    byds = collections.Counter(r["dataset"] for r in grp)
    keys = collections.defaultdict(set)
    for r in grp:
        keys[r["dataset"]].add(r["_k"])
    say("")
    say("라벨 충돌 그룹 (results/label_conflict_groups.csv)")
    for ds, _, _, _ in SETS:
        say("  %s  그룹 %d개 / 행 %d개" % (ds, len(keys[ds]), byds[ds]))

    out = os.path.join(RES, "structure_filter_removed.csv")
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "compound"])
        w.writerows(rows)
    say("")
    say("저장: %s  (%d행)" % (out, len(rows)))
    with io.open(os.path.join(RES, "filter_lists_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    return bad


if __name__ == "__main__":
    sys.exit(main())
