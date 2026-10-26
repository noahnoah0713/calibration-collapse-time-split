# -*- coding: utf-8 -*-
"""원자료에서 분석집합까지의 제외 경로를 결과 파일에서 모아 CSV 로 쓴다.

왜 이 스크립트가 따로 있는가 — 제외 경로의 수를 그림에 손으로 옮겨 적지
않기 위해서다. 각 단계의 수는 아래 둘 중 하나에서만 온다.

  (가) results/label_audit_summary.txt — 원자료 엑셀을 직접 다시 읽어
       만든 검수 파일. 고유 키 수, concern 제외 수, 이진화 대상 수,
       이후 단계 탈락 합계, 분석집합 크기가 여기 있다. src/verify9.py 가
       이 값들을 문서와 대조한다.
  (나) data/processed/ 에 남아 있는 단계별 스냅샷의 행 수.

두 계통은 서로를 검산한다. 단계별 탈락 수의 합이 (가)의 "이후 단계 탈락"
합계와 맞지 않으면 멈춘다.

두 데이터셋의 파이프라인은 순서가 다르다. DILIrank 는 구조·중복·필터를
먼저 적용하고 승인연도를 마지막에 맞췄고, DICTrank 는 승인연도를 중복
제거 직후에 맞췄다. 그림도 각자의 순서대로 그린다 — 한 줄로 맞추면
실제로 일어난 일이 아니게 된다. DICTrank 는 구조·중복 단계의 스냅샷이
남아 있지 않아 그 구간을 한 단계로 묶는다.
"""

import io
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(RES, "curation_chain.csv")


def rows(name):
    return len(pd.read_csv(os.path.join(PROC, name)))


def audit():
    """label_audit_summary.txt 에서 양 끝과 합계를 읽는다."""
    t = io.open(os.path.join(RES, "label_audit_summary.txt"), encoding="utf-8").read()
    blocks = t.split("DILIrank")[1].split("DICTrank")
    out = {}
    for ds, b in (("DILIrank", blocks[0]), ("DICTrank", blocks[1])):
        m = re.search(r"고유 키 (\d+)행\s*\n\s*- concern 제외\(ambiguous 등\) (\d+)행", b)
        assert m, ds
        uniq, amb = int(m.group(1)), int(m.group(2))
        m2 = re.search(r"= 이진화 대상 (\d+)행", b)
        m3 = re.search(r"- 이후 단계에서 탈락 (\d+)행", b)
        m4 = re.search(r"= 분석집합 (\d+)행", b)
        assert m2 and m3 and m4, ds
        out[ds] = dict(
            uniq=uniq,
            amb=amb,
            labelled=int(m2.group(1)),
            later=int(m3.group(1)),
            final=int(m4.group(1)),
        )
        assert out[ds]["uniq"] - amb == out[ds]["labelled"], ds
        assert out[ds]["labelled"] - out[ds]["later"] == out[ds]["final"], ds
    return out


def main():
    a = audit()

    # ---- DILIrank: 단계마다 스냅샷이 남아 있다
    wsm = pd.read_csv(os.path.join(PROC, "dilirank2_with_smiles.csv"))
    lab = wsm.dropna(subset=["y"])
    assert len(lab) == a["DILIrank"]["labelled"], (len(lab), a["DILIrank"]["labelled"])
    n_struct = int(lab["smiles_raw"].notna().sum())
    n_dedup = rows("dilirank2_clean.csv")
    n_carbon = rows("dilirank2_with_year_preconflict.csv")
    n_conf = rows("dilirank2_with_year.csv")
    wy = pd.read_csv(os.path.join(PROC, "dilirank2_with_year.csv"))
    n_year = int(pd.to_numeric(wy["approval_year"], errors="coerce").notna().sum())

    dili = [
        ("unique records in the source file", a["DILIrank"]["uniq"], "", "label_audit_summary.txt"),
        (
            "labelled, ambiguous concern excluded",
            a["DILIrank"]["labelled"],
            "ambiguous concern",
            "label_audit_summary.txt",
        ),
        ("structure obtained", n_struct, "no structure found", "dilirank2_with_smiles.csv"),
        ("distinct skeletons", n_dedup, "duplicate skeleton", "dilirank2_clean.csv"),
        (
            "carbon-containing",
            n_carbon,
            "no carbon after standardisation",
            "dilirank2_with_year_preconflict.csv",
        ),
        (
            "labels agree within a skeleton",
            n_conf,
            "conflicting labels in a group",
            "dilirank2_with_year.csv",
        ),
        (
            "analysis set",
            n_year,
            "no matched approval year",
            "dilirank2_with_year.csv (approval_year)",
        ),
    ]

    # ---- DICTrank: 구조·중복 단계의 스냅샷이 없다. 승인연도까지를 한 단계로 둔다.
    d_year = rows("dictrank_processed_prefilter.csv")
    d_carbon = rows("dictrank_processed_preconflict.csv")
    d_final = rows("dictrank_processed.csv")
    dict_ = [
        ("unique records in the source file", a["DICTrank"]["uniq"], "", "label_audit_summary.txt"),
        (
            "labelled, ambiguous concern excluded",
            a["DICTrank"]["labelled"],
            "ambiguous concern",
            "label_audit_summary.txt",
        ),
        (
            "structure obtained, skeletons distinct, approval year matched",
            d_year,
            "no structure, duplicate skeleton, or no matched approval year",
            "dictrank_processed_prefilter.csv",
        ),
        (
            "carbon-containing",
            d_carbon,
            "no carbon after standardisation",
            "dictrank_processed_preconflict.csv",
        ),
        ("analysis set", d_final, "conflicting labels in a group", "dictrank_processed.csv"),
    ]

    recs = []
    for ds, steps in (("DILIrank", dili), ("DICTrank", dict_)):
        prev = None
        removed_after_label = 0
        for i, (step, n, why, src) in enumerate(steps):
            rm = 0 if prev is None else prev - n
            assert rm >= 0, (ds, step, prev, n)
            if i >= 1:
                removed_after_label += rm if i >= 2 else 0
            recs.append(
                dict(dataset=ds, order=i + 1, step=step, n=n, removed=rm, reason=why, source=src)
            )
            prev = n
        # 검산 1: 마지막이 분석집합이다
        assert prev == a[ds]["final"], (ds, prev, a[ds]["final"])
        # 검산 2: 이진화 이후 탈락의 합이 검수 파일의 합계와 같다
        assert removed_after_label == a[ds]["later"], (ds, removed_after_label, a[ds]["later"])

    df = pd.DataFrame(recs)
    df.to_csv(OUT, index=False)
    print("저장: results/curation_chain.csv")
    for ds in ("DILIrank", "DICTrank"):
        g = df[df.dataset == ds]
        print("  %-9s %s" % (ds, " -> ".join(str(x) for x in g.n.tolist())))
        print(
            "            이진화 이후 탈락 합 %d (검수 파일 %d) 일치"
            % (int(g.removed[1:].sum()) - int(g.removed.iloc[1]), a[ds]["later"])
        )


if __name__ == "__main__":
    main()
