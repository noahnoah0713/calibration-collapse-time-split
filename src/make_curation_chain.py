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
import sys

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


# ---------------------------------------------------------------- 재현 점검
# `python src/make_curation_chain.py --recheck`
#
# 무엇을 보는가 — 원자료와 SMILES 캐시에서 DICTrank 큐레이션을 지금 코드로
# 처음부터 다시 돌려, **나오는 화합물 집합이 저장된 분석집합과 같은지**를
# 본다. 중간 수치가 아니라 집합이 판정 기준이다.
#
# 왜 중간 수치가 아닌가 — 2026-10-26 확인: dictrank_pipeline.standardize 는
# 탄소 조각이 없으면 None 을 돌려준다. 저장된 파일이 만들어질 때는 그
# 규칙이 뒤쪽의 별도 필터였다(그래서 structure_filter_log.txt 가 "0건
# 제거, 이미 적용됨" 이다). 그 결과 같은 목적지로 가는 경로가 둘이다.
#
#   당시 경로   1,087 -> 968(중복) -> 953(연도) -> 932(탄소) -> 919(라벨충돌)
#   현재 경로   1,087 -> 1,057(표준화에서 탄소 제외) -> 945(중복)
#                     -> 932(연도) -> 919(라벨충돌)
#
# 두 경로 모두 932 를 지나 919 로 끝나고, 끝에 남는 화합물도 같다.
# 그러므로 Methods 의 1,087 과 968 은 "그때 그 경로의 수" 로 읽어야 하며,
# 결과가 달라지는 것은 아니다. 이 점검은 그 사실을 계속 확인해 준다.
#
# 몇 분 걸린다(Drugs@FDA 벌크 파일을 읽는다). 평소 재실행에는 넣지 않는다.
def recheck():
    import json

    import rdkit
    from rdkit import Chem, RDLogger

    sp = os.path.join(ROOT, "src")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    from dictrank_pipeline import load_fda_years, map_year, standardize

    RDLogger.DisableLog("rdApp.*")

    d = pd.read_excel(
        os.path.join(ROOT, "data", "raw", "dictrank_dataset_508.xlsx"), sheet_name="Table S1"
    )
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns={"Active Ingredient(s)": "CompoundName", "DICT _ Concern": "concern"})
    n_raw = len(d)
    d["y"] = d["concern"].astype(str).str.lower().str.strip().map({"most": 1, "less": 1, "no": 0})
    d = d.dropna(subset=["y", "CompoundName"]).copy()
    n_lab = len(d)

    cache = json.load(io.open(os.path.join(PROC, "smiles_cache.json"), encoding="utf-8"))
    d["smiles_raw"] = [
        (cache.get(str(n).strip()) or {}).get("smiles_raw") for n in d["CompoundName"]
    ]
    d = d.dropna(subset=["smiles_raw"])
    n_smi = len(d)

    d["smiles"] = d["smiles_raw"].apply(standardize)
    d = d.dropna(subset=["smiles"])
    n_std = len(d)

    d["ikey14"] = d["smiles"].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x))[:14])
    pre = d.copy()
    d = d.drop_duplicates("ikey14").reset_index(drop=True)
    n_ded = len(d)

    mf, ms, mfw = load_fda_years()
    d["approval_year"] = d["CompoundName"].astype(str).apply(lambda n: map_year(n, mf, ms, mfw))
    d = d.dropna(subset=["approval_year"])
    n_year = len(d)

    grp = pre.groupby("ikey14")["y"].nunique()
    bad = set(grp[grp > 1].index)
    fresh = d[~d["ikey14"].isin(bad)]

    cur = pd.read_csv(os.path.join(PROC, "dictrank_processed.csv"))
    a = set(fresh["ikey14"].astype(str))
    b = set(cur["ikey14"].astype(str))
    same = a == b

    L = []
    w = L.append
    w("DICTrank 큐레이션 재현 점검 — 원자료에서 다시 돌려 분석집합과 대조")
    w("=" * 70)
    w(
        "환경: RDKit %s / pandas %s / Python %s"
        % (rdkit.__version__, pd.__version__, sys.version.split()[0])
    )
    w("")
    w("  원자료 행                    %5d" % n_raw)
    w("  라벨 이진화 후               %5d" % n_lab)
    w("  SMILES 확보(캐시 적중)       %5d   (Methods: 1087)" % n_smi)
    w("  표준화 통과                  %5d" % n_std)
    w("  중복 제거(14자리 키)         %5d   (당시 경로: 968)" % n_ded)
    w("  승인연도 확보                %5d" % n_year)
    w("  라벨 충돌 키 제외 후         %5d" % len(fresh))
    w("")
    w("  저장된 분석집합              %5d" % len(cur))
    w("  겹치는 화합물                %5d" % len(a & b))
    w("  재현본에만 있는 것           %5d" % len(a - b))
    w("  저장본에만 있는 것           %5d" % len(b - a))
    w("")
    if same:
        w("판정: 화합물 집합이 완전히 같다. 지금 코드로 원자료에서 다시 돌려도")
        w("      분석집합이 그대로 나온다. 중간 수치가 다른 것은 탄소 규칙이")
        w("      별도 필터에서 표준화 안으로 옮겨간 탓이며, 결과를 바꾸지 않는다.")
    else:
        w("판정: 화합물 집합이 다르다. 투고 전에 반드시 확인할 것.")
    txt = "\n".join(L) + "\n"
    io.open(os.path.join(RES, "curation_chain_recheck.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("저장: results/curation_chain_recheck.txt")
    return 0 if same else 1


if __name__ == "__main__":
    if "--recheck" in sys.argv:
        sys.exit(recheck())
    else:
        main()
