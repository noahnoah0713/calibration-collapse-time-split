"""
SMILES 이름-구조 매칭 검수

왜 필요한가:
    구조는 화합물 '이름'으로 PubChem에 질의해 받아온다. 이름이 애매하거나
    동의어가 겹치면 엉뚱한 분자가 조용히 들어온다. 오류가 나지 않고
    그냥 '다른 화합물'이 되므로, 따로 검사하지 않으면 드러나지 않는다.

무엇을 확인하는가 (전부 오프라인, 네트워크 불필요):
    1) 서로 다른 약 이름이 같은 구조로 합쳐진 경우
       중복 제거 단계에서 한쪽이 조용히 사라진다.
    2) 서로 다른 약 이름이 같은 PubChem CID를 받은 경우
       이름 질의가 같은 화합물로 해석됐다는 뜻.
    3) 두 데이터셋에 공통으로 있는 화합물의 구조 불일치
    4) 표준화 후 분자가 지나치게 작은 경우 (이온·용매·원소 등)
    5) 표준화가 모분자를 바꾼 경우
       염 제거는 정상이지만, 이름에 염 표기가 없는데 바뀌었다면 의심.
    6) PubChem 조회 실패 / CID 없음

사용법:
    python src/check_smiles.py

출력:
    results/smiles_audit.csv
    results/smiles_audit_summary.txt
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MIN_MW = 100  # 이보다 가벼우면 보고
MIN_HEAVY = 6  # 중원자 수가 이보다 적으면 보고

SALT_TOKENS = {
    "hydrochloride",
    "hcl",
    "sulfate",
    "sulphate",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "maleate",
    "tartrate",
    "citrate",
    "acetate",
    "mesylate",
    "besylate",
    "tosylate",
    "fumarate",
    "succinate",
    "phosphate",
    "nitrate",
    "bromide",
    "chloride",
    "iodide",
    "oxalate",
    "malate",
    "lactate",
    "gluconate",
    "stearate",
    "palmitate",
    "carbonate",
    "bitartrate",
    "dihydrochloride",
    "hydrobromide",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "anhydrous",
    "hemihydrate",
    "pamoate",
    "napsylate",
    "embonate",
    "disodium",
    "dipotassium",
    "hyclate",
    "dimaleate",
    "hemifumarate",
    "camsylate",
    "besilate",
    "salt",
    "hydrate",
    "meglumine",
    "trometamol",
    "tromethamine",
    "benzoate",
    "pidolate",
    "camphorsulfonate",
    "edisylate",
    "xinafoate",
    "valerate",
    "propionate",
}

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def norm(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def has_salt_token(name):
    return any(t in norm(name).split() for t in SALT_TOKENS)


def skeleton(smiles):
    """InChIKey 첫 14자리 (골격 해시). 실패하면 None."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def main():
    log("SMILES 이름-구조 매칭 검수")
    log("=" * 74)

    cache_f = PROC / "smiles_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    n_null = sum(1 for v in cache.values() if not v)
    log(
        f"PubChem 캐시 {len(cache):,}건 중 조회 실패 {n_null:,}건 "
        f"({n_null/max(len(cache),1):.1%})"
    )
    log("")

    datasets = [
        ("DILIrank", PROC / "dilirank2_with_year.csv"),
        ("DICTrank", PROC / "dictrank_processed.csv"),
    ]

    rows = []
    per_ds = {}
    for ds, path in datasets:
        if not path.exists():
            log(f"{ds}: {path.name} 없음 - 건너뜀")
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=["smiles"]).copy()
        recs = []
        for _, r in df.iterrows():
            nm = str(r["CompoundName"])
            std = r["smiles"]
            m = Chem.MolFromSmiles(std) if isinstance(std, str) else None
            c = cache.get(nm) or {}
            raw = c.get("smiles_raw")
            recs.append(
                {
                    "dataset": ds,
                    "compound": nm,
                    "pubchem_cid": c.get("pubchem_cid"),
                    "mw_std": Descriptors.MolWt(m) if m else None,
                    "heavy_std": m.GetNumHeavyAtoms() if m else None,
                    "skel_std": skeleton(std),
                    "skel_raw": skeleton(raw) if raw else None,
                    "salt_in_name": has_salt_token(nm),
                    "smiles_std": std,
                }
            )
        d = pd.DataFrame(recs)
        per_ds[ds] = d
        rows.extend(recs)

        log("=" * 74)
        log(f"{ds}   (구조 확보 {len(d)}건)")
        log("=" * 74)

        # 1) 서로 다른 이름 -> 같은 구조
        g = defaultdict(set)
        for _, r in d.iterrows():
            if r["skel_std"]:
                g[r["skel_std"]].add(r["compound"])
        dup = {k: v for k, v in g.items() if len(v) > 1}
        log(f"\n[1] 서로 다른 이름이 같은 구조로 합쳐진 경우: {len(dup)}건")
        if dup:
            log("    (중복 제거 단계에서 각 묶음마다 하나만 남는다)")
            for k, v in list(dup.items())[:25]:
                log(f"    {k}  <-  {sorted(v)}")

        # 2) 서로 다른 이름 -> 같은 CID
        gc = defaultdict(set)
        for _, r in d.iterrows():
            if pd.notna(r["pubchem_cid"]):
                gc[int(r["pubchem_cid"])].add(r["compound"])
        dupc = {k: v for k, v in gc.items() if len(v) > 1}
        log(f"\n[2] 서로 다른 이름이 같은 PubChem CID를 받은 경우: {len(dupc)}건")
        for k, v in list(dupc.items())[:25]:
            log(f"    CID {k}  <-  {sorted(v)}")

        # 4) 너무 작은 분자
        small = d[(d.mw_std.notna()) & ((d.mw_std < MIN_MW) | (d.heavy_std < MIN_HEAVY))]
        log(
            f"\n[4] 표준화 후 분자량 {MIN_MW} 미만 또는 중원자 {MIN_HEAVY}개 미만: "
            f"{len(small)}건"
        )
        if len(small):
            log(
                small[["compound", "mw_std", "heavy_std", "smiles_std"]]
                .sort_values("mw_std")
                .to_string(index=False)
            )

        # 5) 표준화가 모분자를 바꿨는데 이름에 염 표기가 없는 경우
        chg = d[
            (d.skel_raw.notna())
            & (d.skel_std.notna())
            & (d.skel_raw != d.skel_std)
            & (~d.salt_in_name)
        ]
        log(f"\n[5] 표준화로 골격이 바뀌었는데 이름에 염 표기가 없는 경우: {len(chg)}건")
        if len(chg):
            log("    (염·수화물이면 정상. 그 외면 조각 선택이 잘못됐을 수 있다)")
            log(chg[["compound", "mw_std", "skel_raw", "skel_std"]].head(40).to_string(index=False))

        # 6) CID 없음
        nocid = d[d.pubchem_cid.isna()]
        log(f"\n[6] PubChem CID 없음: {len(nocid)}건")
        if len(nocid) and len(nocid) <= 40:
            log(nocid[["compound"]].to_string(index=False))
        log("")

    # 3) 두 데이터셋 교차 대조
    if len(per_ds) == 2:
        a, b = per_ds["DILIrank"].copy(), per_ds["DICTrank"].copy()
        a["k"] = a.compound.map(norm)
        b["k"] = b.compound.map(norm)
        m = (
            a[["k", "compound", "skel_std"]]
            .rename(columns={"compound": "name_dili", "skel_std": "skel_dili"})
            .merge(
                b[["k", "compound", "skel_std"]].rename(
                    columns={"compound": "name_dict", "skel_std": "skel_dict"}
                ),
                on="k",
            )
            .drop_duplicates("k")
        )
        bad = m[(m.skel_dili.notna()) & (m.skel_dict.notna()) & (m.skel_dili != m.skel_dict)]
        log("=" * 74)
        log(f"[3] 두 데이터셋 공통 이름 {len(m)}건의 구조 일치 여부")
        log(f"    불일치: {len(bad)}건")
        if len(bad):
            log(bad[["name_dili", "skel_dili", "skel_dict"]].to_string(index=False))

    out = pd.DataFrame(rows)
    out.to_csv(RES / "smiles_audit.csv", index=False)
    (RES / "smiles_audit_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log("")
    log(f"저장: {RES/'smiles_audit.csv'}")
    log(f"      {RES/'smiles_audit_summary.txt'}")


if __name__ == "__main__":
    main()
