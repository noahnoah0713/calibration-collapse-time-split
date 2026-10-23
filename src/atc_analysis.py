"""
DILIrank 2.0 - ATC 치료군별 층화 분석

PubChem에서 ATC 코드를 조회한 뒤, 검증군(2010년 이후 승인)의 성능을
치료군별로 분해한다. ATC 매칭이 부족할 경우를 대비해 구조 기반
보조 층화(키나아제 억제제 등)도 함께 수행한다.

사용법:
    python src/atc_analysis.py

출력:
    data/processed/atc_cache.json
    results/atc_stratified.csv
    results/atc_summary.txt
"""

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CACHE = PROC / "atc_cache.json"
SEED, CUTOFF = 42, 2010
SLEEP = 0.25

ATC_NAMES = {
    "A": "소화기·대사",
    "B": "혈액·조혈",
    "C": "심혈관",
    "D": "피부",
    "G": "비뇨생식·성호르몬",
    "H": "전신호르몬",
    "J": "전신항감염",
    "L": "항종양·면역조절",
    "M": "근골격",
    "N": "신경계",
    "P": "구충·살충",
    "R": "호흡기",
    "S": "감각기",
    "V": "기타",
}

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


# ------------------------------------------------------------- ATC 조회
def fetch_atc(name: str):
    """PubChem 화합물 이름 -> ATC 코드 목록."""
    try:
        u = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{requests.utils.quote(name)}/cids/JSON"
        )
        r = requests.get(u, timeout=20)
        if r.status_code != 200:
            return []
        cid = r.json()["IdentifierList"]["CID"][0]

        u2 = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
            f"{cid}/JSON?heading=ATC+Code"
        )
        r2 = requests.get(u2, timeout=25)
        if r2.status_code != 200:
            return []
        codes = re.findall(r"\b([A-V]\d{2}[A-Z]{2}\d{2})\b", r2.text)
        return sorted(set(codes))
    except Exception:
        return []


def load_atc(names):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [n for n in dict.fromkeys(names) if n not in cache]
    if todo:
        log(f"ATC 조회 대상 {len(todo)}개 (예상 {len(todo)*SLEEP*2/60:.0f}분)")
        for i, n in enumerate(todo, 1):
            cache[n] = fetch_atc(n)
            time.sleep(SLEEP)
            if i % 25 == 0 or i == len(todo):
                hit = sum(1 for v in cache.values() if v)
                log(f"  [{i}/{len(todo)}] 누적 확보 {hit}건")
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


# ------------------------------------------------------------- 모델
def featurize(smiles):
    fps, desc = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=np.int8))
        desc.append(
            [
                Descriptors.MolWt(m),
                Descriptors.MolLogP(m),
                Descriptors.TPSA(m),
                Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m),
            ]
        )
    return np.hstack([np.array(fps), np.array(desc)])


def make_model():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


def main():
    df = pd.read_csv(PROC / "dilirank2_with_year.csv")
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.reset_index(drop=True)

    log("=" * 60)
    log("ATC 치료군별 층화 분석")
    log("=" * 60)

    cache = load_atc(df["CompoundName"].astype(str).tolist())
    df["atc"] = df["CompoundName"].astype(str).map(lambda n: cache.get(n, []))
    df["atc1"] = df["atc"].map(lambda c: c[0][0] if c else None)

    ok = df["atc1"].notna().sum()
    log(f"\nATC 확보: {ok}/{len(df)} ({ok/len(df):.1%})\n")

    # 모델 학습 (시간분할)
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr
    m = make_model().fit(X[tr], y[tr])
    prob = m.predict_proba(X[te])[:, 1]

    test = df[te].copy()
    test["prob"] = prob
    overall = roc_auc_score(test["y"], test["prob"])
    log(f"검증군 전체 AUROC: {overall:.4f} (n={len(test)})\n")

    # -------------------------------------------------- ATC 1단계별
    log("--- ATC 1단계별 성능 (검증군) ---")
    rows = []
    for code, grp in test.dropna(subset=["atc1"]).groupby("atc1"):
        if len(grp) < 10:
            continue
        pos = grp["y"].mean()
        auc = roc_auc_score(grp["y"], grp["prob"]) if grp["y"].nunique() == 2 else np.nan
        rows.append(
            {
                "atc": code,
                "name": ATC_NAMES.get(code, code),
                "n": len(grp),
                "pos_rate": pos,
                "auroc": auc,
                "delta": overall - auc if pd.notna(auc) else np.nan,
            }
        )
        log(
            f"  {code} {ATC_NAMES.get(code,''):12s} n={len(grp):3d} "
            f"양성{pos:5.0%}  AUROC "
            f"{f'{auc:.4f}' if pd.notna(auc) else '  N/A '}"
        )

    if rows:
        out = pd.DataFrame(rows).sort_values("auroc")
        out.to_csv(RES / "atc_stratified.csv", index=False)
        log("\n  성능 최저 계열:")
        for _, r in out.head(3).iterrows():
            if pd.notna(r["auroc"]):
                log(
                    f"    {r['atc']} {r['name']}: {r['auroc']:.4f} "
                    f"(전체 대비 {r['auroc']-overall:+.4f})"
                )

    # -------------------------------------------------- 항종양제 집중
    log("\n--- 항종양제(L) vs 그 외 ---")
    is_L = test["atc1"] == "L"
    for tag, mask in [("항종양·면역조절 (L)", is_L), ("그 외", test["atc1"].notna() & ~is_L)]:
        g = test[mask]
        if len(g) >= 10 and g["y"].nunique() == 2:
            log(
                f"  {tag:22s} n={len(g):3d} 양성{g['y'].mean():5.0%} "
                f"AUROC {roc_auc_score(g['y'], g['prob']):.4f}"
            )
        else:
            log(f"  {tag:22s} n={len(g):3d} (표본 부족)")

    # -------------------------------------------------- 구조 기반 보조 층화
    log("\n--- 구조 기반 보조 층화 ---")

    def flags(s):
        mol = Chem.MolFromSmiles(s)
        return {
            "MW>500": Descriptors.MolWt(mol) > 500,
            "방향환>=4": Chem.rdMolDescriptors.CalcNumAromaticRings(mol) >= 4,
            "N원자>=6": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N") >= 6,
        }

    fl = pd.DataFrame([flags(s) for s in test["smiles"]], index=test.index)
    for col in fl.columns:
        for tag, mask in [(f"{col} 참", fl[col]), (f"{col} 거짓", ~fl[col])]:
            g = test[mask]
            if len(g) >= 20 and g["y"].nunique() == 2:
                log(f"  {tag:16s} n={len(g):3d} " f"AUROC {roc_auc_score(g['y'], g['prob']):.4f}")

    (RES / "atc_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'atc_stratified.csv'}, {RES/'atc_summary.txt'}")


if __name__ == "__main__":
    main()
