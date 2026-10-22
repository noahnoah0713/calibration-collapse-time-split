"""
ATC 치료군별 / 구조 기반 층화 분석 (KEGG, 두 데이터셋 공통)

PubChem의 ATC 항목은 결측이 많고 응답 파싱이 불안정하여 KEGG로 교체.
KEGG는 전체 약물-ATC 매핑을 2회 요청으로 일괄 확보 가능.

  http://rest.kegg.jp/list/drug      : 전체 약물 ID + 이름
  http://rest.kegg.jp/link/atc/drug  : 약물 -> ATC 링크

사용법:
    python src/atc_kegg.py

DILIrank 와 DICTrank 를 같은 절차로 돌린다. 예전에는 DICTrank 쪽
층화 검정이 dictrank_pipeline.py 안에 있었는데, 그 스크립트는 처리된
CSV 를 다시 만들기 때문에 구조 필터 이후의 재실행 목록에서 빠져 있었다.
그래서 DICTrank 의 p 값만 필터 이전 값으로 남아 있었다. 여기로 옮겨
두 데이터셋이 항상 같은 시점의 값을 갖도록 한다.

출력:
    data/processed/kegg_atc_map.csv
    results/atc_stratified.csv
    results/atc_tests.csv      <- 다중비교 보정이 읽는 파일
    results/atc_summary.txt
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
N_BOOT = 2000
rng = np.random.default_rng(SEED)

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

SALT_WORDS = {
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
}

log_lines = []


def log(m=""):
    print(m)
    log_lines.append(str(m))


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)  # (USP), (TN) 등 제거
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_salt(s: str) -> str:
    t = norm(s).split()
    while len(t) > 1 and t[-1] in SALT_WORDS:
        t.pop()
    return " ".join(t)


# --------------------------------------------------------- KEGG 일괄 수집
def build_kegg_map() -> dict:
    cache = PROC / "kegg_atc_map.csv"
    if cache.exists():
        m = pd.read_csv(cache)
        log(f"KEGG 매핑 캐시 사용: {len(m):,}행")
        return dict(zip(m["key"], m["atc"]))

    log("KEGG에서 약물-ATC 매핑 수집 중...")
    names = requests.get("http://rest.kegg.jp/list/drug", timeout=60).text
    links = requests.get("http://rest.kegg.jp/link/atc/drug", timeout=60).text

    # 약물 ID -> ATC 코드들
    d2a = {}
    for line in links.strip().split("\n"):
        if "\t" not in line:
            continue
        did, atc = line.split("\t")
        d2a.setdefault(did.replace("dr:", ""), []).append(atc.replace("atc:", ""))
    log(f"  ATC 보유 약물: {len(d2a):,}개")

    # 약물 ID -> 이름들
    rows = []
    for line in names.strip().split("\n"):
        if "\t" not in line:
            continue
        did, nm = line.split("\t", 1)
        did = did.replace("dr:", "")
        if did not in d2a:
            continue
        atc = sorted(d2a[did])[0]  # 대표 코드 1개
        for piece in nm.split(";"):
            for key in {norm(piece), strip_salt(piece)}:
                if len(key) >= 4:
                    rows.append({"key": key, "atc": atc})

    m = pd.DataFrame(rows).drop_duplicates("key")
    m.to_csv(cache, index=False)
    log(f"  이름 키 {len(m):,}개 생성 -> {cache.name}")
    return dict(zip(m["key"], m["atc"]))


def lookup(name: str, kmap: dict):
    for k in (norm(name), strip_salt(name)):
        if k in kmap:
            return kmap[k]
    return None


# --------------------------------------------------------------- 모델
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


def boot_auc_diff(y1, p1, y2, p2):
    """두 독립 하위집단 AUROC 차이의 부트스트랩 양측 p값."""
    i1, i2 = np.arange(len(y1)), np.arange(len(y2))
    diffs = []
    for _ in range(N_BOOT):
        s1 = rng.choice(i1, len(i1), replace=True)
        s2 = rng.choice(i2, len(i2), replace=True)
        if len(np.unique(y1[s1])) < 2 or len(np.unique(y2[s2])) < 2:
            continue
        diffs.append(roc_auc_score(y1[s1], p1[s1]) - roc_auc_score(y2[s2], p2[s2]))
    if not diffs:
        return np.nan
    d = np.array(diffs)
    return 2 * min((d <= 0).mean(), (d >= 0).mean())


def fit_temporal(df):
    """시간분할 학습 후 검증군 데이터프레임(+prob)과 전체 AUROC 반환."""
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X[tr], y[tr])
    test = df[te].copy()
    test["prob"] = model.predict_proba(X[te])[:, 1]
    return test


def struct_feats(s):
    m = Chem.MolFromSmiles(s)
    return {
        "MW>500": Descriptors.MolWt(m) > 500,
        "방향환>=4": rdMolDescriptors.CalcNumAromaticRings(m) >= 4,
        "N>=6": sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N") >= 6,
        "logP>=3": Descriptors.MolLogP(m) >= 3,
    }


def analyse(name, df, kmap, rows_strat, rows_tests):
    """한 데이터셋의 ATC 층화 + 구조 층화. 검정 결과를 rows_tests 에 쌓는다."""
    log("=" * 62)
    log(f"{name}  n={len(df)}")
    log("=" * 62)

    df = df.copy()
    df["atc"] = df["CompoundName"].astype(str).map(lambda n: lookup(n, kmap))
    df["atc1"] = df["atc"].str[0]
    ok = df["atc1"].notna().sum()
    log(f"  ATC 확보: {ok}/{len(df)} ({ok/len(df):.1%})")

    log("  ATC 1단계 분포: " + ", ".join(f"{k}{v}" for k, v in df["atc1"].value_counts().items()))

    test = fit_temporal(df)
    overall = roc_auc_score(test["y"], test["prob"])
    log(
        f"  검증군 전체 AUROC: {overall:.4f} (n={len(test)}, "
        f"ATC 보유 {test['atc1'].notna().sum()})"
    )

    # ---------------------------------------------------- ATC 계열별
    log(f"\n  --- ATC 1단계별 성능 (검증군, n>=10) ---")
    for code, g in test.dropna(subset=["atc1"]).groupby("atc1"):
        if len(g) < 10:
            continue
        auc = roc_auc_score(g["y"], g["prob"]) if g["y"].nunique() == 2 else np.nan
        rows_strat.append(
            {
                "dataset": name,
                "atc": code,
                "name": ATC_NAMES.get(code, code),
                "n": len(g),
                "pos_rate": g["y"].mean(),
                "auroc": auc,
            }
        )
        log(
            f"    {code} {ATC_NAMES.get(code,''):16s} n={len(g):3d} "
            f"양성{g['y'].mean():5.0%}  "
            f"AUROC {f'{auc:.4f}' if pd.notna(auc) else ' N/A  '}"
        )

    # ---------------------------------------------------- L vs 그 외
    log(f"\n  --- 항종양·면역조절(L) vs 그 외 ---")
    sub = test.dropna(subset=["atc1"])
    L, O = sub[sub.atc1 == "L"], sub[sub.atc1 != "L"]
    if len(L) >= 10 and L["y"].nunique() == 2 and O["y"].nunique() == 2:
        aL = roc_auc_score(L["y"], L["prob"])
        aO = roc_auc_score(O["y"], O["prob"])
        p = boot_auc_diff(L["y"].values, L["prob"].values, O["y"].values, O["prob"].values)
        log(f"    L    n={len(L):3d} 양성{L['y'].mean():5.0%} AUROC {aL:.4f}")
        log(f"    그외 n={len(O):3d} 양성{O['y'].mean():5.0%} AUROC {aO:.4f}")
        log(
            f"    차이 {aL-aO:+.4f}, 부트스트랩 p = {p:.4f}"
            f"  {'*유의*' if p < 0.05 else '(유의하지 않음)'}"
        )
        rows_tests.append(
            {
                "dataset": name,
                "family": "3_exploratory_subgroup",
                "test": "ATC L(항종양) vs 그외",
                "n_a": len(L),
                "n_b": len(O),
                "auroc_a": aL,
                "auroc_b": aO,
                "diff": aL - aO,
                "p": p,
            }
        )
    else:
        log(f"    L n={len(L)} - 표본 부족으로 검정 불가")

    # ---------------------------------------------------- 구조 기반
    log(f"\n  --- 구조 기반 층화 (부트스트랩 검정) ---")
    F = pd.DataFrame([struct_feats(s) for s in test["smiles"]], index=test.index)
    for col in F.columns:
        a, b = test[F[col]], test[~F[col]]
        if not (len(a) >= 20 and len(b) >= 20 and a["y"].nunique() == 2 and b["y"].nunique() == 2):
            log(f"    {col:10s} 표본 부족으로 검정 불가 " f"(참 n={len(a)}, 거짓 n={len(b)})")
            continue
        aa = roc_auc_score(a["y"], a["prob"])
        ab = roc_auc_score(b["y"], b["prob"])
        p = boot_auc_diff(a["y"].values, a["prob"].values, b["y"].values, b["prob"].values)
        log(
            f"    {col:10s} 참 n={len(a):3d} AUROC {aa:.4f} | "
            f"거짓 n={len(b):3d} AUROC {ab:.4f} | "
            f"차이 {aa-ab:+.4f}, p={p:.4f} {'*' if p < 0.05 else ''}"
        )
        rows_tests.append(
            {
                "dataset": name,
                "family": "3_exploratory_subgroup",
                "test": f"구조: {col}",
                "n_a": len(a),
                "n_b": len(b),
                "auroc_a": aa,
                "auroc_b": ab,
                "diff": aa - ab,
                "p": p,
            }
        )
    log()


def load(path):
    df = pd.read_csv(path).dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)


def main():
    log("=" * 62)
    log("ATC / 구조 기반 층화 분석 (KEGG)")
    log("=" * 62)

    kmap = build_kegg_map()

    # 매핑 정합성 확인 (알려진 정답)
    log("\n[매핑 정합성 확인]")
    for nm, exp in [
        ("Abemaciclib", "L"),
        ("Abiraterone acetate", "L"),
        ("Atorvastatin calcium", "C"),
        ("Acetaminophen", "N"),
        ("Amoxicillin", "J"),
    ]:
        got = lookup(nm, kmap)
        mark = "OK" if got and got[0] == exp else "  "
        log(f"  {mark} {nm:24s} 기대 {exp} / 실제 {got}")
    log()

    rows_strat, rows_tests = [], []
    for name, path in [
        ("DILIrank", PROC / "dilirank2_with_year.csv"),
        ("DICTrank", PROC / "dictrank_processed.csv"),
    ]:
        if not path.exists():
            log(f"[건너뜀] {name}: {path} 없음")
            continue
        analyse(name, load(path), kmap, rows_strat, rows_tests)

    if rows_strat:
        pd.DataFrame(rows_strat).sort_values(["dataset", "auroc"]).to_csv(
            RES / "atc_stratified.csv", index=False
        )
    if rows_tests:
        pd.DataFrame(rows_tests).to_csv(RES / "atc_tests.csv", index=False)

    log("=" * 62)
    log("주의: 여기의 검정은 전부 사후 탐색이다. 다중비교 보정 전의 p 값이므로")
    log("      단독으로 인용하지 말 것. 보정값은")
    log("      python src/multiple_comparison_correction.py 가 계산한다.")

    (RES / "atc_summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'atc_stratified.csv'}")
    log(f"      {RES/'atc_tests.csv'}")
    log(f"      {RES/'atc_summary.txt'}")


if __name__ == "__main__":
    main()
