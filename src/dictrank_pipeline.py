"""
DICTrank (심장독성) 시간분할 재현 실험

DILIrank에서 확인된 현상 - 시간분할 시 성능 저하, 특히 항종양제에서 붕괴 -
가 다른 독성 엔드포인트에서도 나타나는지 검증한다.

DILIrank 파이프라인의 캐시(SMILES, ATC)를 재사용한다.

주의 — 이 스크립트는 data/processed/dictrank_processed.csv 를 "다시 만든다".
구조 필터(apply_structure_filter.py)를 적용한 뒤에 이것을 돌리면 필터가
되돌아간다. 그래서 rerun_all.sh 의 재실행 목록에서 빠져 있고, 원자료부터
다시 만들 때만 실행한다.

그 때문에 예전에는 여기 들어 있던 순열 검정과 ATC 층화 결과만 필터 이전
값으로 남는 문제가 있었다. 두 분석을 모두 옮겼다.

    순열 검정  -> src/permutation_test.py  (두 데이터셋 공통, 1,000회)
    ATC 층화   -> src/atc_kegg.py          (두 데이터셋 공통)

사용법:
    python src/dictrank_pipeline.py

출력:
    data/processed/dictrank_processed.csv
    results/dictrank_results.txt
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED, CUTOFF = 42, 2010
# 순열 반복 수는 src/permutation_test.py 가 관리한다
rng = np.random.default_rng(SEED)

# ATC_NAMES 는 층화 분석과 함께 src/atc_kegg.py 로 옮겼다.

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
    "acid",
    "alfa",
    "beta",
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


def strip_salt(s):
    t = norm(s).split()
    while len(t) > 1 and t[-1] in SALT_WORDS:
        t.pop()
    return " ".join(t)


def first_word(s):
    t = norm(s).split()
    return t[0] if t else ""


# ------------------------------------------------------------ SMILES
def pick_smiles(p):
    for k in ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES"):
        if p.get(k):
            return p[k]
    return None


def fetch_smiles(names):
    cache_f = PROC / "smiles_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    log(f"기존 SMILES 캐시: {len(cache)}건 (DILIrank에서 재사용)")

    todo = [
        n
        for n in dict.fromkeys(names)
        if n not in cache or not (cache.get(n) or {}).get("smiles_raw")
    ]
    log(f"신규 조회 대상: {len(todo)}건")

    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    props = "SMILES,MolecularFormula,MolecularWeight,InChIKey"
    for i, n in enumerate(todo, 1):
        try:
            r = requests.get(
                f"{base}/compound/name/{requests.utils.quote(n)}" f"/property/{props}/JSON",
                timeout=20,
            )
            if r.status_code == 200:
                p = r.json()["PropertyTable"]["Properties"][0]
                cache[n] = {
                    "pubchem_cid": p.get("CID"),
                    "smiles_raw": pick_smiles(p),
                    "inchikey": p.get("InChIKey"),
                }
            else:
                cache[n] = None
        except requests.RequestException:
            cache[n] = None
        time.sleep(0.25)
        if i % 50 == 0 or i == len(todo):
            log(f"  [{i}/{len(todo)}]")
            cache_f.write_text(json.dumps(cache, ensure_ascii=False))
    cache_f.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


# ------------------------------------------------------------ 승인연도
def load_fda_years():
    rawd = RAW / "drugsfda"
    prod = pd.read_csv(
        rawd / "Products.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = pd.read_csv(
        rawd / "Submissions.txt", sep="\t", dtype=str, encoding="latin-1", on_bad_lines="skip"
    )
    subm = subm[subm["SubmissionStatus"] == "AP"].copy()
    subm["date"] = pd.to_datetime(subm["SubmissionStatusDate"], errors="coerce")
    subm = subm.dropna(subset=["date"])
    first = subm.groupby("ApplNo")["date"].min().reset_index().rename(columns={"date": "first_ap"})
    prod = prod.merge(first, on="ApplNo").dropna(subset=["ActiveIngredient"])
    prod["ActiveIngredient"] = prod["ActiveIngredient"].str.split(";")
    prod = prod.explode("ActiveIngredient")
    prod["ing"] = prod["ActiveIngredient"].str.strip()
    ing = prod.groupby("ing")["first_ap"].min().reset_index()
    ing["year"] = ing["first_ap"].dt.year
    ing["k_full"] = ing["ing"].apply(norm)
    ing["k_salt"] = ing["ing"].apply(strip_salt)
    ing["k_first"] = ing["ing"].apply(first_word)

    def mk(c):
        return ing.sort_values("year").drop_duplicates(c).set_index(c)["year"].to_dict()

    return mk("k_full"), mk("k_salt"), mk("k_first")


def map_year(name, m_full, m_salt, m_first):
    n1, n2, n3 = norm(name), strip_salt(name), first_word(name)
    if n1 in m_full:
        return m_full[n1]
    if n2 in m_salt:
        return m_salt[n2]
    if n2 in m_full:
        return m_full[n2]
    if n3 and n3 in m_first:
        return m_first[n3]
    return None


# ------------------------------------------------------------ 공통
def standardize(smi):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    # 탄소 조각이 없으면 무기물·금속착물이므로 제외 (baseline.py와 동일 규칙)
    org = [f for f in frags if any(a.GetSymbol() == "C" for a in f.GetAtoms())]
    if not org:
        return None
    best = max(org, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(best)
    except Exception:
        return None


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


# (boot_diff 는 ATC 층화와 함께 src/atc_kegg.py 로 옮겨 여기서는 삭제)


# ------------------------------------------------------------------ main
def main():
    log("=" * 62)
    log("DICTrank (심장독성) 시간분할 재현 실험")
    log("=" * 62)

    df = pd.read_excel(RAW / "dictrank_dataset_508.xlsx", sheet_name="Table S1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Active Ingredient(s)": "CompoundName", "DICT _ Concern": "concern"})
    log(f"원본: {len(df)}행")

    df["label"] = df["concern"].astype(str).str.lower().str.strip()
    log("\n라벨 분포:")
    log(df["label"].value_counts().to_string())

    # 이진화: most + less = 양성, no = 음성, ambiguous 제외
    df["y"] = df["label"].map({"most": 1, "less": 1, "no": 0})
    df = df.dropna(subset=["y", "CompoundName"]).copy()
    df["y"] = df["y"].astype(int)
    log(f"\n이진화 후: {len(df)}행 (양성 {df.y.sum()}, 음성 {len(df)-df.y.sum()})")

    # SMILES
    log("\n--- SMILES 확보 ---")
    names = df["CompoundName"].astype(str).str.strip().tolist()
    cache = fetch_smiles(names)
    df["smiles_raw"] = [(cache.get(n) or {}).get("smiles_raw") for n in names]
    df = df.dropna(subset=["smiles_raw"])
    df["smiles"] = df["smiles_raw"].apply(standardize)
    df = df.dropna(subset=["smiles"])
    log(f"SMILES 확보: {len(df)}행")

    df["ikey14"] = df["smiles"].apply(lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14])
    n0 = len(df)
    # 중복 제거 "이전" 스냅샷을 남긴다. src/check_labels.py 가 서로 다른
    # 화합물이 한 줄로 접히면서 라벨이 버려지지 않았는지 확인하는 데 쓴다.
    df.to_csv(PROC / "dictrank_prededup.csv", index=False)

    df = df.drop_duplicates("ikey14").reset_index(drop=True)
    log(f"중복 제거: {n0} -> {len(df)}행")

    # 승인연도
    log("\n--- 승인연도 매핑 ---")
    mf, ms, mfw = load_fda_years()
    df["approval_year"] = df["CompoundName"].astype(str).apply(lambda n: map_year(n, mf, ms, mfw))
    df = df.dropna(subset=["approval_year"])
    df["approval_year"] = df["approval_year"].astype(int)
    log(f"승인연도 확보: {len(df)}행")

    # ATC
    kf = PROC / "kegg_atc_map.csv"
    if kf.exists():
        km = pd.read_csv(kf)
        kmap = dict(zip(km["key"], km["atc"]))
        df["atc1"] = (
            df["CompoundName"]
            .astype(str)
            .apply(
                lambda n: (
                    (kmap.get(norm(n)) or kmap.get(strip_salt(n)) or [None])[0]
                    if (kmap.get(norm(n)) or kmap.get(strip_salt(n)))
                    else None
                )
            )
        )
        log(f"ATC 확보: {df['atc1'].notna().sum()}/{len(df)}")

    df.to_csv(PROC / "dictrank_processed.csv", index=False)

    # ---------------------------------------------------- 시간분할
    X = featurize(df["smiles"].tolist())
    y = df["y"].values
    tr = df["approval_year"].values < CUTOFF
    te = ~tr
    log(f"\n학습군(~{CUTOFF-1}): N={tr.sum()}, 양성 {y[tr].mean():.1%}")
    log(f"검증군({CUTOFF}~): N={te.sum()}, 양성 {y[te].mean():.1%}")

    if te.sum() < 30 or len(np.unique(y[te])) < 2:
        log("\n검증군 표본 부족 - 분석 중단")
        (RES / "dictrank_results.txt").write_text("\n".join(log_lines), encoding="utf-8")
        return

    m = make_model().fit(X[tr], y[tr])
    prob = m.predict_proba(X[te])[:, 1]
    yte = y[te]
    auc_t = roc_auc_score(yte, prob)
    pred = (prob >= 0.5).astype(int)

    log("\n--- 시간분할 결과 ---")
    log(f"  AUROC  {auc_t:.4f}")
    log(f"  AUPRC  {average_precision_score(yte, prob):.4f}")
    log(f"  BalAcc {balanced_accuracy_score(yte, pred):.4f}")
    log(f"  MCC    {matthews_corrcoef(yte, pred):.4f}")

    # 순열 검정 — 계산은 src/permutation_test.py 가 단일 출처로 담당한다.
    log("\n--- 순열 검정 (src/permutation_test.py 결과 인용) ---")
    pt_path = RES / "permutation_test.csv"
    perm = None
    if pt_path.exists():
        _pt = pd.read_csv(pt_path)
        _r = _pt[_pt.dataset == "DICTrank"]
        if len(_r):
            perm = _r.iloc[0]
    if perm is None:
        log("  (아직 없음) python src/permutation_test.py 를 실행하면 채워진다.")
    else:
        log(f"  무작위 분할 {int(perm.n_perm_valid)}회 (유효)")
        log(f"  무작위 평균 AUROC {perm.null_mean:.4f} (SD {perm.null_sd:.4f})")
        log(f"  ΔAUROC {perm.null_mean - auc_t:+.4f}")
        log(
            f"  p (Phipson-Smyth) = {perm.p_permutation:.4f}  "
            f"{'*유의*' if perm.p_permutation < 0.05 else '(유의하지 않음)'}"
        )
        if abs(float(perm.auroc_temporal) - auc_t) > 5e-4:
            log(
                f"  [주의] permutation_test.csv 의 시간분할 AUROC "
                f"{perm.auroc_temporal:.4f} 와 여기서 계산한 {auc_t:.4f} 가 "
                f"다릅니다. 두 스크립트가 같은 CSV 를 보고 있는지 확인하세요."
            )

    # ATC 층화
    # ATC 층화는 src/atc_kegg.py 로 옮겼다. 두 데이터셋을 같은 KEGG 매핑과
    # 같은 절차로 돌려야 비교가 되고, 이 스크립트는 재실행 목록에서
    # 빠져 있어 여기 두면 값이 늘 뒤처진다.
    log("\n--- ATC 층화 ---")
    log("  src/atc_kegg.py 로 이동 (results/atc_tests.csv 참고)")

    log("\n" + "=" * 62)
    log("DILIrank 대비 요약")
    log("=" * 62)
    # 예전에는 DILIrank 쪽 숫자를 여기 손으로 적어 두었다. 분석을 다시
    # 돌리면 틀린 값이 남으므로 결과 파일에서 읽는다.
    if pt_path.exists():
        _pt = pd.read_csv(pt_path)
        for _ds in ("DILIrank", "DICTrank"):
            _r = _pt[_pt.dataset == _ds]
            if not len(_r):
                log(f"  [{_ds}]  permutation_test.csv 에 행 없음")
                continue
            _r = _r.iloc[0]
            log(
                f"  [{_ds}]  무작위 {_r.null_mean:.3f} / "
                f"시간분할 {_r.auroc_temporal:.3f} / "
                f"Δ {_r.delta_auroc:+.3f} / p={_r.p_permutation:.4f}"
            )
    else:
        log("  permutation_test.csv 가 아직 없어 비교표를 만들 수 없습니다.")
        log("  python src/permutation_test.py 를 실행하세요.")

    (RES / "dictrank_results.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\n저장: {RES/'dictrank_results.txt'}")


if __name__ == "__main__":
    main()
