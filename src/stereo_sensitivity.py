"""
[보완 5] 구조 동일성 판정 키 민감도 — InChIKey 14자리 vs 27자리 전체

문제:
  apply_label_conflict_filter.py 와 상류 중복 제거가 InChIKey 앞 14자리를
  구조 동일성 키로 쓴다. 앞 14자리는 골격 해시라서 입체이성질체가
  같은 분자로 접힌다. label_audit_summary.txt 는 이렇게 접힌 것이
  DILIrank 21그룹/42행, DICTrank 91그룹/203행이라고 보고하고,
  선택지 (c) 로 "27자리 전체 InChIKey 로 중복 제거" 를 제안한다.
  독성은 입체특이적일 수 있으므로 이 선택에 민감도가 필요하다.

이 스크립트:
  1) 14자리로 접힌 그룹 중 27자리 전체 키가 실제로 다른 것이 몇 건인지 센다
     (= 입체이성질체 때문에 접힌 건수)
  2) 27자리 기준으로 다시 중복 제거·라벨충돌 필터를 적용한 분석집합을 만들고
     시간분할 핵심 지표(AUROC / ECE / Brier 분해)를 재계산해 14자리 결과와 비교
  데이터 파일은 건드리지 않는다. 메모리 안에서만 재구성한다.

  [한계 — 반드시 읽을 것]
  출발점인 *_preconflict.csv 는 이미 상류에서 14자리 기준 중복 제거가
  끝난 파일이다. 따라서 이 스크립트가 되돌릴 수 있는 것은 그 뒤에 남은
  잔여 충돌뿐이고, 상류에서 이미 사라진 행(DILIrank 21행, DICTrank 112행)은
  복원되지 않는다. 완전한 27자리 민감도를 보려면 중복 제거 이전 단계
  (표준화 직후, DILIrank 882행 / DICTrank 1057행)에서 파이프라인을
  다시 타야 한다. 여기 결과는 '하한' 으로 읽을 것.

사용법:
    python src/stereo_sensitivity.py [--reps N]

출력:
    results/stereo_sensitivity.csv
    results/stereo_sensitivity_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)
# 필터 적용 전 파일: 14자리 중복 제거는 이미 상류에서 끝났으므로,
# 접힘을 되돌려 보려면 preconflict(라벨충돌 필터 적용 전) 를 출발점으로 쓴다.
SOURCES = [
    ("DILIrank", "dilirank2_with_year_preconflict.csv", "dilirank2_with_year.csv"),
    ("DICTrank", "dictrank_processed_preconflict.csv", "dictrank_processed.csv"),
]
log = S.Log()
_norm = rdMolStandardize.Normalizer()
_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()


def standardize(smi):
    """baseline.py 의 standardize() 와 같은 순서."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = _norm.normalize(m)
        m = _lfc.choose(m)
        m = _unc.uncharge(m)
        return m
    except Exception:
        return None


def keys(smi):
    m = standardize(smi)
    if m is None:
        return None, None
    try:
        k = Chem.MolToInchiKey(m)
    except Exception:
        return None, None
    return k[:14], k


def metrics(y, p):
    d = S.decompose(y, p)
    return dict(
        auroc=roc_auc_score(y, p),
        ece=S.ece(y, p),
        brier=brier_score_loss(y, p),
        reliability=d["reliability"],
        resolution=d["resolution"],
        uncertainty=d["uncertainty"],
    )


def analyse(df, tag, n_rep):
    """분석집합 하나에 대해 시간분할 지표와 무작위(크기+유병률맞춤) 대조군."""
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr, te = S.temporal_masks(df)
    if tr.sum() < 100 or te.sum() < 40 or len(np.unique(y[te])) < 2:
        log(f"    {tag}: 표본 부족 (학습 {tr.sum()}, 검증 {te.sum()})")
        return None
    n_test, rate_t = int(te.sum()), float(y[te].mean())
    m = S.make_model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    t = metrics(y[te], p)

    acc = {k: [] for k in ("ece", "reliability", "resolution", "auroc")}
    for i in range(n_rep):
        r = np.random.default_rng(S.SEED + i)
        idx = r.permutation(len(y))
        ti, ri = idx[:n_test], idx[n_test:]
        ti = S.subsample_to_rate(y, ti, rate_t, r)
        if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
            continue
        mm = S.make_model().fit(X[ri], y[ri])
        pp = mm.predict_proba(X[ti])[:, 1]
        d = S.decompose(y[ti], pp)
        acc["ece"].append(S.ece(y[ti], pp))
        acc["auroc"].append(roc_auc_score(y[ti], pp))
        acc["reliability"].append(d["reliability"])
        acc["resolution"].append(d["resolution"])
    mean = {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}
    dR = t["reliability"] - mean["reliability"]
    dS = mean["resolution"] - t["resolution"]
    return dict(
        variant=tag,
        n=len(df),
        n_train=int(tr.sum()),
        n_test=n_test,
        pos_rate_test=rate_t,
        **t,
        ece_random_pm=mean["ece"],
        auroc_random=mean["auroc"],
        ece_ratio_pm=t["ece"] / mean["ece"] if mean["ece"] else np.nan,
        d_auroc=mean["auroc"] - t["auroc"],
        share_reliability=100 * dR / (dR + dS) if (dR + dS) else np.nan,
    )


def run(name, pre_path, cur_path, n_rep):
    pre = S.load(pre_path)
    cur = S.load(cur_path)
    log("=" * 84)
    log(f"{name}   필터 전 {len(pre)}행 / 현재 분석집합 {len(cur)}행")
    log("=" * 84)

    k = pre["smiles"].apply(keys)
    pre = pre.assign(k14=[a for a, _ in k], k27=[b for _, b in k])
    pre = pre.dropna(subset=["k14", "k27"]).reset_index(drop=True)

    g14 = pre.groupby("k14")
    folded = g14.filter(lambda d: len(d) > 1)
    n_grp14 = folded["k14"].nunique()
    stereo = folded.groupby("k14").filter(lambda d: d["k27"].nunique() > 1)
    n_grp_stereo = stereo["k14"].nunique()
    log(f"\n[1] 14자리로 같은 키를 가진 그룹: {n_grp14}개 / {len(folded)}행")
    log(
        f"    그중 27자리가 서로 다른 그룹(입체이성질체 등): " f"{n_grp_stereo}개 / {len(stereo)}행"
    )
    if n_grp_stereo:
        ex = (
            stereo.groupby("k14")
            .agg(
                names=("CompoundName", lambda s: " | ".join(map(str, s))),
                ys=("y", lambda s: "/".join(map(str, s))),
            )
            .head(8)
        )
        for kk, r in ex.iterrows():
            log(f"      {kk}  y={r.ys}  {r.names[:88]}")
        if n_grp_stereo > 8:
            log(f"      ... 외 {n_grp_stereo-8}개")

    rows = []
    # 현재(14자리) 분석집합
    r = analyse(cur, "14자리 (현재 분석집합)", n_rep)
    if r:
        rows.append(dict(dataset=name, **r))

    # 27자리 기준 재구성: 27자리로 중복 제거 + 27자리 기준 라벨충돌 필터
    d27 = pre.copy()
    conflict27 = d27.groupby("k27")["y"].nunique() > 1
    bad27 = set(conflict27[conflict27].index)
    d27 = d27[~d27["k27"].isin(bad27)]
    d27 = d27.drop_duplicates(subset="k27", keep="first").reset_index(drop=True)
    log(
        f"\n[2] 27자리 기준 재구성: 라벨충돌 그룹 {len(bad27)}개 제외 후 "
        f"중복 제거 → {len(d27)}행  (14자리 기준 {len(cur)}행 대비 "
        f"{len(d27)-len(cur):+d})"
    )
    r = analyse(d27, "27자리 (입체이성질체 분리)", n_rep)
    if r:
        rows.append(dict(dataset=name, **r))

    out = pd.DataFrame(rows)
    if len(out) == 2:
        a, b = out.iloc[0], out.iloc[1]
        log(f"\n[3] 비교")
        log(f"    {'지표':18s} {'14자리':>10s} {'27자리':>10s} {'차이':>10s}")
        for f, lab in [
            ("n", "분석집합 n"),
            ("n_test", "검증군 n"),
            ("auroc", "시간분할 AUROC"),
            ("ece", "시간분할 ECE"),
            ("ece_ratio_pm", "ECE 배율(유병률맞춤)"),
            ("d_auroc", "ΔAUROC"),
            ("share_reliability", "보정 기여율 %"),
        ]:
            log(f"    {lab:18s} {a[f]:10.4f} {b[f]:10.4f} {b[f]-a[f]:+10.4f}")
    log("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    log("구조 동일성 키 민감도 — InChIKey 14자리 vs 27자리")
    log(f"무작위 대조군 {a.reps}회(크기+유병률 맞춤), seed {S.SEED}, " f"절단 {S.CUTOFF}\n")
    out = []
    for nm, pre, cur in SOURCES:
        pp, cp = PROC / pre, PROC / cur
        if pp.exists() and cp.exists():
            out.append(run(nm, pp, cp, a.reps))
        else:
            log(f"{nm}: 파일 없음 - 건너뜀 ({pre} / {cur})")
    out = [o for o in out if o is not None and len(o)]
    if not out:
        log("결과 없음")
        return
    df = pd.concat(out, ignore_index=True)
    df.to_csv(RES / "stereo_sensitivity.csv", index=False)
    log("=" * 84)
    log("판정")
    log("=" * 84)
    log("27자리로 바꿨을 때 ECE 배율과 보정 기여율이 거의 그대로면,")
    log("입체이성질체 병합이 결론을 만들어낸 것이 아니다.")
    log("크게 달라지면 본문에 두 값을 병기하고 14자리를 택한 이유를 밝혀야 한다.")
    log.save(RES / "stereo_sensitivity_summary.txt")
    print(f"\n저장: {RES/'stereo_sensitivity.csv'}")
    print(f"      {RES/'stereo_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
