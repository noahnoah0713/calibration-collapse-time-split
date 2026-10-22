# -*- coding: utf-8 -*-
"""하위집단 분석을 보정 지표로 다시 한다 — §14 가 AUROC 뿐이다.

왜 필요한가
----------
이 논문의 주장은 "무너진 것은 판별력이 아니라 보정이다" 다. 그런데 하위집단
분석(§14.1 ATC 계열, §14.2 구조 층화)은 **AUROC 로만** 보고하고 있다.
심사자가 바로 지적할 자리다 — 논지가 보정인데 하위집단은 판별력으로 재면,
"보정 붕괴가 특정 치료군이나 특정 구조군에 몰려 있는가" 라는 질문에 답이
없다.

§21(적용범위)은 이미 Tanimoto 3분위와 스캐폴드 신규/기존에 대해 **대조군
있는 ECE 배율**을 냈다. 그 기계를 ATC 계열과 구조 층에 그대로 적용한다.

설계 — §21 과 동일하게
--------------------
각 층에서

    관측   = 시간분할 모델(2010년 이전 학습)의 그 층 ECE
    대조군 = 무작위 분할로 **같은 크기·같은 양성률** 평가집합을 뽑아
             평가, R회 반복 → ECE 배율과 Phipson-Smyth p
    구간   = 층 내부 부트스트랩으로 배율 95% 구간
             (대조군 재표집 변동은 담지 않으므로 실제보다 좁다)

층 정의
-----
    ATC      1단계 계열 (atc_audit.csv 의 atc1), 그리고 L 대 그외
    구조     §14.2 와 같은 네 기준의 참/거짓
             MW>500 · 방향환>=4 · N>=6 · logP>=3

표본부족 기준은 §21·§17.2 와 같다 — n < 40 또는 소수 클래스 < 10.
**DILIrank 검증군의 ATC 계열은 가장 큰 것이 32개뿐이므로 거의 전부
표본부족으로 빠진다.** 그 사실 자체가 한계 8번이 적은 내용이고, 이 절은
그것을 보정 지표에서도 확인한다.

읽는 법
------
배율이 1 을 넘고 구간이 1 을 제외하는 층이 **특정 치료군·구조군에 몰려
있으면** 처방이 생긴다(그 군에는 확률을 쓰지 않는다). 전 층에 고르게
퍼져 있으면 붕괴는 하위집단 현상이 아니다.

사용법
------
    python src/subgroup_calibration.py            # 대조군 30회
    python src/subgroup_calibration.py --reps 5

출력
----
    results/subgroup_calibration.csv
    results/subgroup_calibration_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

MIN_N, MIN_MINOR = 40, 10
N_BOOT = 2000


def phipson_smyth(r, n):
    return (1.0 + r) / (1.0 + n)


def metrics(y, p):
    d = S.decompose(y, p)
    out = {
        "ece": S.ece(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
        "mean_pred": float(np.mean(p)),
        "obs_rate": float(np.mean(y)),
        "mean_resid": float(np.mean(np.asarray(y) - np.asarray(p))),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    try:
        out["slope"] = S.calib_slope(y, p)
    except Exception:
        out["slope"] = float("nan")
    return out


def boot_ratio(y, p, ctrl, seed=S.SEED):
    """§21 의 boot_ratio 와 같은 방식."""
    if ctrl != ctrl or ctrl == 0:
        return float("nan"), float("nan")
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    v = []
    for _ in range(N_BOOT):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p[i]) / ctrl)
    if not v:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def struct_feats(smi):
    """§14.2(atc_kegg.py)와 같은 정의."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {k: False for k in ("MW>500", "방향환>=4", "N>=6", "logP>=3")}
    return {
        "MW>500": Descriptors.MolWt(m) > 500,
        "방향환>=4": rdMolDescriptors.CalcNumAromaticRings(m) >= 4,
        "N>=6": sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N") >= 6,
        "logP>=3": Descriptors.MolLogP(m) >= 3,
    }


def atc_map(name):
    """atc_audit.csv 에서 화합물명 → ATC 1단계. 100% 조인된다(10-20 확인)."""
    p = os.path.join(RES, "atc_audit.csv")
    if not os.path.exists(p):
        return {}
    a = pd.read_csv(p)
    a = a[a.dataset == name]
    return dict(zip(a.compound.astype(str), a.atc1))


def enough(ys):
    if len(ys) < MIN_N:
        return False
    return int(min((ys == 1).sum(), (ys == 0).sum())) >= MIN_MINOR


def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_m, te_m = S.temporal_masks(df)
    pre, post = np.where(tr_m)[0], np.where(te_m)[0]

    m = S.make_model().fit(X[pre], y[pre])
    p_post = m.predict_proba(X[post])[:, 1]
    y_post = y[post]

    amap = atc_map(name)
    atc1 = np.array(
        [amap.get(str(v), None) for v in df["CompoundName"].astype(str).values[post]], dtype=object
    )
    sf = [struct_feats(s) for s in df["smiles"].values[post]]

    log("=" * 126)
    log(
        "%s   학습 %d / 검증 %d   전체 ECE %.4f   ATC 확보 %d/%d"
        % (
            name,
            len(pre),
            len(post),
            S.ece(y_post, p_post),
            int(sum(v is not None and v == v for v in atc1)),
            len(post),
        )
    )
    log("=" * 126)

    strata = []
    for code in sorted(set(v for v in atc1 if isinstance(v, str))):
        strata.append(("ATC %s" % code, atc1 == code))
    strata.append(("ATC L 대 그외: L", atc1 == "L"))
    strata.append(
        ("ATC L 대 그외: 그외", np.array([isinstance(v, str) and v != "L" for v in atc1]))
    )
    for k in ("MW>500", "방향환>=4", "N>=6", "logP>=3"):
        v = np.array([d[k] for d in sf])
        strata.append(("구조 %s 참" % k, v))
        strata.append(("구조 %s 거짓" % k, ~v))

    log(
        "%-22s %6s %8s %8s %8s %9s %9s %8s %-16s %9s %9s"
        % (
            "층",
            "n",
            "양성률",
            "AUROC",
            "ECE",
            "평균잔차",
            "대조ECE",
            "배율",
            "배율 95%CI",
            "유효반복",
            "p",
        )
    )
    log("-" * 126)
    rows = []
    for tag, mask in strata:
        idx = np.where(mask)[0]
        ys, ps = y_post[idx], p_post[idx]
        if not enough(ys):
            log(
                "%-22s %6d  표본부족 (소수클래스 %d) — 계산하지 않음"
                % (tag, len(ys), int(min((ys == 1).sum(), (ys == 0).sum())) if len(ys) else 0)
            )
            rows.append(
                dict(
                    dataset=name,
                    stratum=tag,
                    n=len(ys),
                    pos_rate=float(ys.mean()) if len(ys) else np.nan,
                    underpowered=True,
                )
            )
            continue
        obs = metrics(ys, ps)
        n_s, r_s = len(ys), float(ys.mean())
        E = []
        for i in range(reps):
            r = np.random.default_rng(S.SEED + 41000 + i)
            ti = r.permutation(len(y))[:n_s]
            ti = S.subsample_to_rate(y, ti, r_s, r)
            ri = np.setdiff1d(np.arange(len(y)), ti)
            if len(np.unique(y[ti])) < 2 or len(ri) < 100:
                continue
            mm = S.make_model().fit(X[ri], y[ri])
            E.append(S.ece(y[ti], mm.predict_proba(X[ti])[:, 1]))
        if E:
            E = np.array(E)
            n_ge = int((E >= obs["ece"]).sum())
            pv = phipson_smyth(n_ge, len(E))
            ce = float(E.mean())
            ratio = obs["ece"] / ce
        else:
            pv = ratio = ce = float("nan")
            n_ge = 0
        rlo, rhi = boot_ratio(ys, ps, ce)
        log(
            "%-22s %6d %8.4f %8.4f %8.4f %+9.4f %9.4f %8.2f [%6.2f, %6.2f] %9s %9.4f"
            % (
                tag,
                n_s,
                r_s,
                obs["auroc"],
                obs["ece"],
                obs["mean_resid"],
                ce,
                ratio,
                rlo,
                rhi,
                "%d/%d" % (n_ge, len(E) if len(E) else 0),
                pv,
            )
        )
        rows.append(
            dict(
                dataset=name,
                stratum=tag,
                n=n_s,
                pos_rate=r_s,
                ctrl_ece=ce,
                ece_ratio=ratio,
                ratio_lo=rlo,
                ratio_hi=rhi,
                n_rep_valid=len(E) if len(E) else 0,
                n_ge=n_ge,
                p_phipson_smyth=pv,
                underpowered=False,
                **obs
            )
        )
    log("")
    return rows


def judge(d):
    log("=" * 126)
    log("판정")
    log("=" * 126)
    log("[읽는 법] 배율이 1 을 넘고 구간이 1 을 제외하는 층이 특정 치료군·구조군에")
    log("          몰려 있으면 처방이 생긴다. 고르게 퍼져 있으면 붕괴는 하위집단")
    log("          현상이 아니다. AUROC 순위와 ECE 순위가 다르면 §14 를 AUROC 로만")
    log("          읽어서는 안 된다는 뜻이다.")
    log("")
    ok = d[~d.underpowered.astype(bool)]
    up = d[d.underpowered.astype(bool)]
    log("  계산한 층 %d개 / 표본부족으로 뺀 층 %d개" % (len(ok), len(up)))
    for ds in d.dataset.unique():
        g = ok[ok.dataset == ds]
        gu = up[up.dataset == ds]
        if not len(g):
            log("  %s — 계산 가능한 층이 없다 (전부 표본부족). 한계 8번 그대로다." % ds)
            continue
        ex = g[(g.ratio_lo > 1) | (g.ratio_hi < 1)]
        log("")
        log("  %s — 층 %d개 계산, %d개 표본부족" % (ds, len(g), len(gu)))
        log(
            "     배율 구간이 1 을 제외하는 층 %d/%d%s"
            % (
                len(ex),
                len(g),
                (
                    (
                        " — %s"
                        % ", ".join("%s(%.2f)" % (r.stratum, r.ece_ratio) for _, r in ex.iterrows())
                    )
                    if len(ex)
                    else ""
                ),
            )
        )
        _a = g.loc[g.auroc.idxmin()]
        _e = g.loc[g.ece_ratio.idxmax()]
        log("     AUROC 가 가장 낮은 층: %s (%.4f)" % (_a.stratum, _a.auroc))
        log("     ECE 배율이 가장 큰 층: %s (%.2f)" % (_e.stratum, _e.ece_ratio))
        log(
            "     → %s"
            % (
                "같은 층이다 — AUROC 로 읽어도 같은 결론이 나온다"
                if _a.stratum == _e.stratum
                else "**다른 층이다 — §14 를 AUROC 로만 읽으면 보정이 가장 많이 "
                "무너진 층을 놓친다.**"
            )
        )
    log("")
    log("저장: %s" % os.path.join(RES, "subgroup_calibration.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    log("하위집단 분석을 보정 지표로 다시 한다 — §14 가 AUROC 뿐이다")
    log("층마다 크기·양성률을 맞춘 무작위 대조군 %d회. 시드 %d" % (a.reps, S.SEED))
    log("표본부족 기준 n < %d 또는 소수 클래스 < %d" % (MIN_N, MIN_MINOR))
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.reps)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "subgroup_calibration.csv"), index=False)
    judge(d)
    log.save(Path(RES) / "subgroup_calibration_summary.txt")


if __name__ == "__main__":
    main()
