"""적용범위(applicability domain) — 미설명분이 '낯선 분자' 에 몰려 있는가.

왜 필요한가 (Major 3 의 잔여분)
-------------------------------
§16·§18·§19 가 유병률·크기·개념 이동으로 ECE 초과분의 상당 부분을 설명했다.
남은 질문은 어디에 몰려 있는가 다. 심사자가 반드시 제안할 가설이 하나
있다 — "2010년 이후 분자가 학습군과 화학적으로 낯설어서 그런 것 아닌가."
맞다면 처방이 생긴다: 낯선 분자에는 예측을 내지 않는다(AD 필터).

§14 가 이미 최근접 Tanimoto 로 층화한 ECE 를 냈다. 그런데 그 표는 결론에
쓸 수 없다. 두 가지 때문이다.

  (1) 구간별 무작위 대조군이 없다 (§14 요약문이 자진 신고한 그대로).
  (2) 구간마다 양성률이 다르다. DILIrank 에서 35.9% / 17.4% / 24.1% 다.
      §19 는 ECE 가 학습·검증 유병률 격차의 함수임을 보였으므로, 구간 간
      ECE 차이는 대부분 유병률 교란일 수 있다.

그래서 여기서는 구간마다 크기와 양성률을 맞춘 무작위 대조군을 붙인다.
그래야 "이 구간이 원래 어려운 구간" 과 "시간분할이 이 구간을 망쳤다" 가
갈린다. §16 사다리를 구간 단위로 적용하는 셈이다.

설계
----
  층화 1 : 학습군 최근접 Tanimoto 3분위 (하/중/상)
  층화 2 : Bemis-Murcko 스캐폴드가 학습군에 있었는가 / 처음 보는가
           — 이건 지금까지 어디에서도 계산하지 않았다.

  각 층에서:
    관측    = 시간분할 모델(pre-2010 학습)의 그 층 ECE
    대조군  = 무작위 분할로 같은 크기·같은 양성률 검증군을 뽑아 평가,
              R회 반복 → ECE 배율과 Phipson-Smyth p

  표본부족 기준은 n < 40 또는 소수 클래스 < 10 이다 (§17.2 와 같은 기준).

추가 — AD 필터의 실제 값어치
---------------------------
"가장 낯선 x% 에는 예측하지 않는다" 를 x = 0~50% 로 돌려 보고, 남은 분자의
ECE·AUROC 와 적용률(coverage)을 함께 적는다. 같은 수를 무작위로 버리는
대조군을 나란히 둔다. 둘이 같으면 AD 필터는 '어려운 분자를 골라낸 것' 이
아니라 '그냥 표본을 줄인 것' 이다. 이 부분은 재적합이 없다.

예상
----
§14 의 원 수치는 상위 50% 유사도 쪽이 더 나빴다(0.4700 대 0.4227).
그래서 음성 결과가 나올 가능성이 높다. 음성이어도 쓸모가 있다 — 심사자가
제안할 AD 가설을 대조군 있는 설계로 닫고, §19 의 유병률 교란이 §14 표를
어떻게 왜곡했는지 보여 준다.

사용법
------
    python src/applicability_domain.py              # 대조군 30회
    python src/applicability_domain.py --reps 5     # 짧게 점검

출력
----
    results/applicability_domain.csv
    results/applicability_domain_filter.csv
    results/applicability_domain_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

MIN_N, MIN_MINOR = 40, 10  # §17.2 와 같은 표본부족 기준
DROPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_BOOT = 2000  # 배율 신뢰구간용 (재적합 없음)


def boot_ratio(y, p, ctrl, n_boot=N_BOOT, seed=S.SEED):
    """층 내부 재표집으로 ECE 배율의 95% 구간.

    대조군 평균은 고정하고 관측 쪽만 재표집한다. 즉 이 구간은
    검증군 표본오차만 담고 대조군 재표집 변동은 담지 않는다 —
    따라서 실제보다 좁다. 본문에 그대로 적어야 한다.
    """
    boot_ratio.auroc = None  # 이전 층 값이 남지 않게 먼저 비운다
    if ctrl != ctrl or ctrl == 0:
        return float("nan"), float("nan"), None
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    v, au = [], []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(S.ece(y[i], p[i]) / ctrl)
        au.append(float(roc_auc_score(y[i], p[i])))
    if not v:
        return float("nan"), float("nan"), None
    a = np.sort(np.array(v))
    boot_ratio.auroc = np.array(au)  # 같은 재표집의 AUROC 분포
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), a


def phipson_smyth(r, n):
    return (1.0 + r) / (1.0 + n)


def metrics(y, p):
    d = S.decompose(y, p)
    out = {
        "ece": S.ece(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
        "mean_resid": float(np.mean(np.asarray(y) - np.asarray(p))),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    try:
        out["slope"] = S.calib_slope(y, p)
    except Exception:
        out["slope"] = float("nan")
    return out


def nn_similarity(smiles_tr, smiles_te):
    """§14(calibration_confounders.py)와 같은 구현 — 값 일치를 유지한다."""

    def fp(s):
        return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)

    ftr = [fp(s) for s in smiles_tr]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), ftr)) for s in smiles_te])


def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def enough(y_sub):
    if len(y_sub) < MIN_N:
        return False
    k = int(min((y_sub == 1).sum(), (y_sub == 0).sum()))
    return k >= MIN_MINOR


def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    smi = df["smiles"].values
    tr_m, te_m = S.temporal_masks(df)
    pre, post = np.where(tr_m)[0], np.where(te_m)[0]

    # 시간분할 모델 1회 적합 — 모든 층이 이 예측을 공유한다
    m = S.make_model().fit(X[pre], y[pre])
    p_post = m.predict_proba(X[post])[:, 1]
    y_post = y[post]

    sim = nn_similarity(smi[pre].tolist(), smi[post].tolist())
    sc_tr = set(scaffold_of(s) for s in smi[pre])
    seen = np.array([scaffold_of(s) in sc_tr for s in smi[post]])

    log("=" * 108)
    log("%s   학습 pre-2010 %d개 / 검증 post-2010 %d개" % (name, len(pre), len(post)))
    log("  최근접 Tanimoto 중앙값 %.4f 평균 %.4f" % (np.median(sim), sim.mean()))
    log(
        "  스캐폴드가 학습군에 있던 분자 %d개 (%.1f%%) / 처음 보는 분자 %d개 (%.1f%%)"
        % (int(seen.sum()), 100.0 * seen.mean(), int((~seen).sum()), 100.0 * (~seen).mean())
    )
    log("  전체 ECE %.4f (§8 값)" % S.ece(y_post, p_post))
    log("=" * 108)

    # 3분위 경계
    q1, q2 = np.quantile(sim, [1.0 / 3, 2.0 / 3])
    strata = [
        ("Tanimoto 하위 1/3", sim <= q1),
        ("Tanimoto 중위 1/3", (sim > q1) & (sim <= q2)),
        ("Tanimoto 상위 1/3", sim > q2),
        ("스캐폴드 기존", seen),
        ("스캐폴드 신규", ~seen),
    ]

    log(
        "%-20s %6s %8s %8s %8s %8s %9s %8s %-14s %8s %9s"
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
    log("-" * 124)
    rows, boot, bootA = [], {}, {}
    for tag, mask in strata:
        idx = np.where(mask)[0]
        ys, ps = y_post[idx], p_post[idx]
        if not enough(ys):
            log(
                "%-20s %6d  표본부족 (소수클래스 %d) — 계산하지 않음"
                % (tag, len(ys), int(min((ys == 1).sum(), (ys == 0).sum())))
            )
            rows.append(
                {
                    "dataset": name,
                    "stratum": tag,
                    "n": len(ys),
                    "pos_rate": float(ys.mean()) if len(ys) else np.nan,
                    "underpowered": True,
                }
            )
            continue
        obs = metrics(ys, ps)
        n_s, r_s = len(ys), float(ys.mean())

        # 크기·양성률 맞춤 무작위 대조군
        E = []
        for i in range(reps):
            r = np.random.default_rng(S.SEED + 30000 + i)
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
            ratio = obs["ece"] / float(E.mean())
            ce = float(E.mean())
        else:
            pv = ratio = ce = float("nan")
            n_ge = 0
        rlo, rhi, rdist = boot_ratio(ys, ps, ce)
        boot[tag] = rdist
        bootA[tag] = getattr(boot_ratio, "auroc", None)
        log(
            "%-20s %6d %8.4f %8.4f %8.4f %+8.4f %9.4f %8.2f [%.2f, %.2f] %8s %9.4f"
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
            {
                "dataset": name,
                "stratum": tag,
                "n": n_s,
                "pos_rate": r_s,
                "ctrl_ece": ce,
                "ece_ratio": ratio,
                "ratio_lo": rlo,
                "ratio_hi": rhi,
                "n_rep_valid": len(E) if len(E) else 0,
                "n_ge": n_ge,
                "p_phipson_smyth": pv,
                "underpowered": False,
                **obs,
            }
        )
    lo_d, hi_d = boot.get("Tanimoto 하위 1/3"), boot.get("Tanimoto 상위 1/3")
    grad = (np.nan, np.nan, np.nan)
    if lo_d is not None and hi_d is not None:
        k = min(len(lo_d), len(hi_d))
        rr = np.random.default_rng(S.SEED + 777)
        d = lo_d[rr.permutation(len(lo_d))[:k]] - hi_d[rr.permutation(len(hi_d))[:k]]
        d = np.sort(d)
        grad = (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        _incl0 = grad[1] <= 0.0 <= grad[2]
        log("")
        log("  유사도 기울기 — 보정과 판별이 같이 움직이는가")
        log("    [보정] 하위 1/3 배율 − 상위 1/3 배율")
        log(
            "      평균 %+.2f  95%% CI [%+.2f, %+.2f]  → %s"
            % (
                grad[0],
                grad[1],
                grad[2],
                (
                    "0 을 포함한다 — 기울기를 주장할 수 없다"
                    if _incl0
                    else "0 을 포함하지 않는다 — 기울기가 있다"
                ),
            )
        )
        la, ha = bootA.get("Tanimoto 하위 1/3"), bootA.get("Tanimoto 상위 1/3")
        if la is not None and ha is not None:
            k2 = min(len(la), len(ha))
            r2 = np.random.default_rng(S.SEED + 888)
            da = ha[r2.permutation(len(ha))[:k2]] - la[r2.permutation(len(la))[:k2]]
            da = np.sort(da)
            agrad = (
                float(np.mean(da)),
                float(np.percentile(da, 2.5)),
                float(np.percentile(da, 97.5)),
            )
            _ai = agrad[1] <= 0.0 <= agrad[2]
            log("    [판별] 상위 1/3 AUROC − 하위 1/3 AUROC")
            log(
                "      평균 %+.4f  95%% CI [%+.4f, %+.4f]  → %s"
                % (
                    agrad[0],
                    agrad[1],
                    agrad[2],
                    "0 을 포함한다" if _ai else "0 을 포함하지 않는다 — 차이가 있다",
                )
            )
            # "한쪽은 유의, 한쪽은 비유의" 로 두 효과가 다르다고 결론내는 것은
            # 흔한 오류다 (Gelman & Stern 2006). 두 구간은 눈금도 다르므로
            # 직접 비교가 불가능하다. 그래서 관측된 바만 적고, 보정 쪽
            # 구간이 무엇까지 허용하는지를 함께 밝힌다.
            if (not _ai) and _incl0:
                _wide = max(abs(grad[1]), abs(grad[2]))
                log("    → 판별 기울기는 0 을 벗어나고, 보정 기울기는 벗어나지 않는다.")
                log("       다만 이것만으로 '두 효과가 다르다' 고 결론내면 안 된다 —")
                log("       유의/비유의의 대비는 차이의 검정이 아니고(Gelman & Stern),")
                log("       두 양은 눈금이 달라 직접 비교도 불가능하다.")
                log("       보정 쪽 구간은 최대 %.2f 크기의 기울기까지 허용하므로" % _wide)
                log("       '보정은 유사도와 무관하다' 는 **입증되지 않았다.**")
                log("       말할 수 있는 것: 판별력은 유사도에 따라 변하고,")
                log("       보정 배율에서는 같은 표본으로 변화를 검출하지 못했다.")
            rows.append(
                {
                    "dataset": name,
                    "stratum": "판별 기울기(상위−하위 AUROC)",
                    "n": np.nan,
                    "ece_ratio": agrad[0],
                    "ratio_lo": agrad[1],
                    "ratio_hi": agrad[2],
                    "underpowered": False,
                }
            )
        log("    [주의] 이 구간은 검증군 재표집만 반영한다. 대조군 재표집")
        log("           변동이 빠져 있으므로 **실제보다 좁다.**")
    rows.append(
        {
            "dataset": name,
            "stratum": "유사도 기울기(하위−상위)",
            "n": np.nan,
            "ece_ratio": grad[0],
            "ratio_lo": grad[1],
            "ratio_hi": grad[2],
            "underpowered": False,
        }
    )
    log("")

    # ---------------- AD 필터 — 재적합 없음 --------------------------
    log("AD 필터 — 가장 낯선 x% 를 버리면 무엇이 회복되는가 (재적합 없음)")
    log(
        "  %-8s %8s %8s %8s %8s %12s %12s"
        % ("버림", "적용률", "남은n", "ECE", "AUROC", "무작위버림ECE", "차이")
    )
    log("  " + "-" * 72)
    order = np.argsort(sim)  # 낯선 것부터
    frows = []
    for x in DROPS:
        k = int(round(x * len(sim)))
        keep = order[k:] if k else order
        ys, ps = y_post[keep], p_post[keep]
        if not enough(ys):
            log("  %-8s 표본부족 — 건너뜀" % ("%.0f%%" % (x * 100)))
            continue
        e_ad, a_ad = S.ece(ys, ps), (
            float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else np.nan
        )
        # 같은 수를 무작위로 버리는 대조군 (예측은 고정, 재적합 없음)
        rr = np.random.default_rng(S.SEED + 40000)
        ev = []
        for _ in range(200):
            kp = rr.permutation(len(sim))[k:] if k else np.arange(len(sim))
            if len(np.unique(y_post[kp])) < 2:
                continue
            ev.append(S.ece(y_post[kp], p_post[kp]))
        e_rand = float(np.mean(ev)) if ev else float("nan")
        log(
            "  %-8s %8.3f %8d %8.4f %8.4f %12.4f %+12.4f"
            % (
                "%.0f%%" % (x * 100),
                len(keep) / len(sim),
                len(keep),
                e_ad,
                a_ad,
                e_rand,
                e_ad - e_rand,
            )
        )
        frows.append(
            {
                "dataset": name,
                "drop_frac": x,
                "coverage": len(keep) / len(sim),
                "n_keep": len(keep),
                "ece_ad": e_ad,
                "auroc_ad": a_ad,
                "ece_random_drop": e_rand,
                "delta": e_ad - e_rand,
            }
        )
    log("")
    log("  [읽는 법] 'ECE' 와 '무작위버림ECE' 가 같으면 AD 필터는 어려운 분자를")
    log("            골라낸 것이 아니라 그냥 표본을 줄인 것이다. 차이가 음수로")
    log("            뚜렷해야 필터가 값어치를 한다.")
    log("")
    return rows, frows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    log("적용범위 — 미설명분이 낯선 분자에 몰려 있는가 (Major 3 잔여분)")
    log("층별 대조군 %d회, 시드 %d. 대조군은 층과 크기·양성률을 맞춘다." % (a.reps, S.SEED))
    log("")
    R, F = [], []
    for name, fn in S.DATASETS:
        rr, ff = run(name, fn, a.reps)
        R += rr
        F += ff
    pd.DataFrame(R).to_csv(
        os.path.join(RES, "applicability_domain.csv"), index=False, encoding="utf-8"
    )
    pd.DataFrame(F).to_csv(
        os.path.join(RES, "applicability_domain_filter.csv"), index=False, encoding="utf-8"
    )

    log("=" * 108)
    log("판정")
    log("=" * 108)
    log("[읽는 법] 층별 **배율** 을 본다. 유사도가 낮은 층일수록 배율이 크면")
    log("          적용범위 가설이 지지된다. 배율이 층마다 비슷하면 시간분할의")
    log("          보정 붕괴는 화학적 낯섦이 아니라 **전역적** 이다 — 그 경우")
    log("          AD 필터는 처방이 되지 못하고, §14 의 구간별 ECE 차이는")
    log("          유병률 교란이었다는 뜻이 된다.")
    log("")
    d_ = pd.DataFrame(R)
    for ds in d_.dataset.unique():
        # 기울기 행은 층이 아니므로 층별 목록에서 뺀다
        g = d_[
            (d_.dataset == ds)
            & (~d_.underpowered)
            & (~d_.stratum.isin(["유사도 기울기(하위−상위)", "판별 기울기(상위−하위 AUROC)"]))
        ]
        if not len(g):
            log("  %s — 전 층 표본부족" % ds)
            continue
        log("  %s" % ds)
        for _, r in g.iterrows():
            log(
                "    %-20s 양성률 %.4f  ECE %.4f  배율 %.2f  p %.4f"
                % (r.stratum, r.pos_rate, r.ece, r.ece_ratio, r.p_phipson_smyth)
            )
        for _lab, _key, _fmt in (
            ("보정 기울기(배율)", "유사도 기울기(하위−상위)", "%+.2f"),
            ("판별 기울기(AUROC)", "판별 기울기(상위−하위 AUROC)", "%+.4f"),
        ):
            gr = d_[(d_.dataset == ds) & (d_.stratum == _key)]
            if not len(gr) or gr.iloc[0].ece_ratio != gr.iloc[0].ece_ratio:
                continue
            r0 = gr.iloc[0]
            inc = r0.ratio_lo <= 0.0 <= r0.ratio_hi
            log(
                ("    → %s " + _fmt + " [" + _fmt + ", " + _fmt + "] — %s")
                % (
                    _lab,
                    r0.ece_ratio,
                    r0.ratio_lo,
                    r0.ratio_hi,
                    "**주장 불가** (0 포함)" if inc else "**차이 있음**",
                )
            )
        log("")
    log("저장: %s" % os.path.join(RES, "applicability_domain.csv"))
    log("      %s" % os.path.join(RES, "applicability_domain_filter.csv"))
    log("      %s" % os.path.join(RES, "applicability_domain_summary.txt"))
    log.save(Path(RES) / "applicability_domain_summary.txt")


if __name__ == "__main__":
    main()
