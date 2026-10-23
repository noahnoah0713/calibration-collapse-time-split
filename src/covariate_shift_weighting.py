"""공변량 이동 보정 — 도메인 분류기로 밀도비 가중을 걸면 되는가.

왜 필요한가 (이름이 불릴 처방이 하나 남았다)
------------------------------------------
이 논문은 처방을 네 가지 시도했다 — 사후 재보정(§10), 사전확률 추정(§10.2),
최신성 가중(§20), 학습 시 클래스 가중(§23). 그런데 분포 이동 문헌에서 가장
먼저 나오는 처방이 빠져 있다: 공변량 이동(covariate shift) 하에서의
중요도 가중 학습(importance-weighted ERM). 저장소 전체를 훑어도
density ratio / propensity / importance weight 를 쓰는 스크립트가 없다
(2026-10-19 확인).

더 공교로운 것은 그 가중을 만들 재료가 이미 §13.1 에 있다는 점이다.
시대 분류기가 구조만으로 "2010년 이후인가" 를 AUC 0.6835 / 0.7917 로
맞힌다. 그 분류기의 확률이 바로 밀도비 추정값이다 —

    w(x) = p_post(x) / p_pre(x)  ∝  P(post|x) / (1 − P(post|x))

심사자가 "도메인 적응을 왜 안 해봤나" 라고 쓰면 반박할 자료가 없다.
이 스크립트가 그 구멍을 메운다.

§20 의 최신성 가중과 무엇이 다른가
---------------------------------
최신성 가중은 승인연도 라는 한 축으로 가중을 준다. 중요도 가중은
분자 구조 전체로 준다 — "검증군에 있을 법한 분자" 에 가중을 몰아준다.
연도가 같아도 구조가 미래 쪽이면 가중이 커진다. 둘은 다른 처방이다.

검증군 라벨을 쓰는가 — 쓰지 않는다
---------------------------------
시대 분류기는 라벨 y 를 전혀 쓰지 않는다. 쓰는 것은 "이 분자가 어느
시대인가" 뿐이고, 그것은 배포 시점에 아는 정보다(점수를 매길 분자는 손에
있다). 그래서 §10 의 D(사전확률 보정)나 §23 의 W1 과 달리 이 처방은
원리적으로 배포 가능하다. 단, 미래 분자들의 구조를 모아 두어야 한다는
조건이 붙는다(transductive). 그 조건을 본문에 적는다.

가중은 공짜가 아니다
-------------------
밀도비는 꼬리가 두껍다. 소수 분자에 가중이 몰리면 유효 표본이 붕괴한다.
그래서 모든 팔에 n_eff = (Σw)²/Σw² 를 함께 찍고, 절단(clipping) 수준을
바꾼 팔을 나란히 둔다. 절단 수준 선택은 §20 과 같은 정직 프로토콜로
학습군 안에서만 한다 — 학습군을 다시 시간으로 쪼개 앞부분으로 적합하고
뒷부분에서 고른다. 검증군은 선택에 쓰지 않는다.

사용법
------
    python src/covariate_shift_weighting.py              # 대조군 50회
    python src/covariate_shift_weighting.py --reps 200   # §8 과 같은 반복 수

출력
----
    results/covariate_shift_weighting.csv
    results/covariate_shift_weighting_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

N_FOLD = 5
INNER_CUT = 2000  # 학습군 내부 분할 연도 (§20 과 같은 사고방식)
CAPS = [None, 20.0, 10.0, 5.0, 2.0]  # 밀도비 상한
TOL = 5e-4


def era_oof_prob(X, era, seed=S.SEED):
    """시대 라벨(0=이전, 1=이후)에 대한 out-of-fold 확률. 라벨 y 는 쓰지 않는다."""
    p = np.zeros(len(era), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=seed)
    for tri, tei in skf.split(X, era):
        m = S.make_model().fit(X[tri], era[tri])
        p[tei] = m.predict_proba(X[tei])[:, 1]
    return p


def dens_ratio(p_post, cap):
    """w = P(post|x)/(1-P(post|x)). cap 이 있으면 그 값에서 자른다."""
    q = np.clip(p_post, S.EPS, 1 - S.EPS)
    w = q / (1.0 - q)
    if cap is not None:
        w = np.minimum(w, cap)
    w = w / w.mean()  # 평균 1 로 정규화 (스케일은 무의미하다)
    return w


def n_eff(w):
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def score(y, p):
    d = S.decompose(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "ece": S.ece(y, p),
        "slope": S.calib_slope(y, p),
        "brier": float(np.mean((np.asarray(y) - p) ** 2)),
        "rel": d["reliability"],
        "res": d["resolution"],
        "mean_pred": float(np.mean(p)),
    }


def run(name, fn, reps, ref_pc, ref_ep):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]
    rate_tr, rate_te = float(y[tr].mean()), float(y[te].mean())

    # ---- 시대 분류기 (라벨 y 미사용) ----
    era = te_m.astype(int)
    p_post = era_oof_prob(X, era)
    era_auc = float(roc_auc_score(era, p_post))

    log("=" * 126)
    log(
        "%s   학습 %d (양성률 %.4f) / 검증 %d (%.4f)   대조군 %d회"
        % (name, len(tr), rate_tr, len(te), rate_te, reps)
    )
    log("  시대 분류기 OOF AUC %.4f" % era_auc)
    if ref_ep is not None:
        q = ref_ep[ref_ep.dataset == name]
        if len(q):
            log(
                "  §13.1 의 시대 AUC %.4f 와의 차 %.4f — 교차검증 분할이 다르므로 "
                "같을 필요는 없다(같은 과제·같은 모델인지만 본다)."
                % (float(q.era_auc.iloc[0]), era_auc - float(q.era_auc.iloc[0]))
            )
    log("=" * 126)

    # ---- 기준: 가중 없음 (= §8) ----
    p_base = S.make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    s0 = score(y[te], p_base)
    if ref_pc is not None:
        q = ref_pc[(ref_pc.dataset.str.startswith(name)) & (ref_pc.setting.str.startswith("T"))]
        if len(q):
            d0 = abs(float(q.ECE.iloc[0]) - s0["ece"])
            log(
                "  §8 시간분할 ECE 와의 차 %.2e → %s"
                % (d0, "일치" if d0 < TOL else "**불일치 — 중단**")
            )
            assert d0 < TOL, "§8 재현 실패"

    rows = [
        dict(
            s0,
            dataset=name,
            arm="W0. 가중 없음 (= §8)",
            cap=np.nan,
            n_eff=float(len(tr)),
            w_max=1.0,
            kind="outer",
        )
    ]

    log("")
    log(
        "%-26s %9s %9s %9s %9s %9s %9s %9s"
        % ("팔", "상한", "유효 n", "최대 w", "AUROC", "ECE", "기울기", "해상도")
    )
    log("-" * 126)
    log(
        "%-26s %9s %9.1f %9.2f %9.4f %9.4f %9.4f %9.4f"
        % (
            "W0. 가중 없음 (= §8)",
            "—",
            float(len(tr)),
            1.0,
            s0["auroc"],
            s0["ece"],
            s0["slope"],
            s0["res"],
        )
    )

    for cap in CAPS:
        w = dens_ratio(p_post[tr], cap)
        m = S.make_model().fit(X[tr], y[tr], sample_weight=w)
        s = score(y[te], m.predict_proba(X[te])[:, 1])
        lab = "IW. 밀도비 (절단 없음)" if cap is None else "IW. 밀도비 상한 %g" % cap
        rows.append(
            dict(
                s,
                dataset=name,
                arm=lab,
                cap=(np.nan if cap is None else float(cap)),
                n_eff=n_eff(w),
                w_max=float(w.max()),
                kind="outer",
            )
        )
        log(
            "%-26s %9s %9.1f %9.2f %9.4f %9.4f %9.4f %9.4f"
            % (
                lab,
                "—" if cap is None else "%g" % cap,
                n_eff(w),
                float(w.max()),
                s["auroc"],
                s["ece"],
                s["slope"],
                s["res"],
            )
        )

    # ---- 정직 프로토콜: 학습군 안에서만 절단 수준을 고른다 ----
    log("")
    log("정직 선택 — 학습군을 %d년으로 다시 쪼갠다 (검증군 미사용)" % INNER_CUT)
    i_fit = tr[yr[tr] < INNER_CUT]
    i_sel = tr[yr[tr] >= INNER_CUT]
    if len(i_fit) < 50 or len(i_sel) < 30 or len(np.unique(y[i_sel])) < 2:
        log("  내부 분할 표본 부족 — 정직 선택 생략")
        best_cap, best_lab = None, "정직 선택 불가"
    else:
        i_era = np.zeros(len(y), dtype=int)
        i_era[i_sel] = 1
        sub = np.concatenate([i_fit, i_sel])
        i_p = np.zeros(len(y), dtype=float)
        i_p[sub] = era_oof_prob(X[sub], i_era[sub])
        log(
            "  내부 적합 %d (~%d) / 내부 선택 %d (%d~%d), 내부 시대 AUC %.4f"
            % (
                len(i_fit),
                int(yr[i_fit].max()),
                len(i_sel),
                int(yr[i_sel].min()),
                int(yr[i_sel].max()),
                float(roc_auc_score(i_era[sub], i_p[sub])),
            )
        )
        log("  %-26s %9s %9s %9s %9s" % ("팔", "상한", "유효 n", "내부 ECE", "내부 AUROC"))
        best = None
        for cap in CAPS:
            w = dens_ratio(i_p[i_fit], cap)
            m = S.make_model().fit(X[i_fit], y[i_fit], sample_weight=w)
            s = score(y[i_sel], m.predict_proba(X[i_sel])[:, 1])
            lab = "IW. 밀도비 (절단 없음)" if cap is None else "IW. 밀도비 상한 %g" % cap
            rows.append(
                dict(
                    s,
                    dataset=name,
                    arm="내부 " + lab,
                    cap=(np.nan if cap is None else float(cap)),
                    n_eff=n_eff(w),
                    w_max=float(w.max()),
                    kind="inner",
                )
            )
            log(
                "  %-26s %9s %9.1f %9.4f %9.4f"
                % (lab, "—" if cap is None else "%g" % cap, n_eff(w), s["ece"], s["auroc"])
            )
            if best is None or s["ece"] < best[1]:
                best = (cap, s["ece"], lab)
        best_cap, best_lab = best[0], best[2]
        log("  → 정직 선택: **%s** (내부 ECE %.4f)" % (best_lab, best[1]))

    # ---- 고른 팔의 배율 ----
    sel = [
        r
        for r in rows
        if r["kind"] == "outer"
        and r["arm"] == (best_lab if best_cap is not None else "IW. 밀도비 (절단 없음)")
    ]
    ctrl_rows = []
    if sel:
        for tag, mt, mr in (("R1", False, False), ("R3", True, True)):
            E, _cauc = [], []
            for i in range(reps):
                r = np.random.default_rng(S.SEED + i)
                perm = r.permutation(len(y))
                ti, ri = perm[: len(te)], perm[len(te) :]
                if mt:
                    ti = S.subsample_to_rate(y, ti, rate_te, r)
                if mr:
                    ri = S.subsample_to_rate(y, ri, rate_tr, r)
                if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                    continue
                if len(ti) < 30 or len(ri) < 100:
                    continue
                # 대조군에도 같은 처방을 건다 — 처방의 효과를 공정하게 보려면
                # 대조군 쪽에서도 같은 절차를 돌려야 한다.
                e2 = np.zeros(len(y), dtype=int)
                e2[ti] = 1
                s2 = np.concatenate([ri, ti])
                pp2 = np.zeros(len(y), dtype=float)
                pp2[s2] = era_oof_prob(X[s2], e2[s2])
                _cauc.append(float(roc_auc_score(e2[s2], pp2[s2])))
                w2 = dens_ratio(pp2[ri], best_cap)
                mm = S.make_model().fit(X[ri], y[ri], sample_weight=w2)
                E.append(S.ece(y[ti], mm.predict_proba(X[ti])[:, 1]))
            c = float(np.mean(E)) if E else float("nan")
            ctrl_rows.append(
                {
                    "dataset": name,
                    "arm": sel[0]["arm"],
                    "control": tag,
                    "ctrl_ece": c,
                    "n_rep_valid": len(E),
                    "ctrl_era_auc": (float(np.mean(_cauc)) if _cauc else float("nan")),
                    "ratio": (sel[0]["ece"] / c if c == c and c > 0 else float("nan")),
                }
            )
        log("")
        log("  고른 팔의 배율 (대조군에도 같은 처방을 적용했다)")
        for cr in ctrl_rows:
            log(
                "    %-4s 대조 ECE %.4f (유효 %d회) → 배율 %.2f  "
                "[대조군 시대 AUC %.4f — 무작위 분할이므로 0.5 근처여야 하고, "
                "그러면 가중이 1 에 가까워 처방이 아무 일도 하지 않는다]"
                % (
                    cr["control"],
                    cr["ctrl_ece"],
                    cr["n_rep_valid"],
                    cr["ratio"],
                    cr["ctrl_era_auc"],
                )
            )
    log("")
    return rows, ctrl_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    log("공변량 이동 보정 — 도메인 분류기로 밀도비 가중을 걸면 되는가")
    log(
        "w(x) = P(post|x)/(1-P(post|x)), 평균 1 로 정규화, 상한 %s"
        % ", ".join("없음" if c is None else "%g" % c for c in CAPS)
    )
    log("시대 분류기는 라벨 y 를 쓰지 않는다 — 원리적으로 배포 가능한 처방이다")
    log("(미래 분자들의 구조를 모아 두어야 한다는 조건은 붙는다).")
    log("대조군 %d회, 시드 %d" % (a.reps, S.SEED))
    log("")

    def rd(n):
        p = os.path.join(RES, n)
        return pd.read_csv(p) if os.path.exists(p) else None

    pc, ep = rd("prevalence_control.csv"), rd("era_permutation.csv")
    rows, ctrl = [], []
    for name, fn in S.DATASETS:
        x, c = run(name, fn, a.reps, pc, ep)
        rows += x
        ctrl += c
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "covariate_shift_weighting.csv"), index=False, encoding="utf-8")
    if ctrl:
        pd.DataFrame(ctrl).to_csv(
            os.path.join(RES, "covariate_shift_weighting_ctrl.csv"), index=False, encoding="utf-8"
        )

    log("=" * 126)
    log("판정")
    log("=" * 126)
    log("[읽는 법] ECE 가 내려가도 해상도(res)가 함께 무너지면 처방이 아니다")
    log("          — 예측을 평평하게 만들면 ECE 는 내려간다(§20 에서 겪은 함정).")
    log("          유효 n 도 함께 본다. 밀도비는 꼬리가 두꺼워 표본을 잡아먹는다.")
    log("")
    for d, g in out[out.kind == "outer"].groupby("dataset", sort=False):
        w0 = g[g.arm.str.startswith("W0")].iloc[0]
        iw = g[g.arm.str.startswith("IW")]
        best = iw.loc[iw.ece.idxmin()]
        log("  %s" % d)
        log(
            "    가중 없음  ECE %.4f / AUROC %.4f / 해상도 %.4f / 유효 n %.0f"
            % (w0.ece, w0.auroc, w0.res, w0.n_eff)
        )
        log(
            "    IW 최선    ECE %.4f (%s) / AUROC %.4f / 해상도 %.4f / 유효 n %.0f"
            % (best.ece, best.arm, best.auroc, best.res, best.n_eff)
        )
        log(
            "    ECE 변화 %+.4f, 해상도 변화 %+.4f, 유효 n %.0f → %.0f (%.0f%% 손실)"
            % (
                best.ece - w0.ece,
                best.res - w0.res,
                w0.n_eff,
                best.n_eff,
                100.0 * (1 - best.n_eff / w0.n_eff),
            )
        )
        if best.ece < w0.ece and best.res >= w0.res * 0.8:
            log("    → **ECE 가 내려가고 해상도는 지켰다. 처방 후보다.**")
        elif best.ece < w0.ece:
            log("    → ECE 는 내려갔지만 해상도가 함께 무너졌다. §20 과 같은 함정이다.")
        else:
            log("    → **어떤 상한에서도 ECE 가 내려가지 않는다. 처방이 아니다.**")
        log("")
    log("저장: %s" % os.path.join(RES, "covariate_shift_weighting.csv"))
    log.save(Path(RES) / "covariate_shift_weighting_summary.txt")


if __name__ == "__main__":
    main()
