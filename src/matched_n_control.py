"""
표본크기 일치 유병률 대조군 + 순열 횟수 증대 (2026-09-28)

왜 필요한가
----------
§7 은 "무작위 분할을 시간분할과 동일한 검증군 크기로 분해한다 — 양쪽 열이 같은
표본 크기 위에 있다" 를 설계 원칙으로 선언한다. 그런데 §7.1·§8 의 R2·R3 는
유병률을 부표집으로 맞추는 과정에서 대조군 n 이 줄어든다:

    DILIrank  대조군 n=92   대 시간분할 n=156
    DICTrank  대조군 n=189  대 시간분할 n=267

ECE·reliability·resolution 은 모두 구간당 표본 수에 의존해 위로 편향되는
통계량(δ ≈ Σ o_b(1-o_b)/N)이므로 두 결과가 따라온다:

  (1) 순열 p 가 무효다. 관측값은 n=156 에서, 귀무분포 200개는 n=92 에서
      계산됐다. 귀무가설이 참이어도 두 분포가 다르므로 이 p 는
      "차이 없음" 에 대한 p 가 아니다.
  (2) 보정 기여율이 편향된다. Δrel = rel_T - rel_C 는 (δ_T - δ_C) 만큼
      하향, Δres 는 그만큼 상향 편향되고 분모는 상쇄되므로,
      비율만 |Δδ|/분모 만큼 하향 편향된다. 즉 헤드라인 80.6% / 53.6% 는
      실제보다 낮게 보고돼 있을 가능성이 높다.

해결
----
시간분할 검증군도 대조군과 같은 n 으로 무복원 부표집한다. 시간분할 검증군은
이미 목표 유병률(30.8% / 61.4%)이므로 층화 부표집만 하면 유병률이 보존된다.
그러면 양쪽이 같은 n 이 되어 편향이 상쇄되고 순열 p 가 유효해진다.

동시에 순열 횟수를 200 → 2000 으로 올린다. 200회에서는 p 하한이
1/201 = 0.0050 이라, 5σ 수준 증거(§13 시대 분류기)와 p≈0.03 수준 증거
(§8 DICTrank R3)가 같은 숫자로 인쇄된다.

실험
----
  M1. R1/R2/R3 × {기존 방식, n 일치 방식} — ECE 배율과 p
  M2. R1/R2/R3 × n 일치 — Brier 분해 보정 기여율 (2항·3항 분모)
  M3. 시대 분류기 순열 2000회 (§13(가) 재계산)

사용법
------
    python src/matched_n_control.py --skip-era      # M1·M2 만, 약 25분  <- 먼저 이것
    python src/matched_n_control.py --perm 1000     # M3 까지, 약 4~5시간 (nohup 권장)
    python src/matched_n_control.py --quick         # reps 30 / perm 100, 배관 점검용 (2~3분)

M3 은 순열 1회당 5-fold XGBoost 학습 5번이 필요해 1000회면 데이터셋당 2시간이
넘는다. M1·M2 가 헤드라인 수치를 고치는 쪽이므로 --skip-era 로 먼저 돌리고,
M3 은 밤에 따로 돌리는 것을 권한다.

출력
----
    results/matched_n_control.csv
    results/matched_n_control_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
log = S.Log()


# ------------------------------------------------------------ 층화 부표집
def stratified_subsample(y, idx, n_target, rng):
    """idx 에서 n_target 개를 뽑되 양성률을 최대한 보존한다(무복원)."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    rate = len(pos) / len(idx)
    n_pos = int(round(n_target * rate))
    n_pos = max(1, min(n_pos, len(pos)))
    n_neg = n_target - n_pos
    if n_neg > len(neg):
        n_neg = len(neg)
        n_pos = n_target - n_neg
    if n_pos < 1 or n_neg < 1:
        return None
    return np.concatenate(
        [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    )


def bin_noise(y, p, n_bins=S.N_BINS):
    """구간잡음 편향 δ = Σ_b n_b/N · o_b(1-o_b)/n_b = Σ_b o_b(1-o_b)/N"""
    edges = np.linspace(0, 1, n_bins + 1)
    N = len(p)
    tot = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        o = y[m].mean()
        tot += o * (1 - o) / N
    return float(tot)


# ------------------------------------------------------------ M1 + M2
def run_prevalence(name, path, reps):
    df = S.load(DATA / path)
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr, te = S.temporal_masks(df)
    te_idx = np.where(te)[0]
    n_test = len(te_idx)
    rate_t = float(y[te].mean())
    rate_tr = float(y[tr].mean())

    log("=" * 100)
    log(f"{name}   학습 {int(tr.sum())} / 검증 {n_test}")
    log(f"  검증군 양성률 {rate_t:.4f}  학습군 양성률 {rate_tr:.4f}")
    log("=" * 100)

    m = S.make_model().fit(X[tr], y[tr])
    p_te_full = m.predict_proba(X[te])[:, 1]
    ece_T_full = S.ece(y[te], p_te_full)
    dec_T_full = S.decompose(y[te], p_te_full)
    log(
        f"\n  [T 전체] n={n_test}  ECE {ece_T_full:.4f}  "
        f"AUROC {roc_auc_score(y[te], p_te_full):.4f}  "
        f"rel {dec_T_full['reliability']:.4f}  res {dec_T_full['resolution']:.4f}  "
        f"구간잡음 δ {bin_noise(y[te], p_te_full):.4f}"
    )

    rows = []
    for base, target_te, target_tr in (
        ("R1", None, None),
        ("R2", rate_t, None),
        ("R3", rate_t, rate_tr),
    ):
        # --- 대조군 200회 ---
        ece_c, rel_c, res_c, unc_c, n_c, auc_c, dlt_c = [], [], [], [], [], [], []
        for i in range(reps):
            r = np.random.default_rng(S.SEED + i)
            perm = r.permutation(len(y))
            ti, ri = perm[:n_test], perm[n_test:]
            if target_te is not None:
                ti = S.subsample_to_rate(y, ti, target_te, r)
            if target_tr is not None:
                ri = S.subsample_to_rate(y, ri, target_tr, r)
            if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                continue
            mm = S.make_model().fit(X[ri], y[ri])
            pp = mm.predict_proba(X[ti])[:, 1]
            d = S.decompose(y[ti], pp)
            ece_c.append(S.ece(y[ti], pp))
            rel_c.append(d["reliability"])
            res_c.append(d["resolution"])
            unc_c.append(d["uncertainty"])
            n_c.append(len(ti))
            auc_c.append(roc_auc_score(y[ti], pp))
            dlt_c.append(bin_noise(y[ti], pp))
        n_ctrl = int(round(float(np.mean(n_c))))

        # --- 관측 쪽: (a) 전체 n, (b) 대조군과 같은 n 으로 부표집 ---
        ece_Tm, rel_Tm, res_Tm, unc_Tm, dlt_Tm = [], [], [], [], []
        if n_ctrl < n_test:
            for i in range(reps):
                r = np.random.default_rng(10_000 + S.SEED + i)
                si = stratified_subsample(y, te_idx, n_ctrl, r)
                if si is None or len(np.unique(y[si])) < 2:
                    continue
                ps = p_te_full[np.searchsorted(te_idx, si)]
                d = S.decompose(y[si], ps)
                ece_Tm.append(S.ece(y[si], ps))
                rel_Tm.append(d["reliability"])
                res_Tm.append(d["resolution"])
                unc_Tm.append(d["uncertainty"])
                dlt_Tm.append(bin_noise(y[si], ps))
        else:
            ece_Tm = [ece_T_full]
            rel_Tm = [dec_T_full["reliability"]]
            res_Tm = [dec_T_full["resolution"]]
            unc_Tm = [dec_T_full["uncertainty"]]
            dlt_Tm = [bin_noise(y[te], p_te_full)]

        def share(dr, ds):
            return 100 * dr / (dr + ds) if (dr + ds) else np.nan

        # 기존 방식 (관측=전체 n, 대조군=축소 n)
        dR_old = dec_T_full["reliability"] - float(np.mean(rel_c))
        dS_old = float(np.mean(res_c)) - dec_T_full["resolution"]
        ratio_old = ece_T_full / float(np.mean(ece_c))
        r_ge_old = int(sum(1 for e in ece_c if e >= ece_T_full))
        p_old = (1 + r_ge_old) / (1 + len(ece_c))

        # n 일치 방식 (양쪽 모두 n_ctrl)
        ece_Tm_mean = float(np.mean(ece_Tm))
        dR_new = float(np.mean(rel_Tm)) - float(np.mean(rel_c))
        dS_new = float(np.mean(res_c)) - float(np.mean(res_Tm))
        ratio_new = ece_Tm_mean / float(np.mean(ece_c))
        # p: 부표집 관측분포의 중앙값을 관측값으로 사용 (보수적)
        obs_med = float(np.median(ece_Tm))
        r_ge_new = int(sum(1 for e in ece_c if e >= obs_med))
        p_new = (1 + r_ge_new) / (1 + len(ece_c))

        log(f"\n[{base}]  유효 {len(ece_c)}/{reps} 회   대조군 n={n_ctrl}")
        log(
            f"    대조군       ECE {np.mean(ece_c):.4f} ± {np.std(ece_c):.4f}"
            f"   rel {np.mean(rel_c):.4f}  res {np.mean(res_c):.4f}"
            f"   δ {np.mean(dlt_c):.4f}"
        )
        log(
            f"    [기존] 관측 n={n_test}  ECE {ece_T_full:.4f}"
            f"   배율 {ratio_old:.2f}   p {p_old:.4f}"
            f"   보정기여 {share(dR_old, dS_old):.1f}%"
        )
        log(
            f"    [일치] 관측 n={n_ctrl}   ECE {ece_Tm_mean:.4f} (중앙 {obs_med:.4f})"
            f"   배율 {ratio_new:.2f}   p {p_new:.4f}"
            f"   보정기여 {share(dR_new, dS_new):.1f}%"
            f"   δ {np.mean(dlt_Tm):.4f}"
        )
        log(
            f"    δ 차이(관측−대조) 기존 {bin_noise(y[te],p_te_full)-np.mean(dlt_c):+.4f}"
            f"  → 일치 {np.mean(dlt_Tm)-np.mean(dlt_c):+.4f}"
        )

        rows.append(
            dict(
                dataset=name,
                baseline=base,
                n_rep=len(ece_c),
                n_test_temporal=n_test,
                n_ctrl=n_ctrl,
                ece_ctrl=float(np.mean(ece_c)),
                ece_ctrl_sd=float(np.std(ece_c)),
                rel_ctrl=float(np.mean(rel_c)),
                res_ctrl=float(np.mean(res_c)),
                unc_ctrl=float(np.mean(unc_c)),
                delta_ctrl=float(np.mean(dlt_c)),
                ece_T_full=ece_T_full,
                ratio_old=ratio_old,
                p_old=p_old,
                share_old=share(dR_old, dS_old),
                ece_T_matched=ece_Tm_mean,
                ece_T_matched_med=obs_med,
                ratio_new=ratio_new,
                p_new=p_new,
                share_new=share(dR_new, dS_new),
                rel_T_matched=float(np.mean(rel_Tm)),
                res_T_matched=float(np.mean(res_Tm)),
                delta_T_matched=float(np.mean(dlt_Tm)),
                d_rel_old=dR_old,
                d_res_old=dS_old,
                d_rel_new=dR_new,
                d_res_new=dS_new,
            )
        )
    return rows


# ------------------------------------------------------------ M3 시대 분류기 순열 증대
def run_era(name, path, n_perm):
    from sklearn.model_selection import StratifiedKFold

    df = S.load(DATA / path)
    X = S.featurize(df["smiles"].tolist())
    era = (df["approval_year"].values >= S.CUTOFF).astype(int)

    def cv_auc(labels, seed=S.SEED):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(labels))
        for tri, tei in skf.split(X, labels):
            mm = S.make_model().fit(X[tri], labels[tri])
            oof[tei] = mm.predict_proba(X[tei])[:, 1]
        return roc_auc_score(labels, oof)

    obs = cv_auc(era)
    log("=" * 100)
    log(f"{name}  시대 분류기 순열 {n_perm}회   관측 AUC {obs:.4f}")
    log("=" * 100)
    null = []
    for i in range(n_perm):
        r = np.random.default_rng(S.SEED + i)
        null.append(cv_auc(r.permutation(era), seed=S.SEED))
        if (i + 1) % max(1, n_perm // 10) == 0:
            log(f"    {i+1}/{n_perm}  귀무평균 {np.mean(null):.4f}")
    null = np.array(null)
    r_ge = int((null >= obs).sum())
    p = (1 + r_ge) / (1 + len(null))
    log(
        f"  귀무 평균 {null.mean():.4f} (SD {null.std():.4f})  "
        f"2.5–97.5 {np.percentile(null,2.5):.4f}–{np.percentile(null,97.5):.4f}"
    )
    log(
        f"  관측값 이상 {r_ge}/{len(null)}   p (Phipson–Smyth) = {p:.6f}"
        f"   하한 1/{len(null)+1} = {1/(len(null)+1):.6f}"
    )
    z = (obs - null.mean()) / null.std() if null.std() else np.nan
    log(f"  귀무 SD 기준 z = {z:.1f}")
    return dict(
        dataset=name,
        era_auc=obs,
        n_perm=len(null),
        null_mean=float(null.mean()),
        null_sd=float(null.std()),
        n_ge=r_ge,
        p=p,
        p_floor=1 / (len(null) + 1),
        z=z,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--skip-era",
        action="store_true",
        help="M3(시대 분류기 순열) 생략. M1·M2 만 약 25분에 끝난다.",
    )
    a = ap.parse_args()
    if a.quick:
        a.reps, a.perm = 30, 100

    log("표본크기 일치 유병률 대조군 + 순열 횟수 증대")
    log(f"대조군 반복 {a.reps}회, 시대분류 순열 {a.perm}회, seed {S.SEED}, 절단 {S.CUTOFF}")
    log("")

    rows = []
    for nm, f in S.DATASETS:
        rows += run_prevalence(nm, f, a.reps)
        log("")
    pd.DataFrame(rows).to_csv(RES / "matched_n_control.csv", index=False)

    era = []
    if not a.skip_era:
        for nm, f in S.DATASETS:
            era.append(run_era(nm, f, a.perm))
            log("")
        pd.DataFrame(era).to_csv(RES / "matched_n_era_permutation.csv", index=False)

    log("=" * 100)
    log("판정")
    log("=" * 100)
    log("")
    log(
        f"{'데이터셋':10s} {'대조군':5s} {'배율(기존)':>10s} {'배율(일치)':>10s}"
        f" {'p(기존)':>9s} {'p(일치)':>9s} {'기여(기존)':>10s} {'기여(일치)':>10s}"
    )
    log("-" * 100)
    for r in rows:
        log(
            f"{r['dataset']:10s} {r['baseline']:5s} {r['ratio_old']:10.2f} {r['ratio_new']:10.2f}"
            f" {r['p_old']:9.4f} {r['p_new']:9.4f}"
            f" {r['share_old']:9.1f}% {r['share_new']:9.1f}%"
        )
    log("")
    log("[읽는 법] '일치' 열이 양쪽 n 을 맞춘 값이다. 기여율이 '일치' 에서 올라가면")
    log("          기존 헤드라인(80.6% / 53.6%)이 표본크기 차이 때문에 하향 편향돼")
    log("          있었다는 뜻이다. 배율은 반대로 내려갈 수 있다(대조군 ECE 가 더는")
    log("          작은 n 으로 부풀려지지 않으므로). 두 방향 모두 정상이다.")
    if era:
        log("")
        for e in era:
            log(
                f"{e['dataset']:10s} 시대분류 AUC {e['era_auc']:.4f}  "
                f"p {e['p']:.6f} (하한 {e['p_floor']:.6f})  z {e['z']:.1f}"
            )
        log("  p 가 하한에서 떨어졌으면 §15 FDR 에서 여유가 생긴다.")
    log("")
    log(f"저장: {RES/'matched_n_control.csv'}")
    if era:
        log(f"      {RES/'matched_n_era_permutation.csv'}")
    log(f"      {RES/'matched_n_control_summary.txt'}")
    log.save(RES / "matched_n_control_summary.txt")


if __name__ == "__main__":
    main()
