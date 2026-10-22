# -*- coding: utf-8 -*-
"""개념 이동(concept shift) 검정 + 두 데이터셋 추정치의 상관.

왜 필요한가 (심사 지적 Major 1·2 대응)
--------------------------------------
지금까지 논문은 분포 이동을 둘만 통제했다.
  - 라벨 분포 P(y)      : §16 유병률 사다리 R1→R2→R3
  - 입력 분포 P(x)      : §9 공변량 이동 정량화
빠진 것이 세 번째 — P(y|x) 자체가 변했는가다. 구조-독성 관계가
시대에 따라 바뀌었다면 (적응증 변화, 평가 기준 변화, 신약 양식 변화)
보정 붕괴는 '표본이 작아서' 가 아니라 '관계가 달라서' 생긴 것이다.
이 구분은 서술을 '현상 보고' 에서 '기전 진술' 로 올린다.

설계 — 검증군을 똑같이 고정하고 학습군의 성질만 바꾼다
----------------------------------------------------------
네 팔 모두 2010년 이후 분자 전체를 검증군으로 쓴다 (A·A2 는
out-of-fold 예측이므로 자기 분자로 학습한 모델의 예측이 아니다).

  A   시대일치 (무작위 5-fold)   학습: post-2010 내부 4/5
  A2  시대일치 (스캐폴드 5-fold) 학습: post-2010 내부 4/5, 스캐폴드 분리
  B   정방향 전체               학습: pre-2010 전체          ← §8 의 설정
  C   정방향 크기맞춤            학습: pre-2010 에서 A 와 같은 수
  D   정방향 크기·유병률 맞춤     학습: pre-2010 에서 같은 수 + 같은 양성률

읽는 법
  A 대 D  : 학습 크기·학습 양성률·검증군이 모두 같고 시대만 다르다.
            여기서 차이가 남으면 그것이 개념 이동의 몫이다.
  C 대 B  : 시대가 같고 크기만 다르다 → 표본 크기의 몫.
  A 대 A2 : A 의 학습군은 검증군과 같은 표본에서 나왔으므로 유사체
            (analog series) 가 fold 를 넘나들어 A 를 유리하게 만들 수
            있다. A2 는 스캐폴드를 분리해 그 이득을 없앤다. A2 가
            여전히 D 보다 잘 보정되면 유사체 누출로는 설명되지 않는다.

한계 (본문에 그대로 쓸 것)
  - A·A2 의 통합 예측은 서로 다른 5개 모델에서 나왔다. 모델별 절편
    차이가 통합 보정에 섞인다.
  - A·A2 의 학습 n 은 fold 마다 다르다. C·D 의 목표 n 은 그 평균이다.
  - D 는 양성률을 맞추느라 pre-2010 을 비복원 재표집하므로 화학공간
    대표성이 A 와 완전히 같지는 않다.

같은 실행에 묶는 두 번째 분석 (Major 1: 두 데이터셋의 비독립성)
----------------------------------------------------------------
DILIrank 와 DICTrank 는 ikey14 기준 595개 분자를 공유한다. 심사자는
"그러면 두 결과는 독립 재현이 아니다" 라고 지적할 것이다. §13 은
전용(비공유) 부분집합으로 재현을 보였지만, 공유분에서 두 추정치가
실제로 얼마나 상관돼 있는지는 측정하지 않았다. 여기서 측정한다.

  두 데이터셋 모두에서 2010년 이후인 공유 분자에 대해, 각 데이터셋
  정방향 모델(팔 B)의 잔차 (y - p) 를 짝지어 상관을 본다. 상관이
  낮으면 구조는 겹치지만 오차는 겹치지 않는다 — 두 결과는 사실상
  독립 정보다. 라벨 일치도(phi)도 같이 보고한다.

사용법
------
    python src/concept_shift.py                # 대조군 50회
    python src/concept_shift.py --reps 10      # 짧게 점검

출력
----
    results/concept_shift.csv
    results/concept_shift_overlap.csv
    results/concept_shift_summary.txt
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
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

N_FOLDS = 5
N_BOOT = 2000


# ----------------------------------------------------------------- 공용
def phipson_smyth(r, n):
    """(1+r)/(1+n). 순열/참조분포 p 의 편향보정 추정량 (Phipson & Smyth 2010)."""
    return (1.0 + r) / (1.0 + n)


def pct(part, total):
    """몫을 전체 대비 %로. 전체가 0 에 가까우면 비율을 쓰지 않는다."""
    if total == 0 or abs(total) < 1e-12:
        return "(전체가 0 — 비율 생략)"
    return "(전체의 %+.1f%%)" % (100.0 * part / total)


def metrics(y, p):
    d = S.decompose(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "ece": S.ece(y, p),
        "slope": S.calib_slope(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
        "unc": d["uncertainty"],
    }


def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_folds(smiles_list, n_folds=N_FOLDS):
    """baseline.py 와 같은 구현 (DeepChem 방식: 큰 그룹부터 작은 fold 에)."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[scaffold_of(smi)].append(i)
    ordered = sorted(groups.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_folds)]
    for grp in ordered:
        folds.sort(key=len)
        folds[0].extend(grp)
    return [sorted(f) for f in folds]


def oof(Xs, ys, folds):
    """fold 별로 적합해 out-of-fold 예측을 모은다.

    돌려주는 값: (예측, 평균 학습 n, 평균 학습 양성률).
    스캐폴드 fold 는 층화가 아니므로 학습 양성률이 fold 마다 다르다.
    그래서 전체 양성률을 쓰지 않고 실제 평균을 기록한다.
    """
    p = np.full(len(ys), np.nan)
    tr_n, tr_rate = [], []
    for te in folds:
        te = np.asarray(te, dtype=int)
        tr = np.setdiff1d(np.arange(len(ys)), te)
        if len(np.unique(ys[tr])) < 2 or len(te) == 0:
            continue
        m = S.make_model().fit(Xs[tr], ys[tr])
        p[te] = m.predict_proba(Xs[te])[:, 1]
        tr_n.append(len(tr))
        tr_rate.append(float(ys[tr].mean()))
    nan = float("nan")
    return (p, float(np.mean(tr_n)) if tr_n else nan, float(np.mean(tr_rate)) if tr_rate else nan)


def forward(X, y, tr_idx, te_idx):
    m = S.make_model().fit(X[tr_idx], y[tr_idx])
    return m.predict_proba(X[te_idx])[:, 1]


# ----------------------------------------------------------------- 본체
def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    X = S.featurize(df["smiles"].tolist())
    y = df["y"].values
    tr_mask, te_mask = S.temporal_masks(df)  # pre-2010 / post-2010
    pre = np.where(tr_mask)[0]
    post = np.where(te_mask)[0]
    y_post = y[post]
    X_post = X[post]
    rate_post = float(y_post.mean())
    rate_pre = float(y[pre].mean())

    log("=" * 104)
    log(
        "%s   pre-2010 %d개(양성률 %.4f) / post-2010 %d개(양성률 %.4f)"
        % (name, len(pre), rate_pre, len(post), rate_post)
    )
    log("  네 팔 모두 검증군은 post-2010 %d개 **동일**하다." % len(post))
    log("=" * 104)

    rows = []

    # --- A, A2 : 시대일치 out-of-fold -------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=S.SEED)
    rand_folds = [te.tolist() for _, te in skf.split(X_post, y_post)]
    scaf_folds = scaffold_folds(df["smiles"].values[post].tolist())

    arms = {}
    for tag, folds in (
        ("A. 시대일치 무작위 5-fold", rand_folds),
        ("A2. 시대일치 스캐폴드 5-fold", scaf_folds),
    ):
        p, mean_tr, mean_rate = oof(X_post, y_post, folds)
        if np.isnan(p).any():
            # fold 분할이 한쪽으로 쏠려 학습군이 단일 라벨이 된 경우다.
            # 검증군이 더 이상 네 팔에서 동일하지 않게 되므로 이 팔만 뺀다.
            log(
                "  [경고] %s: out-of-fold 예측 누락 %d개 — 이 팔을 건너뛴다."
                % (tag, int(np.isnan(p).sum()))
            )
            if tag.startswith("A."):
                raise RuntimeError("기준 팔 A 가 실패했다 — 중단한다.")
            continue
        mm = metrics(y_post, p)
        arms[tag] = dict(mm, pred=p, train_n=mean_tr)
        rows.append(
            {
                "dataset": name,
                "arm": tag,
                "train_era": "post-2010",
                "train_n": mean_tr,
                "train_rate": mean_rate,
                "n_test": len(post),
                "test_rate": rate_post,
                "n_rep_valid": N_FOLDS,
                "n_better": np.nan,
                "p_phipson_smyth": np.nan,
                "ece_ratio": np.nan,
                **{k: v for k, v in mm.items()},
            }
        )

    # --- B : 정방향 전체 (§8 설정) ---------------------------------
    p_b = forward(X, y, pre, post)
    mb = metrics(y_post, p_b)
    rows.append(
        {
            "dataset": name,
            "arm": "B. 정방향 전체",
            "train_era": "pre-2010",
            "train_n": float(len(pre)),
            "train_rate": rate_pre,
            "n_test": len(post),
            "test_rate": rate_post,
            "n_rep_valid": 1,
            "n_better": np.nan,
            "p_phipson_smyth": np.nan,
            "ece_ratio": np.nan,
            **mb,
        }
    )

    # --- C, D : 정방향 맞춤 대조군 ---------------------------------
    n_target = int(round(arms["A. 시대일치 무작위 5-fold"]["train_n"]))
    pre_pos = pre[y[pre] == 1]
    pre_neg = pre[y[pre] == 0]
    n_pos_t = int(round(n_target * rate_post))
    n_neg_t = n_target - n_pos_t
    log(
        "  대조군 목표 학습 n = %d (팔 A 의 fold 평균). "
        "유병률 맞춤 시 양성 %d / 음성 %d" % (n_target, n_pos_t, n_neg_t)
    )
    if n_pos_t > len(pre_pos) or n_neg_t > len(pre_neg):
        log(
            "  [경고] pre-2010 에 필요한 양성/음성이 부족하다 "
            "(보유 양성 %d / 음성 %d) — 팔 D 를 건너뛴다." % (len(pre_pos), len(pre_neg))
        )
    log("")

    ece_ref = arms["A. 시대일치 무작위 5-fold"]["ece"]
    for tag, match_rate in (("C. 정방향 크기맞춤", False), ("D. 정방향 크기·유병률맞춤", True)):
        if match_rate and (n_pos_t > len(pre_pos) or n_neg_t > len(pre_neg)):
            continue
        acc = defaultdict(list)
        for i in range(reps):
            r = np.random.default_rng(S.SEED + 10000 + i)
            if match_rate:
                tr = np.concatenate(
                    [
                        r.choice(pre_pos, n_pos_t, replace=False),
                        r.choice(pre_neg, n_neg_t, replace=False),
                    ]
                )
            else:
                tr = r.choice(pre, n_target, replace=False)
            if len(np.unique(y[tr])) < 2:
                continue
            p = forward(X, y, tr, post)
            m = metrics(y_post, p)
            for k, v in m.items():
                acc[k].append(v)
            acc["train_rate"].append(float(y[tr].mean()))
        if not acc["ece"]:
            log("  %s 유효 반복 0회 — 건너뜀" % tag)
            continue
        E = np.array(acc["ece"])
        n_better = int((E <= ece_ref).sum())  # 대조군이 A 만큼 잘 보정된 횟수
        rows.append(
            {
                "dataset": name,
                "arm": tag,
                "train_era": "pre-2010",
                "train_n": float(n_target),
                "train_rate": float(np.mean(acc["train_rate"])),
                "n_test": len(post),
                "test_rate": rate_post,
                "auroc": float(np.mean(acc["auroc"])),
                "ece": float(E.mean()),
                "slope": float(np.mean(acc["slope"])),
                "rel": float(np.mean(acc["rel"])),
                "res": float(np.mean(acc["res"])),
                "unc": float(np.mean(acc["unc"])),
                "n_rep_valid": int(len(E)),
                "n_better": n_better,
                "p_phipson_smyth": phipson_smyth(n_better, len(E)),
                "ece_ratio": float(E.mean()) / ece_ref,
            }
        )

    # --- 표 출력 ----------------------------------------------------
    log(
        "%-28s %7s %7s %7s %8s %8s %8s %8s %8s"
        % ("팔", "학습n", "학습률", "AUROC", "ECE", "기울기", "신뢰도", "A대비", "p")
    )
    log("-" * 104)
    for r_ in rows:
        if r_["dataset"] != name:
            continue
        rat = ("%8.2f" % r_["ece_ratio"]) if r_["ece_ratio"] == r_["ece_ratio"] else "%8s" % "—"
        pv = (
            ("%8.4f" % r_["p_phipson_smyth"])
            if r_["p_phipson_smyth"] == r_["p_phipson_smyth"]
            else "%8s" % "—"
        )
        log(
            "%-28s %7.0f %7.4f %7.4f %8.4f %8.4f %8.4f %s %s"
            % (
                r_["arm"],
                r_["train_n"],
                r_["train_rate"],
                r_["auroc"],
                r_["ece"],
                r_["slope"],
                r_["rel"],
                rat,
                pv,
            )
        )
    log("")
    return rows, {
        # 검증군(post-2010)용 — 잔차 상관에 쓴다
        "ikey": df["ikey14"].values[post],
        "y": y_post,
        "p": p_b,
        "year": df["approval_year"].values[post],
        # 전체 분석집합용 — 공유분 집계·승인연도 불일치 점검에 쓴다
        "all": pd.DataFrame(
            {"ikey": df["ikey14"].values, "y": y, "year": df["approval_year"].values}
        ),
    }


# ------------------------------------------------- 두 데이터셋 상관
def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return pearson(ra, rb)


def boot_ci(a, b, n_boot=N_BOOT, seed=S.SEED):
    r = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if np.std(a[i]) == 0 or np.std(b[i]) == 0:
            continue
        vals.append(pearson(a[i], b[i]))
    if not vals:
        return float("nan"), float("nan"), 0
    v = np.sort(np.array(vals))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def overlap_analysis(store):
    names = list(store.keys())
    if len(names) < 2:
        log("데이터셋이 둘이 아니어서 상관 분석을 건너뛴다.")
        return pd.DataFrame()
    a_name, b_name = names[0], names[1]
    key = ["ikey", "y", "p", "year"]
    A = pd.DataFrame({k: store[a_name][k] for k in key}).dropna(subset=["ikey"])
    B = pd.DataFrame({k: store[b_name][k] for k in key}).dropna(subset=["ikey"])
    j = A.merge(B, on="ikey", suffixes=("_a", "_b"))

    log("=" * 104)
    log("두 데이터셋 추정치의 상관 — 공유 분자 (Major 1 대응)")
    log("=" * 104)
    # 전체 분석집합 기준 공유분 — §13 의 595개와 일치해야 한다
    fa = store[a_name]["all"].dropna(subset=["ikey"])
    fb = store[b_name]["all"].dropna(subset=["ikey"])
    fj = fa.merge(fb, on="ikey", suffixes=("_a", "_b"))
    log(
        "  전체 분석집합: %s %d개 / %s %d개 / 공유 %d개"
        % (a_name, len(fa), b_name, len(fb), len(fj))
    )
    if len(fj):
        log(
            "    라벨 일치 %.1f%%   phi = %+.4f   (%s 양성률 %.4f / %s %.4f)"
            % (
                100.0 * float((fj.y_a == fj.y_b).mean()),
                pearson(fj.y_a.values.astype(float), fj.y_b.values.astype(float)),
                a_name,
                float(fj.y_a.mean()),
                b_name,
                float(fj.y_b.mean()),
            )
        )
        log(
            "    승인연도 불일치 %d개 (공유 분자 %d개 중) — 자료 정제 항목"
            % (int((fj.year_a != fj.year_b).sum()), len(fj))
        )
    log("")
    log("  %s 검증군 %d개 / %s 검증군 %d개" % (a_name, len(A), b_name, len(B)))
    log("  양쪽 검증군(둘 다 post-2010)에 모두 있는 분자 %d개" % len(j))
    log(
        "    = %s 검증군의 %.1f%% / %s 검증군의 %.1f%%"
        % (a_name, 100.0 * len(j) / len(A), b_name, 100.0 * len(j) / len(B))
    )
    if len(j) < 20:
        log("  겹치는 분자가 20개 미만이어서 상관을 보고하지 않는다.")
        return pd.DataFrame()

    log(
        "  승인연도 불일치 %d개 (양쪽 모두 post-2010 인 분자 중)"
        % int((j.year_a != j.year_b).sum())
    )
    log("")

    res_a = (j.y_a - j.p_a).values
    res_b = (j.y_b - j.p_b).values
    out = []
    for tag, va, vb in (
        ("잔차 y-p (부호 있음)", res_a, res_b),
        ("절대잔차 |y-p|", np.abs(res_a), np.abs(res_b)),
        ("라벨 y (phi)", j.y_a.values.astype(float), j.y_b.values.astype(float)),
        ("예측 p", j.p_a.values, j.p_b.values),
    ):
        rp = pearson(va, vb)
        rs = spearman(va, vb)
        lo, hi, nb = boot_ci(va, vb)
        log("  %-22s  r = %+.4f  [%+.4f, %+.4f]   Spearman %+.4f" % (tag, rp, lo, hi, rs))
        out.append(
            {
                "pair": "%s×%s" % (a_name, b_name),
                "quantity": tag,
                "n": len(j),
                "pearson": rp,
                "ci_lo": lo,
                "ci_hi": hi,
                "spearman": rs,
                "n_boot_valid": nb,
            }
        )
    log("")
    # ECE 는 구간별 부호 있는 평균잔차로 만들어진다. 그래서 두 ECE
    # 추정치의 의존성을 재는 올바른 양은 부호 있는 잔차 상관이다.
    # (절대잔차 상관은 오차 '크기' 의 동조만 재므로 의존성을 과소평가한다.)
    r_sgn, lo_s, hi_s = out[0]["pearson"], out[0]["ci_lo"], out[0]["ci_hi"]
    r_abs = out[1]["pearson"]

    def neff(r):
        return 2.0 / (1.0 + r) if (1.0 + r) != 0 else float("nan")

    log("  [읽는 법] 두 결과가 '독립 재현' 인지는 **부호 있는 잔차** 상관으로")
    log("            판단한다 — ECE 가 구간별 부호 있는 평균잔차로 만들어지기")
    log("            때문이다.")
    log("              부호 있는 잔차 r = %+.4f [%+.4f, %+.4f]" % (r_sgn, lo_s, hi_s))
    sig = (
        "신뢰구간이 0 을 포함하지 않는다 — 상관이 있다"
        if (lo_s > 0 or hi_s < 0)
        else "신뢰구간이 0 을 포함한다 — 상관을 단정할 수 없다"
    )
    log("              → %s." % sig)
    log("            두 추정치를 평균하면 분산이 하나의 (1+r)/2 = %.3f 배," % ((1 + r_sgn) / 2))
    log("            즉 유효 표본 수 2/(1+r) = %.2f 개 데이터셋에 해당한다." % neff(r_sgn))
    log("            신뢰구간 상한 r=%+.4f 을 쓰면 %.2f 개 (보수적 하한)." % (hi_s, neff(hi_s)))
    log("            참고 — 절대잔차(오차 크기) 상관 r = %+.4f → %.2f 개." % (r_abs, neff(r_abs)))
    log("            이 근사는 두 추정치의 분산이 같다고 가정한다.")
    log("")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50, help="C·D 대조군 반복 수 (기본 50)")
    a = ap.parse_args()

    log("개념 이동 검정 — P(y|x) 가 시대에 따라 변했는가")
    log("대조군 반복 %d회, fold %d개, 시드 %d, 부트스트랩 %d회" % (a.reps, N_FOLDS, S.SEED, N_BOOT))
    log("")

    rows, store = [], {}
    for name, fn in S.DATASETS:
        rr, st = run(name, fn, a.reps)
        rows += rr
        store[name] = st

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "concept_shift.csv"), index=False, encoding="utf-8")

    ov = overlap_analysis(store)
    if len(ov):
        ov.to_csv(os.path.join(RES, "concept_shift_overlap.csv"), index=False, encoding="utf-8")

    log("=" * 104)
    log("판정")
    log("=" * 104)
    log("[읽는 법] 팔 A 와 팔 D 는 학습 크기·학습 양성률·검증군이 모두 같고")
    log("          학습 분자의 **시대만** 다르다. D 의 ECE 가 A 보다 크고 p 가")
    log("          작으면 P(y|x) 가 변했다는 뜻 — 보정 붕괴가 표본 크기나")
    log("          유병률만으로 설명되지 않는다. 반대로 A 와 D 가 비슷하면")
    log("          남은 원인은 크기·유병률이고, 본문의 기전 서술을 빼야 한다.")
    log("          팔 A2(스캐폴드 분리)가 A 와 비슷해야 유사체 누출이 아니다.")
    log("          p 는 교환가능성을 가정한 순열검정이 아니라 **참조분포 비교**다:")
    log("          재표집 대조군 중 팔 A 만큼(이하로) 잘 보정된 비율의")
    log("          Phipson-Smyth 추정량이다. 효과 크기(ECE 차)를 먼저 읽어야 한다.")
    log("")
    for d in out.dataset.unique():
        g = out[out.dataset == d].set_index("arm")
        a1 = g.loc["A. 시대일치 무작위 5-fold"]
        log("  %s" % d)
        log(
            "    A  시대일치        ECE %.4f  AUROC %.4f  기울기 %.4f"
            % (a1.ece, a1.auroc, a1.slope)
        )
        if "A2. 시대일치 스캐폴드 5-fold" in g.index:
            a2 = g.loc["A2. 시대일치 스캐폴드 5-fold"]
            log("    A2 스캐폴드 분리   ECE %.4f  (A 대비 %+.4f)" % (a2.ece, a2.ece - a1.ece))
        for k in ("C. 정방향 크기맞춤", "D. 정방향 크기·유병률맞춤"):
            if k not in g.index:
                continue
            rr = g.loc[k]
            log(
                "    %-16s ECE %.4f  A 대비 배율 %.2f  p %.4f  "
                "(대조군 %d/%d 회가 A 이상으로 보정)"
                % (
                    k.split(". ")[1],
                    rr.ece,
                    rr.ece_ratio,
                    rr.p_phipson_smyth,
                    int(rr.n_better),
                    int(rr.n_rep_valid),
                )
            )
        b = g.loc["B. 정방향 전체"]
        log("    B  정방향 전체     ECE %.4f  (§8 설정, 학습 n %d)" % (b.ece, int(b.train_n)))
        have_cd = "C. 정방향 크기맞춤" in g.index and "D. 정방향 크기·유병률맞춤" in g.index
        if have_cd:
            c = g.loc["C. 정방향 크기맞춤"]
            d_ = g.loc["D. 정방향 크기·유병률맞춤"]
            d_era = d_.ece - a1.ece  # A → D : 시대만 바뀐다
            d_rate = c.ece - d_.ece  # D → C : 학습 양성률만 바뀐다
            d_size = b.ece - c.ece  # C → B : 학습 크기만 바뀐다
            tot = b.ece - a1.ece  # 합은 정확히 B - A 다
            log("    A→D→C→B 3분해 (합 = B-A = %+.4f)" % tot)
            log("      시대(개념) A→D   %+.4f   %s" % (d_era, pct(d_era, tot)))
            log("      학습 유병률 D→C  %+.4f   %s" % (d_rate, pct(d_rate, tot)))
            log("      학습 크기 C→B    %+.4f   %s" % (d_size, pct(d_size, tot)))
            log("      [주의] 세 몫은 이 경로 순서에서의 몫이다. 순서를 바꾸면")
            log("             교호작용이 다르게 배분된다 (합은 언제나 B-A).")
        log("")

    log("저장: %s" % os.path.join(RES, "concept_shift.csv"))
    if len(ov):
        log("      %s" % os.path.join(RES, "concept_shift_overlap.csv"))
    log("      %s" % os.path.join(RES, "concept_shift_summary.txt"))
    log.save(Path(RES) / "concept_shift_summary.txt")


if __name__ == "__main__":
    main()
