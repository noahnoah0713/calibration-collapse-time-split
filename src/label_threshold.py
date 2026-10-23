"""라벨 이진화 민감도 — 결론이 라벨을 어떻게 잘랐느냐의 산물인가.

왜 필요한가 (가장 급한 민감도)
------------------------------
두 원자료는 모두 3단계 라벨이다.

    DILIrank   vNo 334 / vLess 321 / vMost 205
    DICTrank   no  253 / less  413 / most  253

주 분석은 less 와 most 를 함께 양성으로 묶었다(양성률 0.61 / 0.72).
문헌에서 흔히 쓰는 다른 선택은 most 만 양성 이고, 그러면 양성률이
0.24 / 0.28 로 절반 이하로 떨어진다.

이 논문의 중심 주장은 "학습·검증 유병률 격차가 보정오차를 만든다" 이다.
그런데 그 유병률을 정하는 것이 바로 이 이진화 선택이다. 따라서 심사자의
질문은 피할 수 없다 — "당신 결론은 라벨을 어떻게 잘랐느냐의 산물 아닌가."
§16(큐레이션 민감도)은 결정 3건을 다루지만 라벨 역치는 거기 없다.

설계 (주 분석과 같은 파이프라인을 대체 라벨로 반복)
--------------------------------------------------
  라벨 정의 두 가지
    L1  less+most 양성  ← 주 분석
    L2  most 만 양성    ← 대체
  각각에 대해
    T   시간분할
    R1  무작위 (표준)
    R2  무작위 + 검증군 양성률 맞춤
    R3  무작위 + 학습·검증 양성률 맞춤
  → §8 의 ECE 배율과 §16 사다리를 두 라벨에서 나란히 낸다.

읽는 법
  L2 에서도 배율이 1 을 넘고 사다리의 순서(R1 > R2 > R3)가 유지되면,
  결론은 라벨 역치에 강건하다. 무너지면 본문의 주장 범위를 "less 를
  양성으로 보는 정의 아래에서" 로 좁혀야 한다 — 그 경우에도 숨기지 말고
  그대로 적는다.

주의
  L2 는 양성률이 낮아 양성 표본 수가 줄어든다(DILIrank 검증군 양성
  L1 48개 → L2 21개). 그래서 L2 의 구간은 넓고, L1 과 L2 의 배율을
  직접 빼서 비교하면 안 된다. 각자 자기 대조군과 비교한다.

사용법
------
    python src/label_threshold.py              # 대조군 100회
    python src/label_threshold.py --reps 10    # 짧게 점검

출력
----
    results/label_threshold.csv
    results/label_threshold_summary.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
log = S.Log()

# 원자료의 3단계 라벨 열 이름과 '가장 심각' 범주 (데이터셋마다 표기가 다르다)
LABCOL = {
    "DILIrank": ("label", "vMost", ("vLess", "vMost")),
    "DICTrank": ("label", "most", ("less", "most")),
}
N_BOOT = 2000


def phipson_smyth(r, n):
    return (1.0 + r) / (1.0 + n)


def metrics(y, p):
    d = S.decompose(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "ece": S.ece(y, p),
        "slope": S.calib_slope(y, p),
        "rel": d["reliability"],
        "res": d["resolution"],
    }


def evaluate(X, y, tr, te):
    m = S.make_model().fit(X[tr], y[tr])
    return metrics(y[te], m.predict_proba(X[te])[:, 1])


def run(name, fn, reps):
    df = S.load(os.path.join(PROC, fn))
    col, most, both = LABCOL[name]
    assert col in df.columns, "%s 에 %s 열이 없다" % (name, col)
    raw = df[col].astype(str).str.strip()
    # 표기 흔들림(대소문자·공백)을 흡수한다
    low = raw.str.lower()
    y_l1 = df["y"].values  # 주 분석 라벨
    y_l2 = (low == most.lower()).astype(int).values
    # L1 이 실제로 less+most 와 일치하는지 확인한다 (파이프라인 신뢰 검사)
    y_chk = low.isin([b.lower() for b in both]).astype(int).values
    n_mis = int((y_l1 != y_chk).sum())
    X = S.featurize(df["smiles"].tolist())
    tr_m, te_m = S.temporal_masks(df)
    tr, te = np.where(tr_m)[0], np.where(te_m)[0]

    log("=" * 108)
    log("%s   3단계 라벨 분포: %s" % (name, dict(raw.value_counts())))
    log("  L1(주 분석, %s 양성) 과 CSV y 열 불일치: %d개" % ("+".join(both), n_mis))
    log(
        "  L1 전체 양성률 %.4f / L2(%s 만 양성) 전체 양성률 %.4f"
        % (float(y_l1.mean()), most, float(y_l2.mean()))
    )
    log("=" * 108)

    rows = []
    for ltag, y in (("L1 less+most", y_l1), ("L2 most 만", y_l2)):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            log("%s — 한쪽 클래스가 비어 건너뜀" % ltag)
            continue
        rate_te = float(y[te].mean())
        rate_tr = float(y[tr].mean())
        obs = evaluate(X, y, tr, te)
        log("")
        log(
            "%s   학습 양성률 %.4f / 검증 양성률 %.4f (격차 %+.4f) / "
            "검증 양성 %d개" % (ltag, rate_tr, rate_te, rate_te - rate_tr, int(y[te].sum()))
        )
        log(
            "%-34s %8s %8s %8s %8s %8s %8s"
            % ("설정", "AUROC", "ECE", "기울기", "배율", "유효반복", "p")
        )
        log("-" * 94)
        log(
            "%-34s %8.4f %8.4f %8.4f %8s %8s %8s"
            % ("T. 시간분할", obs["auroc"], obs["ece"], obs["slope"], "—", "—", "—")
        )
        rows.append(
            {
                "dataset": name,
                "label_def": ltag,
                "setting": "T. 시간분할",
                "n_train": len(tr),
                "n_test": len(te),
                "rate_train": rate_tr,
                "rate_test": rate_te,
                "n_test_pos": int(y[te].sum()),
                "ece_ratio": np.nan,
                "n_rep_valid": np.nan,
                "n_ge": np.nan,
                "p_phipson_smyth": np.nan,
                **obs,
            }
        )

        for stag, mt, mr in (
            ("R1. 무작위 (표준)", False, False),
            ("R2. 무작위 + 검증군 비율맞춤", True, False),
            ("R3. 무작위 + 학습·검증 비율맞춤", True, True),
        ):
            E, A, SL = [], [], []
            for i in range(reps):
                r = np.random.default_rng(S.SEED + 50000 + i)
                idx = r.permutation(len(y))
                ti, ri = idx[: len(te)], idx[len(te) :]
                if mt:
                    ti = S.subsample_to_rate(y, ti, rate_te, r)
                if mr:
                    ri = S.subsample_to_rate(y, ri, rate_tr, r)
                if len(np.unique(y[ti])) < 2 or len(np.unique(y[ri])) < 2:
                    continue
                if len(ti) < 30 or len(ri) < 100:
                    continue
                e = evaluate(X, y, ri, ti)
                E.append(e["ece"])
                A.append(e["auroc"])
                SL.append(e["slope"])
            if not E:
                log("%-34s 유효 반복 0회 — 건너뜀" % stag)
                continue
            E = np.array(E)
            n_ge = int((E >= obs["ece"]).sum())
            pv = phipson_smyth(n_ge, len(E))
            ratio = obs["ece"] / float(E.mean())
            log(
                "%-34s %8.4f %8.4f %8.4f %8.2f %8s %8.4f"
                % (
                    stag,
                    float(np.mean(A)),
                    float(E.mean()),
                    float(np.mean(SL)),
                    ratio,
                    "%d/%d" % (n_ge, len(E)),
                    pv,
                )
            )
            rows.append(
                {
                    "dataset": name,
                    "label_def": ltag,
                    "setting": stag,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "rate_train": rate_tr,
                    "rate_test": rate_te,
                    "n_test_pos": int(y[te].sum()),
                    "auroc": float(np.mean(A)),
                    "ece": float(E.mean()),
                    "slope": float(np.mean(SL)),
                    "rel": np.nan,
                    "res": np.nan,
                    "ece_ratio": ratio,
                    "n_rep_valid": len(E),
                    "n_ge": n_ge,
                    "p_phipson_smyth": pv,
                }
            )
    log("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    log("라벨 이진화 민감도 — 결론이 라벨 역치의 산물인가")
    log("대조군 반복 %d회, 시드 %d" % (a.reps, S.SEED))
    log("")
    rows = []
    for name, fn in S.DATASETS:
        rows += run(name, fn, a.reps)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "label_threshold.csv"), index=False, encoding="utf-8")

    log("=" * 108)
    log("판정")
    log("=" * 108)
    log("[읽는 법] L2 에서도 배율이 1 을 넘고 R1 > R2 > R3 순서가 유지되면")
    log("          결론은 라벨 역치에 강건하다. 무너지면 주장 범위를 L1 정의")
    log("          아래로 좁혀야 한다. L2 는 양성 표본이 적어 불확실성이 크므로")
    log("          L1 과 L2 의 배율을 직접 빼서 비교하면 안 된다.")
    log("")
    for d in out.dataset.unique():
        for ltag in out[out.dataset == d].label_def.unique():
            g = out[(out.dataset == d) & (out.label_def == ltag)]
            t = g[g.setting.str.startswith("T")]
            if not len(t):
                continue
            t = t.iloc[0]
            log(
                "  %s / %s — 검증 양성 %d개, 격차 %+.4f, 시간분할 ECE %.4f"
                % (d, ltag, int(t.n_test_pos), t.rate_test - t.rate_train, t.ece)
            )
            seq = []
            for b in ("R1", "R2", "R3"):
                rr = g[g.setting.str.startswith(b)]
                if not len(rr):
                    continue
                rr = rr.iloc[0]
                seq.append(float(rr.ece_ratio))
                log(
                    "      %-34s 배율 %.2f  p %.4f  (%d/%d)"
                    % (
                        rr.setting,
                        rr.ece_ratio,
                        rr.p_phipson_smyth,
                        int(rr.n_ge),
                        int(rr.n_rep_valid),
                    )
                )
            if len(seq) == 3:
                log(
                    "      → 배율 전부 1 초과: %s / 사다리 단조감소(R1>R2>R3): %s"
                    % (
                        "예" if all(x > 1 for x in seq) else "**아니오**",
                        "예" if seq[0] > seq[1] > seq[2] else "**아니오**",
                    )
                )
            log("")
    log("저장: %s" % os.path.join(RES, "label_threshold.csv"))
    log("      %s" % os.path.join(RES, "label_threshold_summary.txt"))
    log.save(Path(RES) / "label_threshold_summary.txt")


if __name__ == "__main__":
    main()
