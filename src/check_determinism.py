# -*- coding: utf-8 -*-
"""격자 모델 4종이 같은 입력에서 같은 출력을 내는지 비트 단위로 확인한다.

왜 필요한가
----------
2026-09-30 감사에서 §9.1(robustness.py)과 §9.5(grid_calibration.py)의
"32조합"이 실제로는 다른 격자였음이 드러났다. 원인은 두 파일이 같은 이름의
모델을 각자 정의해 하이퍼파라미터가 달랐던 것이고(RandomForest
min_samples_leaf, LightGBM num_leaves 대 max_depth, LogisticRegression
max_iter), 정의는 _shared.grid_models() 로 합쳤다.

남은 위험은 같은 정의라도 실행마다 값이 달라지는 것이다. LightGBM 은
스레드 수에 따라 결과가 변하므로 deterministic=True, force_row_wise=True,
n_jobs=1 을 준다. 이 스크립트는 그 가정을 매번 실제로 검사한다.

검사 방법
--------
robustness.py 가 실제로 쓰는 적재 함수(load)와 표현 함수(rep_ecfp_desc),
그리고 _shared.grid_models() 를 그대로 불러, 같은 학습/검증 분할에서
모델을 두 번 적합해 예측확률이 비트 단위로 같은지 본다. 같은 정의를
쓰는지도 함께 확인한다(robustness.models() 와 grid_calibration 의 모델이
같은 객체를 만드는지).

    python src/check_determinism.py

종료코드 0 = 모두 결정적, 1 = 하나라도 실행마다 다름.

2026-10-20 추가 — 격자 모델만 보던 것을 재보정 사다리(§10) 까지 넓혔다.
§10.5 를 §10 과 대조하다가 Platt 의 ECE 가 실행마다 3.6e-05 흔들리는 것을
발견했는데, 이 검사가 그 경로를 보지 않아 놓치고 있었다. 흔들림이 보고
자리수(소수 4자리) 안이면 실패로 올리지 않고 results/determinism.csv 에
크기를 기록한다.
"""

import os
import sys

# 스레드를 1 로 고정한다. 단, --unpinned 로 불렸을 때는 고정하지 않는다 —
# [5] 가 "실제 분석이 돌아간 조건"(스레드 고정 없음)에서의 재현성을 재기
# 위해 이 파일을 자식 프로세스로 다시 부르기 때문이다. 고정한 채로 재면
# 분석 스크립트가 실제로 겪는 흔들림을 못 본다 (2026-10-20).
if "--unpinned" not in sys.argv:
    for _v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_v, "1")

import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared as S
import robustness as RB
import grid_calibration as GC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

fails = []


def check_single_definition():
    """두 스크립트가 같은 정의를 쓰는지 (파라미터 문자열 비교)."""
    print("[1] 모델 정의 단일성")
    a = RB.models()
    b = GC.models() if hasattr(GC, "models") else GC.MODELS
    if callable(b):
        b = b()
    ka, kb = sorted(a), sorted(b)
    if ka != kb:
        fails.append("모델 목록이 다르다: robustness %s vs grid %s" % (ka, kb))
        print("    *** 모델 목록 불일치 ***")
        return
    for k in ka:
        pa = repr(a[k]().get_params())
        pb = repr(b[k]().get_params())
        ok = pa == pb
        print("    %-14s %s" % (k, "동일" if ok else "*** 다르다 ***"))
        if not ok:
            fails.append("%s 하이퍼파라미터가 두 스크립트에서 다르다" % k)
    print()


def check_single_reps():
    """두 스크립트의 표현이 같은 행렬을 만드는지 (dtype 까지) 확인한다.

    2026-10-01: 모델 정의를 합친 뒤에도 32칸 중 12칸이 어긋났다. 원인은
    표현 함수의 dtype 이었다 (robustness float32 / grid int8+float64).
    값은 같아도 정밀도가 다르면 LogisticRegression 과 LightGBM 이 달라진다.
    """
    from rdkit import Chem

    print("[1b] 표현 정의 단일성 (dtype 포함)")
    df = RB.load(PROC / S.DATASETS[0][1]).head(80)
    smi = list(df["smiles"])
    mols = [Chem.MolFromSmiles(x) for x in smi]
    a, b = RB.REPS, GC.REPS
    if sorted(a) != sorted(b):
        fails.append("표현 목록이 다르다: %s vs %s" % (sorted(a), sorted(b)))
        print("    *** 표현 목록 불일치 ***")
        return
    for k in sorted(a):
        Xa, Xb = a[k](mols), b[k](mols)
        ok = Xa.dtype == Xb.dtype and Xa.shape == Xb.shape and np.array_equal(Xa, Xb)
        print("    %-12s %-9s %s" % (k, str(Xa.dtype), "동일" if ok else "*** 다르다 ***"))
        if not ok:
            fails.append(
                "표현 %s 가 두 스크립트에서 다르다 "
                "(dtype %s vs %s, shape %s vs %s)" % (k, Xa.dtype, Xb.dtype, Xa.shape, Xb.shape)
            )
    # 주 분석(featurize)과도 같은가 — 격자의 ECFP4+desc 칸이 주 분석 값과
    # 같아야 한다는 주장의 근거다.
    Xg, Xf = a["ECFP4+desc"](mols), S.featurize(smi)
    ok = Xg.dtype == Xf.dtype and np.array_equal(Xg, Xf)
    print("    %-12s %-9s %s" % ("vs 주 분석", str(Xf.dtype), "동일" if ok else "*** 다르다 ***"))
    if not ok:
        fails.append(
            "격자의 ECFP4+desc 가 주 분석 featurize() 와 다르다 "
            "(dtype %s vs %s)" % (Xg.dtype, Xf.dtype)
        )
    print()


def check_metric_definitions():
    """같은 이름의 지표 함수가 여러 파일에 복제돼 있다. 전부 같은가.

    파일에서 해당 def 블록을 떼어 numpy 만 있는 이름공간에서 실행하므로
    sklearn·rdkit 가 없어도 (calib_slope 만 sklearn 이 필요하다) 돈다.
    입력에는 구간 경계에 정확히 걸리는 값을 섞는다 — 경계 처리가
    다르면 거기서 갈린다.

    2026-10-21 확장. 처음에는 ece 만 봤는데, 실제로 복제된 것은 여섯
    종류였다 — ece 7곳, calib_slope 3곳, bin_table 3곳,
    murphy_decomposition 3곳, subsample_to_rate 2곳(decompose 는 한
    곳뿐이다). 특히 subsample_to_rate 는 어떤 행을 고를지 정하는
    함수라서 두 사본이 갈라지면 지표가 아니라 유병률 맞춤 대조군의
    구성 자체가 달라진다. 그래서 값이 아니라 뽑힌 인덱스 배열을
    비교한다.
    """
    import glob
    import re

    import pandas as pd  # calibration.ece_mce 가 DataFrame 을 만든다
    from sklearn.linear_model import LogisticRegression

    print("[1c] 지표 정의 단일성 (복제된 정의가 전부 같은가, 비트 단위)")
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.4).astype(int)
    p = rng.random(500)
    p[:20] = np.linspace(0, 1, 20)  # 구간 경계 포함
    bins_fixed = S.bin_table(y, p)
    idx_all = np.arange(500)

    def _rng():
        return np.random.default_rng(7)  # 사본마다 **같은** 난수열

    # (이름, 호출, 기준값, 비교) — 비교는 비트 단위 동일성을 뜻한다.
    specs = [
        ("ece", lambda f: float(f(y, p)), float(S.ece(y, p)), lambda a, b: a == b, "%.17g"),
        (
            "calib_slope",
            lambda f: float(f(y, p)),
            float(S.calib_slope(y, p)),
            lambda a, b: a == b,
            "%.17g",
        ),
        (
            "bin_table",
            lambda f: tuple(map(tuple, f(y, p))),
            tuple(map(tuple, bins_fixed)),
            lambda a, b: a == b,
            None,
        ),
        (
            "murphy_decomposition",
            lambda f: tuple(sorted((k, float(v)) for k, v in f(bins_fixed).items())),
            tuple(sorted((k, float(v)) for k, v in S.murphy_decomposition(bins_fixed).items())),
            lambda a, b: a == b,
            None,
        ),
        # 유병률 세 값을 쓴다. 이 입력(양성 178 / 음성 322)에서
        # 0.37·0.63 은 n_neg_a 분기, 0.17 은 n_pos_b 분기를 지나고,
        # 0.17 과 0.63 은 몫이 정수가 아니라 반올림 처리의 차이까지
        # 드러난다. 0.2·0.3·0.5·0.8 처럼 몫이 딱 맞는 값만 넣으면
        # round 를 int 로 바꿔치기해도 결과가 같아 못 잡는다 (2026-10-21).
        (
            "subsample_to_rate",
            lambda f: tuple(
                np.asarray(f(y, idx_all, t, _rng())).tobytes() for t in (0.17, 0.37, 0.63)
            ),
            tuple(
                np.asarray(S.subsample_to_rate(y, idx_all, t, _rng())).tobytes()
                for t in (0.17, 0.37, 0.63)
            ),
            lambda a, b: a == b,
            None,
        ),
    ]

    ns_base = {"np": np, "N_BINS": S.N_BINS, "EPS": S.EPS, "LogisticRegression": LogisticRegression}
    files = sorted(glob.glob(str(ROOT / "src" / "*.py")))
    n_defs = 0
    dup_rows = []
    for name, call, ref, same_fn, fmt in specs:
        pat = re.compile(r"\ndef %s\(.*?\n(?=\n\ndef |\n\nclass |\Z)" % name, re.S)
        found = []
        for f in files:
            src = Path(f).read_text(encoding="utf-8")
            m = pat.search(src)
            if not m:
                continue
            base = os.path.basename(f)
            g = dict(ns_base)
            try:
                exec(m.group(0), g)
                found.append((base, call(g[name])))
            except Exception as e:
                fails.append("%s 의 %s 정의를 떼어 실행할 수 없다: %s" % (base, name, e))
        n_defs += len(found)
        n_same = sum(1 for _b, _v in found if same_fn(_v, ref))
        dup_rows.append(
            {
                "metric": name,
                "n_defs": len(found),
                "n_identical": n_same,
                "all_identical": bool(n_same == len(found)),
            }
        )
        print("    %s — 정의 %d개  (기준 _shared.%s)" % (name, len(found), name))
        if len(found) < 2:
            print("        복제 없음 — 비교할 것이 없다")
        for base, val in found:
            ok = same_fn(val, ref)
            shown = (fmt % val) if fmt else "—"
            print("        %-32s %-24s %s" % (base, shown, "동일" if ok else "*** 다르다 ***"))
            if not ok:
                fails.append("%s 의 %s 가 _shared.%s 와 비트 단위로 다르다" % (base, name, name))
    print("    복제 정의 총 %d개 확인" % n_defs)

    print()
    # 같은 값을 다른 이름으로 계산하는 사본. 이름으로만 찾으면
    # 빠진다 — calibration.calib_slope_intercept 가 그래서 10-20
    # 정밀도 수정에서 누락됐다 (2026-10-21).
    aliases = [
        ("calibration.py", "ece_mce", lambda f: float(f(y, p)[0]), float(S.ece(y, p)), "ece"),
        (
            "calibration.py",
            "calib_slope_intercept",
            lambda f: float(f(y, p)[0]),
            float(S.calib_slope(y, p)),
            "calib_slope",
        ),
    ]
    print("    별명 복제 (이름은 다르지만 같은 값을 내야 하는 것)")
    for base, name, call, ref, refname in aliases:
        src = (ROOT / "src" / base).read_text(encoding="utf-8")
        m = re.search(r"\ndef %s\(.*?\n(?=\n\ndef |\n\nclass |\Z)" % name, src, re.S)
        if not m:
            fails.append("%s 에서 %s 정의를 찾지 못했다" % (base, name))
            continue
        g = dict(ns_base)
        g["pd"] = pd
        try:
            exec(m.group(0), g)
            val = call(g[name])
        except Exception as e:
            fails.append("%s.%s 를 떼어 실행할 수 없다: %s" % (base, name, e))
            continue
        ok = val == ref
        print(
            "        %-22s %s -> %-24s %s"
            % (base + "." + name, refname, "%.17g" % val, "동일" if ok else "*** 다르다 ***")
        )
        dup_rows.append(
            {
                "metric": "%s (별명: %s)" % (name, refname),
                "n_defs": 1,
                "n_identical": int(ok),
                "all_identical": bool(ok),
            }
        )
        if not ok:
            fails.append(
                "%s.%s 가 _shared.%s 와 비트 단위로 다르다 "
                "(%.17g vs %.17g)" % (base, name, refname, val, ref)
            )
    # 보고서가 이 숫자들을 산문에 직접 적지 않도록 CSV 로 남긴다.
    pd.DataFrame(dup_rows, columns=["metric", "n_defs", "n_identical", "all_identical"]).to_csv(
        ROOT / "results" / "metric_duplicates.csv", index=False
    )
    print("    기록: results/metric_duplicates.csv  (%d행)" % len(dup_rows))
    print()


def check_precision():
    """로짓·보정기울기 계산이 입력 dtype 에 상관없이 float64 인가.

    왜 필요한가 — 2026-10-20. XGBoost 의 predict_proba 는 float32 를
    돌려준다. 로짓 변환이 입력을 캐스팅하지 않으면 그 뒤의 로지스틱 적합이
    단정밀도로 일어나고, 같은 지표가 어디서 왔는지에 따라 정밀도가
    달라진다. 실제로 §10 의 Platt 계수가 §10.5 와 2.1e-04 어긋나 있었고
    (src/diff_ladder.py 단계 10 이 찾아냈다), 원인을 두 번 엉뚱하게
    추측한 뒤에야 찾았다. 캐스팅 누락은 다음 일곱 군데였다 —
    recalibration.to_logit·calib_line, recalibration_sensitivity.logit,
    tiebreak_sensitivity.to_logit, _shared.calib_slope,
    prevalence_control.calib_slope, robustness.calib_slope.

    검사 방법 — float32 배열 p 와 그것을 미리 float64 로 넓힌 배열을
    넣어 결과를 비교한다. 두 입력은 같은 수이므로, 함수가 내부에서
    float64 로 올리면 결과가 비트 단위로 같고, 올리지 않으면 앞쪽만
    단정밀도로 계산되어 달라진다.

    (처음에는 float32 와 별개의 float64 난수를 넣어 비교했는데, 그 둘은
    애초에 다른 수여서 캐스팅이 있어도 달라진다 — 검사가 전부 실패로
    나왔다. 2026-10-20 수정.)
    """
    import glob
    import re

    print("[1d] 정밀도 — float32 입력과 float64 입력이 같은 결과를 내는가")
    rng = np.random.default_rng(1)
    p32 = (rng.random(300) * 0.998 + 0.001).astype(np.float32)
    p64 = p32.astype(np.float64)  # **같은 수**를 넓힌 것
    y = (rng.random(300) < 0.45).astype(int)
    n_ok = 0
    for f in sorted(glob.glob(str(ROOT / "src" / "*.py"))):
        src = Path(f).read_text(encoding="utf-8")
        # calib_slope_intercept 는 10-20 에 빠졌다 — 이름이 달라서
        # 목록에 안 걸렸고, 실제로 캐스팅이 없었다 (2026-10-21).
        for nm in ("to_logit", "logit", "calib_slope", "calib_line", "calib_slope_intercept"):
            m = re.search(r"\ndef %s\(.*?\n(?=\n\ndef |\n\nclass |\Z)" % nm, src, re.S)
            if not m:
                continue
            g = {"np": np, "EPS": S.EPS, "N_BINS": S.N_BINS}
            g["S"] = S
            try:
                from sklearn.linear_model import LogisticRegression

                g["LogisticRegression"] = LogisticRegression
                exec(m.group(0), g)
                fn = g[nm]
                a = fn(y, p32) if nm.startswith("calib") else fn(p32)
                b = fn(y, p64) if nm.startswith("calib") else fn(p64)
            except Exception as e:
                fails.append("%s / %s 를 떼어 실행할 수 없다: %s" % (os.path.basename(f), nm, e))
                continue
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            same = np.array_equal(a, b)
            mx = float(np.max(np.abs(a - b))) if a.size else 0.0
            n_ok += 1
            print(
                "    %-32s %-11s %s  최대차 %.3e"
                % (os.path.basename(f), nm, "동일" if same else "*** 다르다 ***", mx)
            )
            if not same:
                fails.append(
                    "%s / %s 가 float32 입력을 내부에서 float64 로 "
                    "올리지 않는다 (최대차 %.3e) — 캐스팅을 넣을 것" % (os.path.basename(f), nm, mx)
                )
    print("    확인한 함수 %d개" % n_ok)
    print()


def check_determinism():
    print("[2] 두 번 적합 결과 비트 단위 동일성")
    for ds, fn in S.DATASETS:
        df = RB.load(PROC / fn)
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
        keep = [i for i, m in enumerate(mols) if m is not None]
        if len(keep) != len(mols):
            fails.append("%s: SMILES 파싱 실패 %d건" % (ds, len(mols) - len(keep)))
        df = df.iloc[keep].reset_index(drop=True)
        mols = [mols[i] for i in keep]
        X = RB.rep_ecfp_desc(mols)
        y = df["y"].values
        tr, te = S.temporal_masks(df)
        print("    %s  n=%d  학습 %d / 검증 %d" % (ds, len(df), tr.sum(), te.sum()))
        from sklearn.metrics import roc_auc_score

        for mname, mk in S.grid_models().items():
            p1 = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            p2 = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            same = np.array_equal(p1, p2)
            mx = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0
            # 실제로 보고되는 값도 같은지 본다. 예측확률이 마지막 비트에서
            # 달라도 ECE 는 구간 경계에서 증폭될 수 있으므로 따로 확인한다.
            yt = y[te]
            m1 = (roc_auc_score(yt, p1), S.ece(yt, p1), S.calib_slope(yt, p1))
            m2 = (roc_auc_score(yt, p2), S.ece(yt, p2), S.calib_slope(yt, p2))
            msame = all(a == b for a, b in zip(m1, m2))
            print(
                "        %-14s 예측 최대차 %.3e  %s | AUROC/ECE/기울기 %s"
                % (
                    mname,
                    mx,
                    "동일" if same else "*** 다르다 ***",
                    "동일" if msame else "*** 다르다 ***",
                )
            )
            if not same:
                fails.append("%s / %s 예측이 실행마다 다르다 (최대차 %.3e)" % (ds, mname, mx))
            if not msame:
                fails.append(
                    "%s / %s 의 보고 지표가 실행마다 다르다 "
                    "(AUROC %.12f/%.12f, ECE %.12f/%.12f, "
                    "기울기 %.12f/%.12f)" % (ds, mname, m1[0], m2[0], m1[1], m2[1], m1[2], m2[2])
                )
        print()


def check_recalibration():
    """재보정 사다리(§10)가 실행마다 같은 값을 내는지.

    왜 따로 보는가 — 2026-10-20 에 §10.5 를 §10 과 대조하면서, 입력이 비트
    단위로 같은데도 Platt 의 ECE 만 3.6e-05 어긋나는 것을 발견했다.
    A·E·Isotonic 은 1e-17 로 일치했다. 즉 흔들리는 것은 모델도 분할도
    아니고 `LogisticRegression(penalty=None, solver="lbfgs")` 가 닿는 해다.
    그런데 이 검사는 격자 모델만 보고 있어서 그 사실을 잡지 못했다.

    여기서는 §10 의 경로(전체학습 → 보정셋 → Platt·Isotonic·사전확률·
    Oracle)를 두 번 돌려 보고 지표가 같은지 본다. 어긋나면 그 크기를
    기록한다 — 1e-4 미만이면 보고 자리수(소수 4자리)에 영향이 없으므로
    실패로 올리지 않고 결과 파일에 남긴다. 그보다 크면 실패다.

    이 검사가 무엇을 못 보는지 적어 둔다 — 한 프로세스 안에서 두 번
    돌리는 것이므로 BLAS 스레드 상태가 같다. 실행(프로세스)을 건너뛰는
    흔들림은 여기서 0 으로 나올 수 있다. 그쪽은 `extra_calibrators.py` 가
    저장된 `recalibration_results.csv` 와 대조해서 잡는다(그 대조에서
    3.6e-05 이 나왔다). 그러므로 결과는 이렇게 읽는다.

        여기 0, extra_calibrators 0        어느 쪽으로도 결정적이다
        여기 0, extra_calibrators 3.6e-05  흔들림이 실행 간에만 있다
        여기 >0                            한 프로세스 안에서도 흔들린다
    """
    import pandas as pd
    import recalibration as RC

    print("[3] 재보정 사다리(§10) 실행 간 동일성")
    rows = []
    for ds, fn in S.DATASETS:
        # recalibration.py 의 run() 과 같은 적재·정렬·분할·특성화를 쓴다.
        # 그 파일은 적재를 run() 안에서 하므로 여기서 그대로 옮겨 적는다
        # (옮겨 적은 것이 어긋나면 아래 ECE 가 §10 과 달라져 바로 드러난다).
        df = pd.read_csv(PROC / fn)
        df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
        df["approval_year"] = df["approval_year"].astype(int)
        df["y"] = df["y"].astype(int)
        df = df.sort_values("approval_year").reset_index(drop=True)
        X = RC.featurize(df["smiles"].tolist())
        y = df["y"].values
        yr = df["approval_year"].values
        tr_all = yr < RC.CUTOFF
        te = ~tr_all
        tr_idx = np.where(tr_all)[0]
        n_cal = int(len(tr_idx) * RC.CALIB_FRAC)
        fit_idx, cal_idx = tr_idx[:-n_cal], tr_idx[-n_cal:]
        yte = y[te]

        def once():
            m_full = RC.make_model().fit(X[tr_all], y[tr_all])
            p_base = m_full.predict_proba(X[te])[:, 1]
            m_fit = RC.make_model().fit(X[fit_idx], y[fit_idx])
            p_cal = m_fit.predict_proba(X[cal_idx])[:, 1]
            p_te_fit = m_fit.predict_proba(X[te])[:, 1]
            return {
                "A": p_base,
                "A'": p_te_fit,
                "B": RC.platt(p_cal, y[cal_idx], p_te_fit),
                "C": RC.isotonic(p_cal, y[cal_idx], p_te_fit),
                "D": RC.prior_shift(p_base, y[tr_all].mean(), yte.mean()),
                "E": RC.platt(p_base, yte, p_base),
            }

        r1, r2 = once(), once()
        for k in ("A", "A'", "B", "C", "D", "E"):
            dp = float(np.max(np.abs(r1[k] - r2[k])))
            e1, e2 = RC.ece(yte, r1[k]), RC.ece(yte, r2[k])
            de = abs(e1 - e2)
            tag = (
                "동일"
                if dp == 0.0 and de == 0.0
                else ("보고 자리수 안에서 같다" if de < 1e-4 else "*** 보고 자리수가 달라진다 ***")
            )
            print(
                "    %s %-3s 예측 최대차 %.3e  ECE %.6f/%.6f 차 %.3e  %s"
                % (ds, k, dp, e1, e2, de, tag)
            )
            rows.append(
                dict(
                    dataset=ds,
                    arm=k,
                    scope="within_process",
                    max_pred_diff=dp,
                    ece_run1=e1,
                    ece_run2=e2,
                    ece_diff=de,
                    within_report_digits=de < 1e-4,
                )
            )
            if de >= 1e-4:
                fails.append(
                    "%s / %s 의 ECE 가 실행마다 보고 자리수에서 " "다르다 (차 %.3e)" % (ds, k, de)
                )
        print()
    pd.DataFrame(rows).to_csv(ROOT / "results" / "determinism.csv", index=False)
    print("    저장: results/determinism.csv")
    print()


def ladder_once(ds, fn):
    """§10 의 사다리를 한 번 계산해 여섯 팔의 예측을 돌려준다."""
    import pandas as pd
    import recalibration as RC

    df = pd.read_csv(PROC / fn)
    df = df.dropna(subset=["approval_year", "y", "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df["y"] = df["y"].astype(int)
    df = df.sort_values("approval_year").reset_index(drop=True)
    X = RC.featurize(df["smiles"].tolist())
    y = df["y"].values
    yr = df["approval_year"].values
    tr_all = yr < RC.CUTOFF
    te = ~tr_all
    tr_idx = np.where(tr_all)[0]
    n_cal = int(len(tr_idx) * RC.CALIB_FRAC)
    fit_idx, cal_idx = tr_idx[:-n_cal], tr_idx[-n_cal:]
    yte = y[te]
    m_full = RC.make_model().fit(X[tr_all], y[tr_all])
    p_base = m_full.predict_proba(X[te])[:, 1]
    m_fit = RC.make_model().fit(X[fit_idx], y[fit_idx])
    p_cal = m_fit.predict_proba(X[cal_idx])[:, 1]
    p_te_fit = m_fit.predict_proba(X[te])[:, 1]
    return yte, {
        "A": p_base,
        "A'": p_te_fit,
        "B": RC.platt(p_cal, y[cal_idx], p_te_fit),
        "C": RC.isotonic(p_cal, y[cal_idx], p_te_fit),
        "D": RC.prior_shift(p_base, y[tr_all].mean(), yte.mean()),
        "E": RC.platt(p_base, yte, p_base),
    }


def check_cross_process():
    """프로세스를 달리해 두 번 계산한 결과가 같은가 (스레드 고정 없이).

    왜 이것이 따로 필요한가 — [2]·[3] 은 한 프로세스 안에서 두 번 돌리고,
    이 파일은 맨 위에서 스레드를 1 로 고정한다. 그래서 두 검사는 실제
    분석 스크립트가 돌아간 조건(n_jobs=-1, 스레드 고정 없음)에서의
    재현성을 재지 못한다.

    실제로 2026-10-20 에 §10.5 를 §10 과 대조했을 때 DICTrank Platt 의
    ECE 가 3.6e-05 어긋났다. 적재·분할·특성화·모델 정의가 모두 같고
    ([1]·[1b]·[1c] 가 확인한다) 한 프로세스 안에서는 비트 단위로 같으므로
    ([2]·[3]), 남는 설명은 실행(프로세스) 간 다중스레드 누적 순서다.
    이 검사가 그것을 직접 재서 그 설명이 맞는지 확인한다.

    같으면 → 흔들림의 원인은 다른 데 있다 (설명을 다시 찾아야 한다).
    다르면 → 보고 자리수에 영향이 있는지 그 크기로 판단한다.
    """
    import subprocess
    import tempfile
    import pandas as pd

    print("[5] 프로세스를 달리해 두 번 (스레드 고정 없음)")
    env = dict(os.environ)
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.pop(v, None)
    rows = []
    for ds, fn in S.DATASETS:
        outs = []
        for run in (1, 2):
            tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
            tmp.close()
            r = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--unpinned",
                    "--emit-ladder",
                    tmp.name,
                    "--dataset",
                    ds,
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                fails.append("%s 자식 프로세스 실패: %s" % (ds, r.stderr[-300:]))
                outs = []
                break
            outs.append(np.load(tmp.name, allow_pickle=False))
            os.unlink(tmp.name)
        if len(outs) != 2:
            continue
        a, b = outs
        yte = a["y"]
        for k in ("A", "A'", "B", "C", "D", "E"):
            dp = float(np.max(np.abs(a[k] - b[k])))
            e1, e2 = float(S.ece(yte, a[k])), float(S.ece(yte, b[k]))
            de = abs(e1 - e2)
            tag = (
                "동일"
                if dp == 0.0 and de == 0.0
                else ("보고 자리수 안에서 같다" if de < 1e-4 else "*** 보고 자리수가 달라진다 ***")
            )
            print("    %s %-3s 예측 최대차 %.3e  ECE 차 %.3e  %s" % (ds, k, dp, de, tag))
            rows.append(
                dict(
                    dataset=ds,
                    arm=k,
                    scope="cross_process",
                    max_pred_diff=dp,
                    ece_run1=e1,
                    ece_run2=e2,
                    ece_diff=de,
                    within_report_digits=de < 1e-4,
                )
            )
            if de >= 1e-4:
                fails.append(
                    "%s / %s 의 ECE 가 프로세스 간에 보고 자리수에서 "
                    "다르다 (차 %.3e)" % (ds, k, de)
                )
        print()
    if rows:
        p = ROOT / "results" / "determinism_cross_process.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        print("    저장: results/determinism_cross_process.csv")
    print()


def main():
    if "--emit-ladder" in sys.argv:
        # 자식 프로세스 모드 — 사다리 한 번 계산해 npz 로 떨어뜨린다.
        path = sys.argv[sys.argv.index("--emit-ladder") + 1]
        want = sys.argv[sys.argv.index("--dataset") + 1]
        for ds, fn in S.DATASETS:
            if ds == want:
                yte, d = ladder_once(ds, fn)
                np.savez(path, y=yte, **d)
                return
        raise SystemExit("데이터셋을 찾지 못했다: %s" % want)
    print("격자 모델 결정성 검사")
    print("=" * 62)
    print("python %s  numpy %s" % (sys.version.split()[0], np.__version__))
    print()
    check_single_definition()
    check_single_reps()
    check_metric_definitions()
    check_precision()
    check_determinism()
    check_recalibration()
    check_cross_process()
    print("실패 %d건" % len(fails))
    for f in fails:
        print("  X", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
