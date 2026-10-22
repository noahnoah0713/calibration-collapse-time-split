"""
보완 분석 공용 모듈 (2026-09-22 추가)

기존 스크립트들이 featurize/make_model/ece 를 각자 복제하고 있어서,
새로 추가하는 보완 분석은 이 모듈 하나를 공유한다.
값이 기존 결과와 일치해야 하므로 상수와 구현을 그대로 옮겼다.
"""

import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

SEED, CUTOFF = 42, 2010
N_BINS = 10
EPS = 1e-6


# ------------------------------------------------------------ 데이터
def load(path, label_col="y"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["approval_year", label_col, "smiles"]).copy()
    df["approval_year"] = df["approval_year"].astype(int)
    df[label_col] = df[label_col].astype(int)
    return df.reset_index(drop=True)


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


def make_model(seed=SEED):
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


# ---------------------------------------------------- 격자 모델 (단일 정의)
# 2026-09-30: robustness.py 와 grid_calibration.py 가 같은 이름의 모델에
# 서로 다른 하이퍼파라미터를 쓰고 있었다 (RandomForest min_samples_leaf,
# LightGBM num_leaves 대 max_depth, LogisticRegression max_iter). 그래서 두
# 절의 "32조합" 이 실제로는 다른 격자였다. 정의를 이 한 곳으로 합쳐
# 구조적으로 재발을 막는다. 두 스크립트는 이 팩토리만 쓴다.
#
# 결정성: LightGBM 은 스레드 수에 따라 결과가 달라지므로
# deterministic=True, force_row_wise=True, n_jobs=1 을 준다.
# XGBoost·RandomForest 는 random_state 고정 시 n_jobs 와 무관하게 재현된다
# (src/check_determinism.py 가 매번 비트 단위로 확인한다).


class DeterministicRF(RandomForestClassifier):
    """예측 누적을 단일 스레드로 고정한 RandomForest.

    2026-09-30 결정성 검사에서 같은 데이터·같은 시드로 두 번 적합했을 때
    예측확률이 실행마다 마지막 비트에서 달라졌다 (DILIrank 최대차 1.1e-16,
    DICTrank 2.2e-16). 원인은 sklearn 의 predict_proba 가 나무별 예측을
    joblib 스레드로 **공유 배열에 덧셈 누적**하기 때문이다. 부동소수 덧셈은
    순서에 따라 마지막 비트가 달라지고, 스레드 완료 순서는 매번 다르다.

    적합(fit)은 나무마다 난수 시드가 정해져 있어 스레드 수와 무관하게 같다.
    그래서 비싼 적합은 병렬로 두고, 값싼 예측만 단일 스레드로 만든다.

    1e-16 은 보고 자리수(소수 4자리)에 영향을 줄 수 없지만, ECE 는 예측값을
    구간으로 나누므로 구간 경계에 정확히 놓인 값 하나가 옆 구간으로 넘어가면
    차이가 증폭될 수 있다. 재현성을 '자리수 안에서 같다' 가 아니라 '비트가
    같다' 로 두는 편이 논문 근거로 안전하다.
    """

    def predict_proba(self, X):
        keep = self.n_jobs
        try:
            self.n_jobs = 1
            return super().predict_proba(X)
        finally:
            self.n_jobs = keep


def grid_models():
    """모델×표현 격자에서 쓰는 모델 4종. 이름 -> 무인자 생성 함수."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = {
        "XGBoost": lambda: make_model(),
        "RandomForest": lambda: DeterministicRF(
            n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "LogisticReg": lambda: make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
        ),
    }
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return m
    m["LightGBM"] = lambda: LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    return m


def grid_reps():
    """모델×표현 격자에서 쓰는 분자 표현 4종. 이름 -> mols 를 받는 함수.

    2026-10-01: robustness.py 와 grid_calibration.py 가 같은 이름의 표현을
    각자 정의하면서 **dtype 이 달랐다**. robustness 는 지문·기술자 모두
    float32 였고, grid_calibration 은 지문 int8 + 기술자 float64 (hstack 뒤
    float64) 였다. 값은 같고 정밀도만 달랐으므로 트리 모델(XGBoost,
    RandomForest)은 같은 결과를 냈지만, 스케일링과 최적화를 거치는
    LogisticRegression 과 연속 변수를 구간화하는 LightGBM 은 달라졌다.
    실제로 32칸 중 12칸(LogisticReg 8, LightGBM 4)이 어긋났고, LightGBM 은
    연속 기술자가 들어가는 표현(ECFP4+desc, RDKit desc)에서만 어긋났다.
    이것이 §9.1 과 §9.5 불일치의 두 번째 원인이다 (첫 번째는 모델
    하이퍼파라미터, 2026-09-30 수정).

    정의를 이 한 곳으로 합치고 dtype 은 주 분석 featurize() 와 같은 쪽
    — 지문 int8, 기술자 float64 — 으로 맞춘다. 그래서 격자의
    XGBoost × ECFP4+desc 칸은 주 분석 값과 같아야 한다.
    """
    from rdkit.Chem import MACCSkeys

    def _fp(m):
        return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), dtype=np.int8)

    def _desc(m):
        return [
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            Descriptors.TPSA(m),
            Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m),
        ]

    def ecfp_desc(mols):
        return np.hstack([np.array([_fp(m) for m in mols]), np.array([_desc(m) for m in mols])])

    def ecfp(mols):
        return np.array([_fp(m) for m in mols])

    def maccs(mols):
        return np.array([np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.int8) for m in mols])

    def desc(mols):
        return np.array([_desc(m) for m in mols])

    return {"ECFP4+desc": ecfp_desc, "ECFP4": ecfp, "MACCS": maccs, "RDKit desc": desc}


# ------------------------------------------------------------ 보정 지표
def ece(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        out += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(out)


def calib_slope(y, p):
    # 2026-10-20: XGBoost 의 predict_proba 는 float32 를 돌려준다.
    # 캐스팅하지 않으면 로짓과 그 뒤의 로지스틱 적합이 **단정밀도**로
    # 일어나고, 같은 지표가 어디서 왔는지에 따라 정밀도가 달라진다
    # (§10 Platt 계수가 §10.5 와 2.1e-04 어긋난 원인이다,
    # src/diff_ladder.py 단계 10). float64 로 올려 통일한다.
    p = np.asarray(p, dtype=np.float64)
    lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def bin_table(y, p, n_bins=N_BINS):
    """(평균예측, 실제양성률, n) 목록. murphy_decomposition 의 입력."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def murphy_decomposition(bins):
    N = sum(n for _, _, n in bins)
    obar = sum(o * n for _, o, n in bins) / N
    uncertainty = obar * (1 - obar)
    resolution = sum(n * (o - obar) ** 2 for _, o, n in bins) / N
    reliability = sum(n * (f - o) ** 2 for f, o, n in bins) / N
    return dict(
        N=N,
        obar=obar,
        uncertainty=uncertainty,
        resolution=resolution,
        reliability=reliability,
        brier_approx=uncertainty - resolution + reliability,
    )


def decompose(y, p):
    return murphy_decomposition(bin_table(y, p))


# ------------------------------------------------------------ 유병률 맞춤
def subsample_to_rate(y, idx, target_rate, r):
    """prevalence_control.py 와 동일한 구현 (값 일치 유지)."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    n_neg_a = int(round(len(pos) * (1 - target_rate) / max(target_rate, EPS)))
    n_pos_b = int(round(len(neg) * target_rate / max(1 - target_rate, EPS)))
    if n_neg_a <= len(neg):
        return np.concatenate([pos, r.choice(neg, n_neg_a, replace=False)])
    if n_pos_b <= len(pos):
        return np.concatenate([r.choice(pos, n_pos_b, replace=False), neg])
    return idx


def temporal_masks(df):
    tr = df["approval_year"].values < CUTOFF
    return tr, ~tr


DATASETS = [("DILIrank", "dilirank2_with_year.csv"), ("DICTrank", "dictrank_processed.csv")]


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, m=""):
        print(m, flush=True)
        self.lines.append(str(m))

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
