"""같은 지표의 여러 사본을 각각 정답과 대조한다.

이 저장소는 지표 함수를 한 군데 두지 않았다. ece 는 일곱 번, calib_slope
와 bin_table 과 murphy_decomposition 은 세 번씩 정의되어 있다. 분석
스크립트를 하나씩 따로 돌릴 수 있게 하려고 그렇게 두었고, 사본끼리
어긋나지 않는지는 check_determinism.py 가 본다.

그런데 사본끼리 일치한다는 것은 옳다는 뜻이 아니다. 일곱 개가 똑같이
틀려 있어도 그 점검은 통과한다. 그래서 여기서는 사본 하나하나를 손으로
계산한 답과 직접 맞춰본다.

정의 개수도 함께 고정한다. 사본을 하나 더 만들고 등록하지 않으면
여기서 걸린다 (기대값의 출처는 results/metric_duplicates.csv).
"""

import contextlib
import glob
import io
import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _quiet_import(name):
    """분석 스크립트는 들여올 때 머리글을 찍는다. 조용히 들여온다."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return __import__(name)


# 정답은 tests/test_metrics.py 와 같다. 손으로 계산한 값이다.
Y2 = np.array([0] * 10 + [1] * 10)
P2 = np.array([0.05] * 10 + [0.95] * 10)
ECE2 = 0.05

Y8 = np.array([0, 0, 1, 1, 1, 1, 1, 0])
P8 = np.array([0.05] * 4 + [0.95] * 4)
MURPHY8 = {
    "N": 8,
    "obar": 0.625,
    "uncertainty": 0.234375,
    "resolution": 0.015625,
    "reliability": 0.12125,
    "brier_approx": 0.34,
}

ECE_MODULES = [
    "_shared",
    "baseline_cv",
    "prevalence_control",
    "recalibration",
    "robustness",
    "seed_sensitivity",
    "year_sensitivity",
]
SLOPE_MODULES = ["_shared", "prevalence_control", "robustness"]
BIN_MODULES = ["_shared", "brier_decomposition", "brier_decomposition_cutoff"]
SUBSAMPLE_MODULES = ["_shared", "prevalence_control"]

EXPECTED_COUNTS = {
    "ece": 7,
    "calib_slope": 3,
    "bin_table": 3,
    "murphy_decomposition": 3,
    "subsample_to_rate": 2,
    "ece_mce": 1,
    "calib_slope_intercept": 1,
}


class DefinitionCount(unittest.TestCase):
    def test_counts_match_the_recorded_inventory(self):
        src = sorted(glob.glob(os.path.join(ROOT, "src", "*.py")))
        found = dict.fromkeys(EXPECTED_COUNTS, 0)
        for path in src:
            with io.open(path, encoding="utf-8") as fh:
                text = fh.read()
            for name in found:
                found[name] += len(re.findall(r"^def %s\(" % name, text, re.M))
        self.assertEqual(found, EXPECTED_COUNTS)


class EveryEce(unittest.TestCase):
    def test_each_copy_gives_the_hand_computed_value(self):
        for name in ECE_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                self.assertAlmostEqual(mod.ece(Y2, P2), ECE2, places=12)

    def test_each_copy_is_zero_on_perfect_calibration(self):
        y = np.array([1] * 5 + [0] * 5)
        p = np.full(10, 0.5)
        for name in ECE_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                self.assertAlmostEqual(mod.ece(y, p), 0.0, places=12)

    def test_the_alias_in_calibration_agrees(self):
        cal = _quiet_import("calibration")
        ece, mce, table = cal.ece_mce(Y2, P2)
        self.assertAlmostEqual(ece, ECE2, places=12)
        self.assertAlmostEqual(mce, 0.05, places=12)
        self.assertEqual(len(table), 2)


class EverySlope(unittest.TestCase):
    def test_each_copy_agrees_on_perfect_calibration(self):
        lp = np.linspace(-4, 4, 33)
        pr = 1 / (1 + np.exp(-lp))
        ys, ps = [], []
        for q in pr:
            n1 = int(round(q * 200))
            ys += [1] * n1 + [0] * (200 - n1)
            ps += [q] * 200
        y, p = np.array(ys), np.array(ps)
        for name in SLOPE_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                self.assertAlmostEqual(mod.calib_slope(y, p), 1.0, places=2)

    def test_the_alias_in_calibration_returns_the_same_slope(self):
        r = np.random.default_rng(1)
        p = r.uniform(0.02, 0.98, 400)
        y = (r.uniform(size=400) < p).astype(int)
        shared = _quiet_import("_shared")
        cal = _quiet_import("calibration")
        slope, _intercept = cal.calib_slope_intercept(y, p)
        self.assertAlmostEqual(slope, shared.calib_slope(y, p), places=10)


class Float32Regression(unittest.TestCase):
    """10-20 과 10-21 에 잡힌 정밀도 결함을 네 군데 모두에서 고정한다.

    같은 값을 float32 와 float64 로 넣으면 답이 완전히 같아야 한다.
    로짓 변환 전에 float64 로 올리지 않으면 1e-9 수준으로 갈라진다.
    열 번째 결함은 이름이 calib_slope 가 아니라는 이유로 사본 목록에서
    빠져 있었다. 이름이 아니라 동작으로 걸러내려고 둔 검사다.
    """

    def setUp(self):
        r = np.random.default_rng(0)
        self.p32 = r.uniform(0.01, 0.99, 500).astype(np.float32)
        self.y = (r.uniform(size=500) < self.p32).astype(int)
        self.p64 = self.p32.astype(np.float64)

    def test_every_calib_slope_copy_is_dtype_insensitive(self):
        for name in SLOPE_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                self.assertEqual(
                    mod.calib_slope(self.y, self.p32),
                    mod.calib_slope(self.y, self.p64),
                )

    def test_the_alias_is_dtype_insensitive(self):
        cal = _quiet_import("calibration")
        self.assertEqual(
            cal.calib_slope_intercept(self.y, self.p32),
            cal.calib_slope_intercept(self.y, self.p64),
        )


class EveryDecomposition(unittest.TestCase):
    def test_each_copy_gives_the_hand_computed_terms(self):
        for name in BIN_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                d = mod.murphy_decomposition(mod.bin_table(Y8, P8))
                for key, want in MURPHY8.items():
                    self.assertAlmostEqual(d[key], want, places=12, msg=key)

    def test_each_copy_matches_the_brier_score(self):
        brier = float(np.mean((P8 - Y8) ** 2))
        for name in BIN_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                d = mod.murphy_decomposition(mod.bin_table(Y8, P8))
                self.assertAlmostEqual(d["brier_approx"], brier, places=12)


class EverySubsample(unittest.TestCase):
    def test_each_copy_hits_the_target_rate(self):
        y = np.array([1] * 10 + [0] * 90)
        idx = np.arange(100)
        for name in SUBSAMPLE_MODULES:
            with self.subTest(module=name):
                mod = _quiet_import(name)
                sel = mod.subsample_to_rate(y, idx, 0.5, np.random.default_rng(42))
                self.assertAlmostEqual(float(y[sel].mean()), 0.5, places=12)
                self.assertEqual(len(sel), 20)

    def test_the_copies_pick_the_same_rows_from_the_same_seed(self):
        y = np.array([1] * 25 + [0] * 75)
        idx = np.arange(100)
        picks = [
            sorted(
                _quiet_import(name)
                .subsample_to_rate(y, idx, 0.4, np.random.default_rng(42))
                .tolist()
            )
            for name in SUBSAMPLE_MODULES
        ]
        self.assertEqual(picks[0], picks[1])


if __name__ == "__main__":
    unittest.main()
