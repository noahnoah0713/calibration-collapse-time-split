# -*- coding: utf-8 -*-
"""지표 함수의 정답 대조.

여기 적힌 기대값은 손으로 계산한 것이다. verify9~verify12 는 '결과끼리
서로 맞는가' 를 보기 때문에, 지표 함수가 통째로 틀려 있으면 모든 결과가
사이좋게 틀린 채로 통과한다. 그 구멍을 메우려고 둔 파일이다.

    python -m unittest discover -s tests -v

rdkit·xgboost·scikit-learn 이 있는 환경에서 돌려야 한다 (conda env pharmml).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import _shared as S


class Constants(unittest.TestCase):
    """논문 전체가 이 네 값에 묶여 있다. 조용히 바뀌면 모든 수치가 흔들린다."""

    def test_values(self):
        self.assertEqual(S.SEED, 42)
        self.assertEqual(S.CUTOFF, 2010)
        self.assertEqual(S.N_BINS, 10)
        self.assertEqual(S.EPS, 1e-6)


class Ece(unittest.TestCase):
    def test_hand_computed(self):
        # 10구간. 0.05 열 개(실제 0) 와 0.95 열 개(실제 1).
        # 첫 구간 기여 0.5 * |0 - 0.05| = 0.025
        # 끝 구간 기여 0.5 * |1 - 0.95| = 0.025
        y = np.array([0] * 10 + [1] * 10)
        p = np.array([0.05] * 10 + [0.95] * 10)
        self.assertAlmostEqual(S.ece(y, p), 0.05, places=12)

    def test_perfect_calibration_is_zero(self):
        y = np.array([1] * 5 + [0] * 5)
        p = np.full(10, 0.5)
        self.assertAlmostEqual(S.ece(y, p), 0.0, places=12)

    def test_upper_edge_belongs_to_last_bin(self):
        # 마지막 구간만 닫힌 구간이다. p=1.0 이 버려지면 ECE 가 과소평가된다.
        y = np.array([1, 1, 0, 0])
        p = np.full(4, 1.0)
        self.assertAlmostEqual(S.ece(y, p), 0.5, places=12)


class BinTable(unittest.TestCase):
    def test_empty_bins_are_skipped(self):
        y = np.array([0, 1])
        p = np.array([0.05, 0.95])
        self.assertEqual(len(S.bin_table(y, p)), 2)

    def test_row_contents(self):
        y = np.array([1, 1, 0, 0])
        p = np.full(4, 1.0)
        self.assertEqual(S.bin_table(y, p), [(1.0, 0.5, 4)])


class Murphy(unittest.TestCase):
    """구간 안에서 예측값이 하나면 분해식이 Brier 와 정확히 같아야 한다."""

    def setUp(self):
        self.y = np.array([0, 0, 1, 1, 1, 1, 1, 0])
        self.p = np.array([0.05] * 4 + [0.95] * 4)

    def test_identity_matches_brier(self):
        d = S.decompose(self.y, self.p)
        brier = float(np.mean((self.p - self.y) ** 2))
        self.assertAlmostEqual(d["brier_approx"], brier, places=12)
        self.assertAlmostEqual(brier, 0.34, places=12)

    def test_hand_computed_terms(self):
        d = S.decompose(self.y, self.p)
        self.assertAlmostEqual(d["obar"], 0.625, places=12)
        self.assertAlmostEqual(d["uncertainty"], 0.234375, places=12)
        self.assertAlmostEqual(d["resolution"], 0.015625, places=12)
        self.assertAlmostEqual(d["reliability"], 0.12125, places=12)
        self.assertEqual(d["N"], 8)

    def test_decompose_is_bin_table_plus_murphy(self):
        self.assertEqual(
            S.decompose(self.y, self.p),
            S.murphy_decomposition(S.bin_table(self.y, self.p)),
        )


class CalibSlope(unittest.TestCase):
    def test_perfect_calibration_gives_one(self):
        lp = np.linspace(-4, 4, 33)
        pr = 1 / (1 + np.exp(-lp))
        ys, ps = [], []
        for q in pr:
            n1 = int(round(q * 200))
            ys += [1] * n1 + [0] * (200 - n1)
            ps += [q] * 200
        self.assertAlmostEqual(S.calib_slope(np.array(ys), np.array(ps)), 1.0, places=2)

    def test_float32_input_gives_the_identical_answer(self):
        # 2026-10-20 과 10-21 에 잡힌 결함. XGBoost 의 predict_proba 는
        # float32 를 돌려준다. 로짓 변환을 단정밀도로 하면 같은 지표가
        # 출처에 따라 갈렸다. 수리는 함수 첫 줄의 float64 캐스팅이다.
        #
        # 같은 값을 두 dtype 으로 넣는다. 값이 같으므로 답도 완전히
        # 같아야 한다. 캐스팅을 빼면 1e-9 수준으로 갈라진다 — 그래서
        # 근사가 아니라 일치를 요구한다.
        r = np.random.default_rng(0)
        p32 = r.uniform(0.01, 0.99, 500).astype(np.float32)
        y = (r.uniform(size=500) < p32).astype(int)
        self.assertEqual(S.calib_slope(y, p32), S.calib_slope(y, p32.astype(np.float64)))


class SubsampleToRate(unittest.TestCase):
    def test_hits_the_target_rate(self):
        y = np.array([1] * 10 + [0] * 90)
        idx = np.arange(100)
        sel = S.subsample_to_rate(y, idx, 0.5, np.random.default_rng(S.SEED))
        self.assertAlmostEqual(float(y[sel].mean()), 0.5, places=12)
        self.assertEqual(len(sel), 20)

    def test_returns_input_when_a_class_is_missing(self):
        y = np.zeros(10, dtype=int)
        idx = np.arange(10)
        sel = S.subsample_to_rate(y, idx, 0.5, np.random.default_rng(S.SEED))
        self.assertEqual(list(sel), list(idx))


class TemporalMasks(unittest.TestCase):
    def test_cutoff_year_is_test_side(self):
        import pandas as pd

        df = pd.DataFrame({"approval_year": [2009, 2010, 2011]})
        tr, te = S.temporal_masks(df)
        self.assertEqual(list(tr), [True, False, False])
        self.assertEqual(list(te), [False, True, True])


if __name__ == "__main__":
    unittest.main()
