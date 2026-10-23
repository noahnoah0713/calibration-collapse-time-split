"""입력이 무엇이든 성립해야 하는 성질.

정답 대조(test_metrics.py)는 손으로 계산할 수 있는 작은 예제만 덮는다.
여기서는 임의의 입력에 대해 깨지면 안 되는 성질을 본다. 부호 하나,
클리핑 한 번, 구간 경계 하나가 틀어지면 작은 예제는 넘어가도 이쪽이
걸린다. 난수는 전부 고정 seed 라 실행할 때마다 같은 입력을 쓴다.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import _shared as S


def sample(n=500, seed=0):
    r = np.random.default_rng(seed)
    p = r.uniform(0.0, 1.0, n)
    y = (r.uniform(size=n) < p).astype(int)
    return y, p


class EceProperties(unittest.TestCase):
    def test_row_order_does_not_matter(self):
        y, p = sample()
        r = np.random.default_rng(1)
        o = r.permutation(len(y))
        self.assertAlmostEqual(S.ece(y, p), S.ece(y[o], p[o]), places=12)

    def test_bounded_between_zero_and_one(self):
        for seed in range(10):
            y, p = sample(seed=seed)
            e = S.ece(y, p)
            self.assertGreaterEqual(e, 0.0)
            self.assertLessEqual(e, 1.0)

    def test_constant_forecast_equals_the_gap_to_the_base_rate(self):
        # 모두 같은 값을 예측하면 ECE 는 |실제 양성률 - 그 값| 이다.
        y = np.array([1] * 30 + [0] * 70)
        p = np.full(100, 0.45)
        self.assertAlmostEqual(S.ece(y, p), abs(0.30 - 0.45), places=12)

    def test_more_bins_never_lowers_it_below_the_coarse_value_by_much(self):
        # 구간을 잘게 쪼개면 ECE 는 보통 커진다. 작아지더라도 미세해야 한다.
        y, p = sample(n=2000, seed=5)
        self.assertGreaterEqual(S.ece(y, p, n_bins=20) + 1e-9, S.ece(y, p, n_bins=5))


class DecompositionProperties(unittest.TestCase):
    def test_terms_stay_in_their_ranges(self):
        for seed in range(10):
            y, p = sample(seed=seed)
            d = S.decompose(y, p)
            self.assertGreaterEqual(d["reliability"], 0.0)
            self.assertGreaterEqual(d["resolution"], 0.0)
            self.assertGreaterEqual(d["uncertainty"], 0.0)
            self.assertLessEqual(d["uncertainty"], 0.25)

    def test_tracks_the_brier_score_within_the_binning_error(self):
        # 10구간이면 구간 안의 예측값 분산만큼 어긋난다. 200회 중
        # 최대 0.0043 이었으므로 0.01 이면 넉넉하면서도 느슨하지 않다.
        for seed in range(20):
            y, p = sample(seed=seed)
            d = S.decompose(y, p)
            brier = float(np.mean((p - y) ** 2))
            self.assertLess(abs(d["brier_approx"] - brier), 0.01)

    def test_row_order_does_not_matter(self):
        # 부동소수점 덧셈 순서 때문에 비트까지 같지는 않다. 1e-12 이면
        # 순서 효과는 통과하고 실제 계산 오류는 걸린다.
        y, p = sample()
        r = np.random.default_rng(2)
        o = r.permutation(len(y))
        a, b = S.decompose(y, p), S.decompose(y[o], p[o])
        self.assertEqual(a["N"], b["N"])
        for key in ("obar", "uncertainty", "resolution", "reliability", "brier_approx"):
            with self.subTest(term=key):
                self.assertAlmostEqual(a[key], b[key], places=12)


class BinTableProperties(unittest.TestCase):
    def test_counts_add_up_to_the_sample_size(self):
        y, p = sample()
        self.assertEqual(sum(n for _, _, n in S.bin_table(y, p)), len(y))

    def test_bins_come_out_in_increasing_order(self):
        y, p = sample()
        means = [f for f, _, _ in S.bin_table(y, p)]
        self.assertEqual(means, sorted(means))

    def test_every_point_lands_in_exactly_one_bin(self):
        # 0.0 과 1.0 을 포함해도 총합이 유지되어야 한다.
        y = np.array([0, 1, 0, 1])
        p = np.array([0.0, 1.0, 0.5, 0.1])
        self.assertEqual(sum(n for _, _, n in S.bin_table(y, p)), 4)


class SlopeProperties(unittest.TestCase):
    def test_row_order_does_not_matter(self):
        y, p = sample()
        r = np.random.default_rng(4)
        o = r.permutation(len(y))
        self.assertAlmostEqual(S.calib_slope(y, p), S.calib_slope(y[o], p[o]), places=8)

    def test_scaling_the_logit_divides_the_slope(self):
        # 로짓을 c 배 하면 기울기는 정확히 1/c 배가 된다. 로짓 변환이나
        # 클리핑이 틀어지면 이 항등식이 먼저 깨진다.
        r = np.random.default_rng(3)
        lp = r.normal(0, 2, 4000)
        p = 1 / (1 + np.exp(-lp))
        y = (r.uniform(size=4000) < p).astype(int)
        base = S.calib_slope(y, p)
        for c in (2.0, 0.5):
            with self.subTest(scale=c):
                scaled = 1 / (1 + np.exp(-lp * c))
                self.assertAlmostEqual(S.calib_slope(y, scaled), base / c, places=3)


class SubsampleProperties(unittest.TestCase):
    def test_result_is_a_subset_without_repeats(self):
        y = np.array([1] * 40 + [0] * 60)
        idx = np.arange(100)
        sel = S.subsample_to_rate(y, idx, 0.3, np.random.default_rng(S.SEED))
        self.assertTrue(set(sel.tolist()) <= set(idx.tolist()))
        self.assertEqual(len(set(sel.tolist())), len(sel))

    def test_same_seed_picks_the_same_rows(self):
        y = np.array([1] * 40 + [0] * 60)
        idx = np.arange(100)
        a = S.subsample_to_rate(y, idx, 0.3, np.random.default_rng(S.SEED))
        b = S.subsample_to_rate(y, idx, 0.3, np.random.default_rng(S.SEED))
        self.assertEqual(a.tolist(), b.tolist())


class TemporalMaskProperties(unittest.TestCase):
    def test_the_two_masks_partition_the_rows(self):
        import pandas as pd

        df = pd.DataFrame({"approval_year": [1998, 2005, 2009, 2010, 2015, 2021]})
        tr, te = S.temporal_masks(df)
        self.assertEqual(list(tr & te), [False] * 6)
        self.assertEqual(list(tr | te), [True] * 6)
        self.assertEqual(int(tr.sum()), 3)


if __name__ == "__main__":
    unittest.main()
