"""모델링 쪽의 성질 검사 — 수치가 아니라 규약을 고정한다.

지표 함수와 달리 featurize·make_model 의 출력은 라이브러리 버전에 따라
값이 달라진다(rdkit 의 기술자 구현, 부스팅 라이브러리의 수치 경로).
그래서 여기서는 특정 값을 고정하지 않고, 버전이 바뀌어도 반드시
성립해야 하는 것만 본다 — 모양, 자료형, 같은 입력에 같은 출력,
같은 seed 에 같은 예측.

rdkit·xgboost·scikit-learn 이 필요하다. CI 의 tests 작업은
requirements.txt 로 설치한 뒤 이 파일을 돌린다.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import _shared as S

# 작고 서로 다른 분자들. 값이 아니라 규약을 보는 것이므로 10개면 충분하다.
SMILES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "c1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "ClCCl",
    "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1O",
    "CCN(CC)CC",
    "C1CCNCC1",
]
Y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])


class Featurize(unittest.TestCase):
    def setUp(self):
        self.X = S.featurize(SMILES)

    def test_shape_is_fingerprint_plus_six_descriptors(self):
        self.assertEqual(self.X.shape, (len(SMILES), 2048 + 6))

    def test_fingerprint_block_is_binary(self):
        fp = self.X[:, :2048]
        self.assertTrue(np.isin(fp, (0.0, 1.0)).all())
        self.assertGreater(fp.sum(), 0)

    def test_descriptor_block_is_finite(self):
        self.assertTrue(np.isfinite(self.X[:, -6:]).all())

    def test_molecular_weight_column_is_ordered_as_documented(self):
        # 마지막 여섯 열은 MolWt, MolLogP, TPSA, HDonors, HAcceptors, RotB 순이다.
        # 에탄올(46) 이 이부프로펜(206) 보다 가볍다는 것만 본다 — 구현에
        # 무관하게 성립하고, 열 순서가 뒤바뀌면 깨진다.
        mw = self.X[:, -6]
        self.assertLess(mw[0], mw[4])
        self.assertLess(mw[0], 60)

    def test_same_input_gives_the_same_matrix(self):
        self.assertTrue(np.array_equal(self.X, S.featurize(SMILES)))

    def test_row_order_follows_input_order(self):
        rev = S.featurize(list(reversed(SMILES)))
        self.assertTrue(np.array_equal(self.X, rev[::-1]))


class Models(unittest.TestCase):
    def test_same_seed_gives_identical_predictions(self):
        X = S.featurize(SMILES)
        a = S.make_model().fit(X, Y).predict_proba(X)[:, 1]
        b = S.make_model().fit(X, Y).predict_proba(X)[:, 1]
        self.assertTrue(np.array_equal(a, b))

    def test_the_grid_always_holds_the_three_core_models(self):
        g = S.grid_models()
        for name in ("XGBoost", "RandomForest", "LogisticReg"):
            self.assertIn(name, g)

    def test_lightgbm_is_in_the_grid_exactly_when_it_is_installed(self):
        # grid_models() 는 lightgbm 이 없으면 조용히 3종만 돌려준다. 그러면
        # 격자가 32칸이 아니라 24칸이 되고, verify11 은 두 파일을 서로
        # 비교하므로 양쪽이 같이 줄면 알아차리지 못한다. 논문 수치는 4종
        # 격자에서 나온 것이므로 이 조건을 눈에 보이게 고정해 둔다.
        import importlib.util

        has = importlib.util.find_spec("lightgbm") is not None
        self.assertEqual("LightGBM" in S.grid_models(), has)
        if not has:
            self.skipTest("lightgbm 이 없다 — 격자는 24칸으로 줄어든다")

    def test_deterministic_forest_repeats_itself(self):
        X = S.featurize(SMILES)
        rf = S.grid_models()["RandomForest"]
        a = rf().fit(X, Y).predict_proba(X)[:, 1]
        b = rf().fit(X, Y).predict_proba(X)[:, 1]
        self.assertTrue(np.array_equal(a, b))

    def test_probabilities_stay_in_range(self):
        X = S.featurize(SMILES)
        p = S.make_model().fit(X, Y).predict_proba(X)[:, 1]
        self.assertTrue(((p >= 0) & (p <= 1)).all())


class SplitLogic(unittest.TestCase):
    def test_the_cutoff_year_goes_to_the_test_side(self):
        import pandas as pd

        df = pd.DataFrame({"approval_year": [S.CUTOFF - 1, S.CUTOFF, S.CUTOFF + 1]})
        tr, te = S.temporal_masks(df)
        self.assertEqual(list(tr), [True, False, False])

    def test_the_two_sides_never_share_a_row(self):
        import pandas as pd

        df = pd.DataFrame({"approval_year": [1990, 2005, 2010, 2020]})
        tr, te = S.temporal_masks(df)
        self.assertEqual(int((tr & te).sum()), 0)
        self.assertEqual(int((tr | te).sum()), len(df))


if __name__ == "__main__":
    unittest.main()
