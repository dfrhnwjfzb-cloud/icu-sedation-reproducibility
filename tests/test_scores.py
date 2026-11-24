import sys
import unittest
from pathlib import Path

import src.round1.scores as scores_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
score = scores_module.score
apache2_gate = scores_module.apache2_gate


class Rules(unittest.TestCase):
    def test_thresholds(self):
        for var, x, expected, vent in [
            ("pf", 99, 4, True),
            ("pf", 100, 3, True),
            ("pf", 200, 2, True),
            ("pf", 300, 1, True),
            ("pf", 400, 0, True),
            ("pf", 80, 2, False),
            ("platelet", 20, 3, False),
            ("platelet", 50, 2, False),
            ("platelet", 100, 1, False),
            ("platelet", 150, 0, False),
            ("bilirubin", 1.2, 1, False),
            ("bilirubin", 2, 2, False),
            ("bilirubin", 6, 3, False),
            ("bilirubin", 12, 4, False),
            ("gcs", 15, 0, False),
            ("gcs", 14, 1, False),
            ("gcs", 12, 2, False),
            ("gcs", 9, 3, False),
            ("gcs", 5, 4, False),
            ("creatinine", 5, 4, False),
            ("creatinine", 3.5, 3, False),
            ("urine24", 200, 3, False),
            ("urine24", 500, 0, False),
            ("dopamine", 5, 2, False),
            ("dopamine", 15, 3, False),
            ("dopamine", 16, 4, False),
            ("norepinephrine", 0.1, 3, False),
            ("norepinephrine", 0.11, 4, False),
            ("map", 70, 0, False),
        ]:
            with self.subTest(var=var, x=x):
                self.assertEqual(score(var, x, vent), expected)

    def test_missing(self):
        self.assertIsNone(score("gcs", None))

    def test_apache_unknown(self):
        with self.assertRaises(ValueError):
            apache2_gate({"pH": None}, 60, None, None)
