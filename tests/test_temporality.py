import unittest
from datetime import datetime, timedelta

import src.round1.temporality as temporality_module

in_window = temporality_module.in_window
clip_interval = temporality_module.clip_interval
usable_urine = temporality_module.usable_urine
total_score = temporality_module.total_score


class TimeRules(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2020, 1, 3)
        self.i = self.t - timedelta(hours=30)

    def test_strict_boundary(self):
        self.assertFalse(in_window(self.t, self.t, self.i))
        self.assertFalse(in_window(self.t + timedelta(seconds=1), self.t, self.i))
        self.assertTrue(in_window(self.t - timedelta(hours=24), self.t, self.i))

    def test_early_time_zero(self):
        i = self.t - timedelta(hours=2)
        self.assertFalse(in_window(i - timedelta(seconds=1), self.t, i))
        self.assertTrue(in_window(i, self.t, i))

    def test_infusion_clipping(self):
        self.assertEqual(
            clip_interval(
                self.t - timedelta(hours=1), self.t + timedelta(hours=8), self.t, self.i
            ),
            (self.t - timedelta(hours=1), self.t),
        )
        self.assertIsNone(
            clip_interval(self.t, self.t + timedelta(hours=1), self.t, self.i)
        )

    def test_urine(self):
        ws = self.t - timedelta(hours=24)
        self.assertFalse(usable_urine(6, ws, ws))
        self.assertFalse(usable_urine(24, ws - timedelta(minutes=1), ws))
        self.assertTrue(usable_urine(24, ws, ws))
        self.assertFalse(usable_urine(None, ws, ws))

    def test_missing_organ_not_silently_complete(self):
        self.assertIsNone(total_score([2, None, 1]))
        self.assertEqual(total_score([2, None, 1], True), 3)
