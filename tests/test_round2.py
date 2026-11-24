import unittest

import numpy as np
import pandas as pd

import src.round2.common as p


class Round2Tests(unittest.TestCase):
    def test_subject_cluster_multiplicities(self):
        subject = np.array([0, 0, 1])
        counts = np.array([2, 1])
        ix = np.repeat(np.arange(3), counts[subject])
        self.assertEqual(ix.tolist(), [0, 0, 1, 1, 2])
        fold = np.array([0, 1, 2])
        self.assertEqual(fold[ix].tolist(), [0, 0, 1, 1, 2])

    def test_missing_cns_retained(self):
        x = pd.read_csv(p.R / "data/round2_scoring_audit.csv")
        self.assertEqual(int(x.cns_G1.notna().sum()), 759)
        self.assertTrue(x.loc[x.cns_G1.isna(), "SOFA_G1_CNS_observed"].isna().all())

    def test_no_raw_gcs_component_future(self):
        z = pd.read_csv(p.R / "data/gcs_auditable_triplets.csv")
        w = pd.read_csv(p.OLD / "data/windows.csv")
        z = z.merge(w[["stay_id", "t0", "ws"]], on="stay_id", validate="many_to_one")
        for part in ["eyes", "motor", "verbal"]:
            self.assertTrue(
                (pd.to_datetime(z[part + "_charttime"]) < pd.to_datetime(z.t0)).all()
            )
            self.assertTrue(
                (pd.to_datetime(z[part + "_storetime"]) < pd.to_datetime(z.t0)).all()
            )
            self.assertTrue(
                (pd.to_datetime(z[part + "_charttime"]) >= pd.to_datetime(z.ws)).all()
            )
            self.assertTrue(
                (
                    (
                        pd.to_datetime(z.evaluation_time)
                        - pd.to_datetime(z[part + "_charttime"])
                    ).dt.total_seconds()
                    <= 6 * 3600
                ).all()
            )

    def test_ps_derivative_math(self):
        z = pd.read_csv(p.R / "outputs/PS_derivative_numerical_audit.csv")
        self.assertLess(z.absolute_error.max(), 1e-06)

    def test_raw_urine_exact_netting(self):
        z = pd.read_csv(p.R / "data/raw_vs_derived_urine.csv")
        self.assertTrue((z._merge == "both").all())
        self.assertTrue(np.allclose(z.urineoutput_derived, z.urineoutput_raw))

    def test_original_point_reproduced(self):
        old = pd.read_csv(p.OLD / "outputs/all_models.csv")
        new = pd.read_csv(p.R / "outputs/bootstrap_point_estimates.csv")
        for q in new[new.model != "ModelW"].itertuples():
            r = old[
                (old.model == q.model)
                & (old.group == q.group)
                & (old.outcome == q.outcome)
            ].iloc[0]
            self.assertAlmostEqual(q.RD, r.risk_difference, places=9)
            self.assertAlmostEqual(q.RR, r.risk_ratio, places=9)

    def test_bootstrap_keeps_invalid_risk(self):
        z = pd.read_csv(p.R / "bootstrap/formal.csv")
        bad = z[z.risk_outside_unit]
        self.assertEqual(len(bad), 1)
        self.assertTrue(np.isfinite(bad.RD).all())
        self.assertEqual(len(z), 6000)

    def test_all_expected_bootstrap_sets(self):
        z = pd.read_csv(p.R / "bootstrap/formal_convergence.csv")
        self.assertEqual(len(z), 3000)
        self.assertTrue(z.success.all())
        self.assertTrue((z.groupby(["group", "model"]).size() == 500).all())
        self.assertEqual(len(np.load(p.R / "bootstrap/subject_draw_indices.npy")), 610)
