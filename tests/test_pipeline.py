"""End-to-end protocol: calibration strictly before holdout, no leakage,
exact tables, full N x N swap matrix, determinism."""
import json
import math
import unittest
from unittest import mock

import kinematics_auth as ka
import synth_traces as synth

SEED = 4141
QUIET = dict(log=lambda *a: None)


def strip(rep):
    r = dict(rep)
    r.pop("elapsed_s")
    return r


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []
        real_cal, real_hold = ka.calibrate, synth.holdout_split

        def cal(*a, **k):
            cls.calls.append("calibrate")
            return real_cal(*a, **k)

        def hold(*a, **k):
            cls.calls.append("holdout_split")
            return real_hold(*a, **k)
        with mock.patch.object(ka, "calibrate", cal), mock.patch.object(synth, "holdout_split", hold):
            cls.rep = ka.run_benchmark(SEED, alphas=(0.0, 1.0), **QUIET)
        cls.rep2 = ka.run_benchmark(SEED, alphas=(0.0, 1.0), **QUIET)
        cls.rep_other_holdout = ka.run_benchmark(SEED, alphas=(0.0,), holdout_seed=SEED + 99, **QUIET)

    def test_threshold_fixed_before_holdout_exists(self):
        # first calibrate (threshold), then the holdout generator; the 2nd calibrate
        # call is the post-hoc diagnostic inside evaluate()
        self.assertEqual(self.calls[:2], ["calibrate", "holdout_split"])

    def test_no_leakage_threshold_independent_of_holdout(self):
        a, b = self.rep["frozen"], self.rep_other_holdout["frozen"]
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertEqual(a["freeze_digest"], b["freeze_digest"])
        self.assertNotEqual(self.rep["table"], self.rep_other_holdout["table"])  # holdout really changed

    def test_evaluate_refuses_unfrozen_or_foreign_threshold(self):
        model = {"profiles": {}, "background": {}}
        with self.assertRaises(RuntimeError):
            ka.evaluate(model, None, {}, [], [], (0.0,), 0, 0)
        fz = ka.Frozen(0.0, "0" * 64, "x")
        with self.assertRaises(RuntimeError):
            ka.evaluate(model, fz, {}, [], [], (0.0,), 0, 0)

    def test_deterministic(self):
        self.assertEqual(json.dumps(strip(self.rep), sort_keys=True), json.dumps(strip(self.rep2), sort_keys=True))

    def test_table_covers_spec(self):
        classes = {(r["regime"], r["class"]) for r in self.rep["table"]}
        for reg in ("fast", "slow"):
            for c in ("genuine_normal", "fatigued_self", "random_stranger", "param_mimic", "swap_offdiag"):
                self.assertIn((reg, c), classes)

    def test_table_arithmetic_exact(self):
        for r in self.rep["table"]:
            self.assertEqual(r["rate"], r["errors"] / r["n"])
            self.assertGreaterEqual(r["wilson95_upper"], r["rate"])
        for m, cls in (("FRR", ("genuine_normal", "fatigued_self")),
                       ("FAR", ("random_stranger", "param_mimic", "swap_offdiag"))):
            k = sum(r["errors"] for r in self.rep["table"] if r["class"] in cls)
            n = sum(r["n"] for r in self.rep["table"] if r["class"] in cls)
            self.assertEqual((self.rep["pooled"][m]["errors"], self.rep["pooled"][m]["n"]), (k, n))

    def test_counts(self):
        t = {(r["regime"], r["class"]): r["n"] for r in self.rep["table"]}
        self.assertEqual(t[("fast", "genuine_normal")], 8 * 8)
        self.assertEqual(t[("slow", "fatigued_self")], 8 * 8)
        self.assertEqual(t[("fast", "random_stranger")], 20 * 8)
        self.assertEqual(t[("slow", "param_mimic")], 8 * 8)
        self.assertEqual(t[("fast", "swap_offdiag")], 8 * 7 * 8)

    def test_swap_matrix_full_and_consistent(self):
        for reg, m in self.rep["swap_matrix"].items():
            self.assertEqual(len(m), 8)
            for a, row in m.items():
                self.assertEqual(len(row), 8)
                for b, cell in row.items():
                    k, n = map(int, cell.split("/"))
                    self.assertEqual(n, 8)
            diag_rejects = sum(8 - int(m[p][p].split("/")[0]) for p in m)
            frr = [r for r in self.rep["table"] if r["regime"] == reg and r["class"] == "genuine_normal"][0]
            self.assertEqual(diag_rejects, frr["errors"])
            off = sum(int(m[a][b].split("/")[0]) for a in m for b in m if a != b)
            far = [r for r in self.rep["table"] if r["regime"] == reg and r["class"] == "swap_offdiag"][0]
            self.assertEqual(off, far["errors"])

    def test_clone_is_accepted_more_than_mimic(self):
        sw = {s["alpha"]: s["accepted"] / s["n"] for s in self.rep["mimic_strength_sweep"]}
        self.assertGreater(sw[1.0], 0.8)          # a perfect clone is the same operator
        self.assertLess(sw[0.0], 0.25)            # copying the visible stats is not enough

    def test_quality_floor(self):
        """Loose sanity bounds (actual numbers are in the README/evidence)."""
        self.assertLess(self.rep["pooled"]["FAR"]["rate"], 0.10)
        self.assertLess(self.rep["pooled"]["FRR"]["rate"], 0.10)

    def test_sequential_scores_shapes(self):
        self.assertEqual([s["segments_seen"] for s in self.rep["sequential"]], [1, 2, 3, 4])


class TestCalibrate(unittest.TestCase):
    def test_separable(self):
        c = ka.calibrate([2.0, 3.0, 4.0], [-1.0, 0.0, 1.0])
        self.assertEqual((c["cal_far"], c["cal_frr"]), (0.0, 0.0))
        self.assertEqual(c["threshold"], 1.5)

    def test_overlap_minimises_sum(self):
        c = ka.calibrate([0.0, 1.0, 2.0, 3.0], [-2.0, -1.0, 0.5, 1.5])
        self.assertAlmostEqual(c["cal_far"] + c["cal_frr"], 0.5)

    def test_wilson(self):
        self.assertAlmostEqual(ka.wilson_upper(0, 64), 0.0566, places=4)
        self.assertAlmostEqual(ka.wilson_upper(1, 64), 0.0833, places=4)
        self.assertIsNone(ka.wilson_upper(0, 0))


class TestModel(unittest.TestCase):
    def test_llr_clipped_and_missing_skipped(self):
        bg = {f: (0.0, 1.0) for f in ka.FEATURES}
        prof = {f: (0.0, 0.1) for f in ka.FEATURES}
        v = {f: None for f in ka.FEATURES}
        self.assertIsNone(ka.segment_llr(v, prof, bg))
        v["curv"] = 100.0
        self.assertEqual(ka.segment_llr(v, prof, bg), ka.LLR_CLIP[0])
        v["curv"] = 0.0
        self.assertAlmostEqual(ka.segment_llr(v, prof, bg), math.log(10.0))

    def test_drift_floor(self):
        bg = {f: (0.0, 1.0) for f in ka.FEATURES}
        vecs = [{f: 0.5 for f in ka.FEATURES} for _ in range(6)]
        prof = ka.fit_profile(vecs, bg)
        self.assertAlmostEqual(prof["curv"][1], ka.DRIFT * 1.0)

    def test_sequential_running_mean(self):
        bg = {f: (0.0, 1.0) for f in ka.FEATURES}
        prof = {f: (0.0, 1.0) for f in ka.FEATURES}
        vecs = [{f: 0.0 for f in ka.FEATURES}] * 4
        self.assertEqual(ka.sequential_scores(vecs, prof, bg), [0.0] * 4)


if __name__ == "__main__":
    unittest.main()
