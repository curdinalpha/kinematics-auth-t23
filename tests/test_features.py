"""Feature extraction: validation, exact edit accounting, and the invariances
the spec asks for (no assumption on sensor rate, absolute speed or channel)."""
import copy
import math
import statistics as st
import unittest

import kinematics_auth as ka
import synth_traces as synth

IDS = synth.build_population(5)


def sess(who="P00", regime="fast", idx=0, cond="normal"):
    return synth.session(IDS, 5, "test", who, regime, idx, cond)


def feats(s):
    v = ka.validate(s)
    f = ka.traj_features(ka.reaches(v["ticks"]))
    f.update(ka.burst_features(ka.annotate(v["events"])))
    return f


def population_spread():
    rows = [feats(sess(w, r)) for w in sorted(IDS)[:24] for r in synth.REGIMES]
    out = {}
    for f in ka.FEATURES:
        vals = [x[f] for x in rows if x[f] is not None]
        med = st.median(vals)
        out[f] = max(1e-6, st.median(abs(v - med) for v in vals) * 1.4826)
    return out


SPREAD = population_spread()


class TestValidation(unittest.TestCase):
    def base(self):
        return {"ticks": [[0, 0.1, 0.1, "move"], [10, 0.2, 0.2, "press"]],
                "events": [[0, "DIRECT_KEY", "INSERT", 1]]}

    def test_ok(self):
        ka.validate(self.base())

    def test_rejects(self):
        cases = [
            ("ticks", 0, [0, 1.2, 0.1, "move"]),
            ("ticks", 0, [0, 0.1, -0.01, "move"]),
            ("ticks", 0, [0, 0.1, 0.1, "hover"]),
            ("ticks", 1, [-5, 0.1, 0.1, "move"]),
            ("ticks", 0, [0, float("nan"), 0.1, "move"]),
            ("ticks", 0, [0, 0.1, 0.1]),
            ("events", 0, [0, "MORSE", "INSERT", 1]),
            ("events", 0, [0, "DIRECT_KEY", "UNDO", 1]),
            ("events", 0, [0, "DIRECT_KEY", "INSERT", 0]),
            ("events", 0, [0, "DIRECT_KEY", "INSERT", 1.5]),
        ]
        for key, i, val in cases:
            s = self.base()
            s[key][i] = val
            with self.assertRaises(ka.InputError, msg=str(val)):
                ka.validate(s)
        with self.assertRaises(ka.InputError):
            ka.validate({"ticks": []})

    def test_batch_insert_alias(self):
        s = self.base()
        s["events"] = [[0, "BATCH_INSERT", "INSERT", 3]]
        self.assertEqual(ka.validate(s)["events"][0][1], "BATCH_CHUNK")


class TestReachGeometry(unittest.TestCase):
    def line(self, n=50, lateral=0.0, overshoot=0.0):
        ticks = [[0, 0.2, 0.5, "release"]]
        for i in range(1, n + 1):
            u = i / n
            s = u + overshoot * math.sin(math.pi * u) if overshoot else u
            ticks.append([i * 10.0, 0.2 + 0.5 * s, 0.5 + lateral * math.sin(math.pi * u), "move"])
        ticks.append([n * 10.0 + 100, 0.7, 0.5, "press"])
        return ticks

    def test_straight_constant_speed(self):
        r = ka.reaches(ka.validate({"ticks": self.line(), "events": []})["ticks"])
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0]["curv"], 0.0, places=9)
        self.assertAlmostEqual(r[0]["half"], 0.5, delta=0.03)
        self.assertEqual(r[0]["over"], 0.0)

    def test_left_bow_is_positive(self):
        r = ka.reaches(ka.validate({"ticks": self.line(lateral=0.05), "events": []})["ticks"])
        self.assertGreater(r[0]["curv"], 0.05)  # moving +x, bowing +y = left of travel

    def test_short_reach_ignored(self):
        ticks = [[0, 0.5, 0.5, "release"], [10, 0.52, 0.5, "move"], [20, 0.53, 0.5, "move"],
                 [30, 0.54, 0.5, "move"], [40, 0.54, 0.5, "press"]]
        self.assertEqual(ka.reaches(ka.validate({"ticks": ticks, "events": []})["ticks"]), [])


class TestEdits(unittest.TestCase):
    def test_exact_counts(self):
        ev = []
        t = 0
        for burst in range(4):
            t += 2000                     # pause
            for i in range(5):
                ev.append([t, "DIRECT_KEY", "INSERT", 1])
                t += 100
        ev.insert(7, [ev[6][0] + 50, "DIRECT_KEY", "BACKSPACE", 1])
        ev.insert(8, [ev[7][0] + 50, "DIRECT_KEY", "BACKSPACE", 1])
        ev.append([t + 1500, "DIRECT_KEY", "REPLACE", 4])
        f = ka.burst_features(ka.annotate(ka.validate({"ticks": [], "events": ev})["events"]))
        self.assertEqual(f["edit"], 2 / 20)
        self.assertEqual(f["bs_len"], 2.0)
        self.assertEqual(f["repl"], 0.5)
        self.assertEqual(f["epause"], 0.5)     # the REPLACE came after a pause
        self.assertAlmostEqual(f["blen"], math.log(5))
        self.assertAlmostEqual(f["bcv"], 0.0)

    def test_batch_backspace_equals_run(self):
        a = [[0, "DIRECT_KEY", "INSERT", 1], [100, "DIRECT_KEY", "INSERT", 1],
             [200, "DIRECT_KEY", "BACKSPACE", 1], [250, "DIRECT_KEY", "BACKSPACE", 1],
             [300, "DIRECT_KEY", "BACKSPACE", 1]]
        b = [[0, "BATCH_CHUNK", "INSERT", 1], [100, "BATCH_CHUNK", "INSERT", 1],
             [200, "BATCH_CHUNK", "BACKSPACE", 3]]
        fa = ka.burst_features(ka.annotate(ka.validate({"ticks": [], "events": a})["events"]))
        fb = ka.burst_features(ka.annotate(ka.validate({"ticks": [], "events": b})["events"]))
        self.assertEqual((fa["bs_len"], fa["edit"]), (fb["bs_len"], fb["edit"]))


class TestInvariance(unittest.TestCase):
    def assertClose(self, a, b, tol):
        for f in ka.FEATURES:
            if a[f] is None or b[f] is None:
                self.assertEqual(a[f], b[f], f)
                continue
            self.assertLessEqual(abs(a[f] - b[f]), tol * SPREAD[f], f"{f}: {a[f]} vs {b[f]}")

    def test_time_scaling_exact(self):
        """Uniformly faster/slower hardware or operator: identical features."""
        for regime in synth.REGIMES:
            s = sess(regime=regime)
            t = copy.deepcopy(s)
            for row in t["ticks"] + t["events"]:
                row[0] = row[0] * 3.7
            self.assertClose(feats(s), feats(t), 1e-6)

    def test_time_translation_exact(self):
        s = sess()
        t = copy.deepcopy(s)
        for row in t["ticks"] + t["events"]:
            row[0] += 1e7
        self.assertClose(feats(s), feats(t), 1e-6)

    def test_channel_label_irrelevant(self):
        s = sess()
        t = copy.deepcopy(s)
        for row in t["events"]:
            row[1] = "VIRTUAL_GRID"
        self.assertEqual(feats(s), feats(t))

    def test_sensor_rate_downsampling(self):
        """8 ms ticks vs every 7th tick (~56 ms): trajectory features move by
        well under the population spread, averaged over 10 sessions. (Harsher
        than the real slow regime, where reaches are also 2.5-3.5x longer.)"""
        diffs = {f: [] for f in ka.FEATURES[:7]}
        for i in range(10):
            s = sess("P0%d" % (i % 8), "fast", i)
            ds, c = [], 0
            for tk in s["ticks"]:
                if tk[3] != "move":
                    ds.append(tk)
                else:
                    c += 1
                    if c % 7 == 0:
                        ds.append(tk)
            a, b = feats(s), feats({"ticks": ds, "events": s["events"]})
            for f in diffs:
                diffs[f].append(abs(a[f] - b[f]) / SPREAD[f])
        for f, d in diffs.items():
            self.assertLess(st.fmean(d), 0.75, f"{f}: {d}")

    def test_rechunking_direct_to_batch(self):
        """The same burst stream delivered as 3-symbol BATCH_CHUNKs keeps the
        burst features within a fraction of the population spread."""
        s = sess()
        ev = s["events"]
        out, buf = [], 0
        for i, e in enumerate(ev):
            if e[2] == "INSERT":
                buf += 1
                nxt = ev[i + 1] if i + 1 < len(ev) else None
                if buf == 3 or nxt is None or nxt[2] != "INSERT" or nxt[0] - e[0] > 1000:
                    out.append([e[0], "BATCH_CHUNK", "INSERT", buf])
                    buf = 0
            elif e[2] == "BACKSPACE" and out and out[-1][2] == "BACKSPACE" and not buf:
                out[-1][3] += 1
            else:
                out.append([e[0], "BATCH_CHUNK", e[2], e[3]])
        a = feats(s)
        b = feats({"ticks": s["ticks"], "events": out})
        for f in ("blen", "pause", "edit", "bs_len", "repl"):
            self.assertLessEqual(abs(a[f] - b[f]), 0.6 * SPREAD[f], f"{f}: {a[f]} vs {b[f]}")

    def test_regimes_overlap_for_same_identity(self):
        """Same identity, fast vs slow regime: median distance (in population
        spreads) is smaller than to other identities."""
        def dist(a, b):
            return st.fmean(abs(a[f] - b[f]) / SPREAD[f] for f in ka.FEATURES
                            if a[f] is not None and b[f] is not None)
        same, other = [], []
        for i in range(8):
            fa = feats(sess("P0%d" % i, "fast", 1))
            same.append(dist(fa, feats(sess("P0%d" % i, "slow", 1))))
            other.append(dist(fa, feats(sess("P0%d" % ((i + 3) % 8), "slow", 1))))
        self.assertLess(st.median(same), st.median(other))


class TestSynth(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(sess(), sess())
        self.assertNotEqual(sess(idx=0)["ticks"], sess(idx=1)["ticks"])

    def test_schema_and_regimes(self):
        f, s = sess(regime="fast"), sess(regime="slow")
        ka.validate(f)
        ka.validate(s)
        self.assertEqual({e[1] for e in f["events"]}, {"DIRECT_KEY"})
        self.assertTrue({e[1] for e in s["events"]} <= {"VIRTUAL_GRID", "BATCH_CHUNK"})
        self.assertTrue(any(e[3] > 1 for e in s["events"] if e[2] == "INSERT"))
        gap = lambda x: st.median(b[0] - a[0] for a, b in zip(x["ticks"], x["ticks"][1:]) if b[3] == "move")
        self.assertLess(gap(f), 10)
        self.assertGreater(gap(s), 40)
        for x in (f, s):
            self.assertTrue({e[2] for e in x["events"]} >= {"INSERT", "BACKSPACE"})

    def test_mimic_copies_visible_only(self):
        m = synth.make_mimic(IDS["P00"], IDS["M00"], 0.0)
        for k in synth.VISIBLE:
            self.assertEqual(m[k], IDS["P00"][k])
        self.assertEqual(m["curv"], IDS["M00"]["curv"])
        self.assertEqual(synth.make_mimic(IDS["P00"], IDS["M00"], 1.0), IDS["P00"])

    def test_disjoint_identity_pools(self):
        cal = synth.calibration_split(IDS, 5, n_gen=1, n_mimic=2)
        hold = synth.holdout_split(IDS, 5, n_gen=1, n_mimic=2)
        imp = lambda rows: {r["source"] for r in rows if r["condition"] in ("stranger", "mimic")}
        self.assertEqual(imp(cal) & imp(hold), set())


if __name__ == "__main__":
    unittest.main()
