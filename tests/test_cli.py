"""One-command offline run, file round trip, enroll/score CLI."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args, cwd=ROOT, timeout=300):
    return subprocess.run([PY, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)


class TestCLI(unittest.TestCase):
    def test_no_network_modules(self):
        p = run("-c", "import sys, kinematics_auth, synth_traces; "
                      "print(sorted(m for m in ('socket','ssl','urllib.request','http.client') if m in sys.modules))")
        self.assertEqual(p.stdout.strip(), "[]")

    def test_synth_files_and_data_dir_bench(self):
        with tempfile.TemporaryDirectory() as d:
            p = run("synth_traces.py", "--seed", "77", "--out", d)
            self.assertEqual(p.returncode, 0, p.stderr)
            digest = {f: hashlib.sha256(Path(d, f).read_bytes()).hexdigest() for f in sorted(os.listdir(d))}
            self.assertEqual(sorted(digest), ["calibration.jsonl", "enroll.jsonl", "holdout.jsonl"])
            p2 = run("synth_traces.py", "--seed", "77", "--out", d)
            digest2 = {f: hashlib.sha256(Path(d, f).read_bytes()).hexdigest() for f in sorted(os.listdir(d))}
            self.assertEqual(digest, digest2)  # byte-identical regeneration
            b = run("kinematics_auth.py", "bench", "--data", d, "--seed", "77")
            self.assertEqual(b.returncode, 0, b.stderr)
            self.assertIn("POOLED FAR", b.stdout)
            mem = run("-c", "import kinematics_auth as k; r=k.run_benchmark(77, alphas=(0.0,), log=lambda *a: None);"
                            "print(k.format_report(r).split('swap matrix')[0])")
            table_file = b.stdout.split("seed=")[1].split("swap matrix")[0]
            table_mem = mem.stdout.split("seed=")[1]
            self.assertEqual(table_file.strip(), table_mem.strip())  # files == in-memory pipeline

    def test_enroll_and_score(self):
        with tempfile.TemporaryDirectory() as d:
            run("synth_traces.py", "--seed", "5", "--out", d)
            model = os.path.join(d, "model.json")
            e = run("kinematics_auth.py", "enroll", "--data", os.path.join(d, "enroll.jsonl"), "--out", model)
            self.assertEqual(e.returncode, 0, e.stderr)
            self.assertIn("enrolled 8 profiles", e.stdout)
            # score the first 3 holdout sessions
            hold = Path(d, "holdout.jsonl").read_text().splitlines()[:3]
            Path(d, "few.jsonl").write_text("\n".join(hold) + "\n")
            s = run("kinematics_auth.py", "score", "--model", model, "--data", os.path.join(d, "few.jsonl"))
            self.assertEqual(s.returncode, 0, s.stderr)
            self.assertEqual(len(s.stdout.strip().splitlines()), 3)
            self.assertIn('"score"', s.stdout)

    def test_bad_input_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d, "bad.jsonl")
            bad.write_text('{"ticks": [[0, 2.0, 0.1, "move"]], "events": [], "claimed": "P00"}\n')
            run("synth_traces.py", "--seed", "5", "--out", d)
            model = os.path.join(d, "model.json")
            run("kinematics_auth.py", "enroll", "--data", os.path.join(d, "enroll.jsonl"), "--out", model)
            s = run("kinematics_auth.py", "score", "--model", model, "--data", str(bad))
            self.assertNotEqual(s.returncode, 0)
            self.assertIn("outside [0,1]", s.stderr)


if __name__ == "__main__":
    unittest.main()
