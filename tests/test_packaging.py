import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PackagingTests(unittest.TestCase):
    def test_cli_help(self):
        for name in ("step1_dense_blocker", "step2_cross_encoder", "step3_matcher"):
            result = subprocess.run([sys.executable, str(ROOT / "src" / (name + ".py")), "--help"], capture_output=True, text=True, check=True)
            self.assertIn("--data-dir", result.stdout)

    def test_defaults_are_inside_repository(self):
        env = {k:v for k,v in os.environ.items() if not k.startswith("BER_")}
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run([sys.executable, "-c", "from ber.paths import WORK, OUT; print(WORK); print(OUT)"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), [str(ROOT / "work"), str(ROOT / "output")])

    def test_archive_layout_with_synthetic_data(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            shutil.copytree(ROOT / "src", p / "src", ignore=shutil.ignore_patterns("__pycache__"))
            for name in ("README.md", "TECHNICAL_REPORT.md", "requirements.txt", "run_pipeline.sh", "read_before_run.md"):
                shutil.copy2(ROOT / name, p / name)
            (p / "output").mkdir()
            for filename, column in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
                (p / "output" / filename).write_text("source1_entity_id\t" + column + "\nS1_1\tS2_1\n", encoding="utf-8")
            (p / ".env").write_text("SYNTHETIC_SECRET=not-for-archive", encoding="utf-8")
            result = subprocess.run([sys.executable, str(p / "src/make_submission_zip.py"), "--team", "smoke"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with zipfile.ZipFile(p / "smoke_submission.zip") as z:
                names = z.namelist()
                self.assertIn("Documentation_template.md", names)
                self.assertIn("code/business_entity_resolution/src/ber/paths.py", names)
                self.assertIn("output/candidate_pairs.tsv", names)
                self.assertFalse(any(".env" in n or "__pycache__" in n for n in names))
