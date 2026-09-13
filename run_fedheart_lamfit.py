"""Fed-Heart lambda_fit sweep under the cross-validated protocol.

The earlier sweep that put Fed-Heart's optimum at 0.3 ran on the superseded single-split
protocol, so its numbers are not comparable to anything we report now. Every arm on the site
uses lambda_fit 0.1, which is NHANES's optimum applied to Fed-Heart because
run_method_matrix.py holds one constant for both tabular datasets. This re-runs the sweep
under 5-fold CV to find out whether the anchors' penalty on Fed-Heart is real or an artifact
of borrowing the wrong value.
"""
from __future__ import annotations
import json, subprocess, sys

for lam in ["0.05", "0.1", "0.3", "0.5"]:
    print(f"\n=== Fed-Heart CV, lambda_fit={lam} ===", flush=True)
    subprocess.run([sys.executable, "run_fedheart_cv.py",
                    "--out", f"runs/fedheart_cv_lam{lam}",
                    "--seeds", "42", "1337", "7", "2024", "31337",
                    "--anchor-weight", lam], check=False)
