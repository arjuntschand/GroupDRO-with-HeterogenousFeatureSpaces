"""(Moved to testFiles/) Smoke test: run a short training config and assert checkpoint creation.

This is not a unit test harness; consider replacing with pytest-based tests under `tests/`.
"""

import os
import subprocess

def test_training():
    # Remove previous runs
    if os.path.exists('runs'):
        subprocess.run(['rm', '-rf', 'runs'])
    # Run training for a few epochs
    result = subprocess.run([
        'python', '-m', 'dro_hetero_anchors.src.train', '--config', 'experiments/digits_centralized.yaml'
    ], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    # Check for output files
    assert os.path.exists('runs/last.ckpt'), "Checkpoint not created!"
    print("Training test passed: checkpoint created.")

if __name__ == "__main__":
    test_training()
