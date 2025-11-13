import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResultsLogger:
    """Log training/evaluation metrics in JSONL and CSV for easy aggregation."""

    def __init__(self, run_dir: str, filename_stem: str = "metrics"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / f"{filename_stem}.jsonl"
        self.csv_path = self.run_dir / f"{filename_stem}.csv"
        self._buffer: List[Dict[str, Any]] = []
        self._csv_header_written = False

    def log_epoch(self, record: Dict[str, Any]):
        """Append a record for an epoch. Call save() occasionally to persist."""
        self._buffer.append(record)

    @staticmethod
    def _flatten(value: Any) -> Any:
        # For CSV, convert complex structures to JSON strings.
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    def save(self):
        if not self._buffer:
            return
        # Append to JSONL
        with open(self.jsonl_path, "a") as jf:
            for rec in self._buffer:
                jf.write(json.dumps(rec) + "\n")

        # Append to CSV (write header if new)
        # Use union of keys across buffered records for robust header
        all_keys = []
        seen = set()
        for rec in self._buffer:
            for k in rec.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

        write_header = not self._csv_header_written or not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=all_keys)
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            for rec in self._buffer:
                flat = {k: self._flatten(rec.get(k)) for k in all_keys}
                writer.writerow(flat)

        # Clear buffer after persisting
        self._buffer.clear()
