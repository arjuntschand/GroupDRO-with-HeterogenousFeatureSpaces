import csv
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunSummary:
    run_name: str
    run_path: str  # relative to repo root
    metrics_path: str
    epochs: int
    final_acc: Optional[float]
    final_worst_group_acc: Optional[float]
    best_worst_group_acc: Optional[float]
    final_loss: Optional[float]
    config_guess: Optional[str]
    updated_at: str


def find_repo_root(start: Path) -> Path:
    """Find repo root by looking for a 'runs' folder and the anchor package."""
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "dro_hetero_anchors").exists():
            return p
    return cur


def iter_run_dirs(repo_root: Path) -> List[Path]:
    candidates = [
        repo_root / "runs",
        repo_root / "dro_hetero_anchors" / "runs",
    ]
    out = []
    for c in candidates:
        if not c.exists():
            continue
        for child in sorted(c.iterdir()):
            if child.is_dir():
                out.append(child)
    return out


def parse_metrics_jsonl(jsonl_path: Path) -> Tuple[List[Dict], Optional[str]]:
    lines = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                # tolerate partial/corrupt lines
                continue
    # Try to find config path if present in any line
    config_guess = None
    for rec in lines:
        for key in ("config", "config_path", "experiment", "exp_config"):
            if key in rec and isinstance(rec[key], str) and rec[key].endswith((".yml", ".yaml")):
                config_guess = rec[key]
                break
        if config_guess:
            break
    return lines, config_guess


def summarize_run(run_dir: Path, repo_root: Path) -> Optional[RunSummary]:
    jsonl = run_dir / "metrics.jsonl"
    if not jsonl.exists():
        return None
    lines, config_guess = parse_metrics_jsonl(jsonl)
    if not lines:
        return None

    # Final metrics = last complete line
    final = lines[-1]
    final_acc = final.get("acc") or final.get("accuracy")
    final_worst = final.get("worst_group_acc") or final.get("worst_group_accuracy")
    final_loss = final.get("loss")

    # Best worst-group across epochs
    best_worst = None
    for rec in lines:
        w = rec.get("worst_group_acc") or rec.get("worst_group_accuracy")
        if isinstance(w, (int, float)):
            best_worst = w if best_worst is None else max(best_worst, w)

    epochs = 0
    for rec in lines:
        if isinstance(rec.get("epoch"), int):
            epochs = max(epochs, rec["epoch"])

    rel_path = run_dir.relative_to(repo_root).as_posix()
    rel_metrics = jsonl.relative_to(repo_root).as_posix()

    return RunSummary(
        run_name=run_dir.name,
        run_path=rel_path,
        metrics_path=rel_metrics,
        epochs=epochs,
        final_acc=final_acc if isinstance(final_acc, (int, float)) else None,
        final_worst_group_acc=final_worst if isinstance(final_worst, (int, float)) else None,
        best_worst_group_acc=best_worst,
        final_loss=final_loss if isinstance(final_loss, (int, float)) else None,
        config_guess=config_guess,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )


def write_csv(summaries: List[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(summaries[0]).keys()) if summaries else [
                "run_name", "run_path", "metrics_path", "epochs", "final_acc",
                "final_worst_group_acc", "best_worst_group_acc", "final_loss",
                "config_guess", "updated_at"
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))


def write_sqlite(summaries: List[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS run_summary (
                run_name TEXT PRIMARY KEY,
                run_path TEXT,
                metrics_path TEXT,
                epochs INTEGER,
                final_acc REAL,
                final_worst_group_acc REAL,
                best_worst_group_acc REAL,
                final_loss REAL,
                config_guess TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
        # Upsert rows
        for s in summaries:
            cur.execute(
                """
                INSERT INTO run_summary (
                    run_name, run_path, metrics_path, epochs, final_acc,
                    final_worst_group_acc, best_worst_group_acc, final_loss,
                    config_guess, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_name) DO UPDATE SET
                    run_path=excluded.run_path,
                    metrics_path=excluded.metrics_path,
                    epochs=excluded.epochs,
                    final_acc=excluded.final_acc,
                    final_worst_group_acc=excluded.final_worst_group_acc,
                    best_worst_group_acc=excluded.best_worst_group_acc,
                    final_loss=excluded.final_loss,
                    config_guess=excluded.config_guess,
                    updated_at=excluded.updated_at
                """,
                (
                    s.run_name, s.run_path, s.metrics_path, s.epochs, s.final_acc,
                    s.final_worst_group_acc, s.best_worst_group_acc, s.final_loss,
                    s.config_guess, s.updated_at,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    run_dirs = iter_run_dirs(repo_root)

    summaries: List[RunSummary] = []
    for rd in run_dirs:
        s = summarize_run(rd, repo_root)
        if s is not None:
            summaries.append(s)

    # Sort by updated_at then name
    summaries.sort(key=lambda x: (x.updated_at, x.run_name))

    # Outputs under repo_root/runs
    out_csv = repo_root / "runs" / "index.csv"
    out_sqlite = repo_root / "runs" / "metrics.sqlite"

    write_csv(summaries, out_csv)
    write_sqlite(summaries, out_sqlite)

    print(f"Indexed {len(summaries)} runs")
    print(f"CSV: {out_csv.relative_to(repo_root)}")
    print(f"SQLite: {out_sqlite.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
