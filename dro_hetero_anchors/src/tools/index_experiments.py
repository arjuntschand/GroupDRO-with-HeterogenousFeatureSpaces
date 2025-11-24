import csv
from pathlib import Path
from typing import List


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "dro_hetero_anchors").exists():
            return p
    return cur


def list_experiment_files(exp_dir: Path) -> List[Path]:
    return sorted([p for p in exp_dir.glob("*.y*ml") if p.is_file()])


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    # Experiments now live at the repository root for symmetry with `runs/`
    exp_dir = repo_root / "experiments"
    out_csv = exp_dir / "INDEX.csv"

    files = list_experiment_files(exp_dir)

    # Heuristic title from filename
    def title_for(p: Path) -> str:
        name = p.stem.replace("_", " ")
        return name

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "title", "notes"],
        )
        writer.writeheader()
        for p in files:
            writer.writerow({
                "file": p.name,
                "title": title_for(p),
                "notes": "",
            })

    print(f"Wrote {out_csv.relative_to(repo_root)} with {len(files)} entries")


if __name__ == "__main__":
    main()
