"""Generate a LaTeX contribution table from Git history."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


MODULE_PREFIXES = (
    ("src/evaluation/", "evaluation"),
    ("src/data.py", "data.py"),
    ("src/models.py", "models.py"),
    ("src/losses.py", "losses.py"),
    ("src/baselines.py", "baselines.py"),
    ("src/train.py", "train.py"),
    ("src/tune.py", "tune.py"),
    ("configs/", "configs"),
    ("report/", "report"),
    ("results/", "results"),
    ("data_scripts/", "data_scripts"),
    ("README.md", "documentation"),
)

DEFAULT_ALIASES = {
    "fahdbusiness125@gmail.com": "Rai Fahd Sultan",
    "raifahd": "Rai Fahd Sultan",
    "TahaImran1": "Taha Imran",
    "tahaimran561177@gmail.com": "Taha Imran",
}


def map_file_to_module(file_path: str) -> str:
    """Map a repository path to a project module, or mark it as other."""
    normalized = file_path.replace("\\", "/")
    for prefix, module in MODULE_PREFIXES:
        if normalized.startswith(prefix):
            return module
    return "other"


def collect_git_contributions(
    repo_root: Path,
    author_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Collect commits and added lines grouped by author and project module."""
    command = [
        "git",
        "-C",
        str(repo_root),
        "log",
        "--numstat",
        "--format=COMMIT%x09%H%x09%an%x09%ae",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    aliases = {**DEFAULT_ALIASES, **(author_aliases or {})}
    rows: list[dict[str, Any]] = []
    current: dict[str, str] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith("COMMIT\t"):
            _, commit, name, email = line.split("\t", maxsplit=3)
            current = {
                "commit": commit,
                "author": aliases.get(email, aliases.get(name, name)),
                "email": email,
            }
            continue
        if not current or not line.strip():
            continue
        match = re.match(r"^(\d+|-)\s+(\d+|-)\s+(.+)$", line)
        if not match:
            continue
        added = 0 if match.group(1) == "-" else int(match.group(1))
        file_path = match.group(3)
        rows.append(
            {
                **current,
                "module": map_file_to_module(file_path),
                "added_lines": added,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["author", "email", "module", "commits", "lines_added"]
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["author", "email", "module"], as_index=False)
        .agg(commits=("commit", "nunique"), lines_added=("added_lines", "sum"))
        .sort_values(["author", "module"])
        .reset_index(drop=True)
    )


def write_contribution_table(contributions: pd.DataFrame, output_path: Path) -> None:
    """Write contribution data as a compact booktabs LaTeX table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = contributions.to_latex(
        index=False,
        escape=True,
        column_format="llrrr",
        caption="Project contribution by author and module",
        label="tab:contributions",
    )
    table = table.replace("\\begin{table}\n", "\\begin{table}[htbp]\n\\centering\n")
    output_path.write_text(table, encoding="utf-8")


def _load_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Author alias config must be a mapping.")
    return {str(key): str(value) for key, value in loaded.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the contribution table.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--aliases", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report") / "tables" / "contribution_table.tex",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    contributions = collect_git_contributions(root, _load_aliases(args.aliases))
    write_contribution_table(contributions, output)
    print(f"Wrote contribution table to {output}")


if __name__ == "__main__":
    main()
