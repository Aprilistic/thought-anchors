"""Interactive Plotly visualizations for the rollout pipeline.

This is a lightweight bridge between the repo's current rollout artifacts:
- base_solution.json
- chunks.json
- chunks_labeled.json (from analyze_rollouts.py)

and interactive HTML exploration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import plotly.express as px
except Exception as e:  # pragma: no cover
    px = None
    _PLOTLY_IMPORT_ERROR = e


def _iter_problem_dirs(rollouts_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in rollouts_dir.iterdir()
            if p.is_dir() and p.name.startswith("problem_")
        ]
    )


def _load_chunks_labeled(problem_dir: Path) -> List[Dict]:
    fp = problem_dir / "chunks_labeled.json"
    if not fp.exists():
        return []
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def build_chunk_dataframe(
    rollouts_dir: Path, *, limit_problems: Optional[int] = None
) -> pd.DataFrame:
    rows: List[Dict] = []
    for pdir in _iter_problem_dirs(rollouts_dir)[: limit_problems or 10**9]:
        labeled = _load_chunks_labeled(pdir)
        for chunk in labeled:
            if not isinstance(chunk, dict):
                continue

            tags = chunk.get("function_tags", [])
            if not isinstance(tags, list):
                tags = []

            primary = None
            for t in tags:
                if isinstance(t, str) and t.strip() and t.strip().lower() != "unknown":
                    primary = t.strip()
                    break
            if primary is None:
                primary = "unknown"

            rows.append(
                {
                    "problem": pdir.name,
                    "chunk_idx": chunk.get("chunk_idx"),
                    "primary_tag": primary,
                    "all_tags": ",".join([t for t in tags if isinstance(t, str)]),
                    "summary": chunk.get("summary", ""),
                    "chunk": chunk.get("chunk", ""),
                    "accuracy": chunk.get("accuracy", None),
                    "score": chunk.get("score", None),
                    "different_trajectories_fraction": chunk.get(
                        "different_trajectories_fraction", None
                    ),
                    "overdeterminedness": chunk.get("overdeterminedness", None),
                    "counterfactual_importance_accuracy": chunk.get(
                        "counterfactual_importance_accuracy", None
                    ),
                    "counterfactual_importance_score": chunk.get(
                        "counterfactual_importance_score", None
                    ),
                    "resampling_importance_accuracy": chunk.get(
                        "resampling_importance_accuracy", None
                    ),
                    "resampling_importance_score": chunk.get(
                        "resampling_importance_score", None
                    ),
                    "forced_importance_accuracy": chunk.get(
                        "forced_importance_accuracy", None
                    ),
                    "forced_importance_score": chunk.get(
                        "forced_importance_score", None
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Plotly visualizations for rollouts"
    )
    parser.add_argument(
        "--rollouts-dir",
        type=str,
        required=True,
        help="Directory containing problem_* folders (e.g. .../correct_base_solution)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/interactive",
        help="Directory to write HTML outputs",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="different_trajectories_fraction",
        choices=[
            "different_trajectories_fraction",
            "overdeterminedness",
            "counterfactual_importance_accuracy",
            "counterfactual_importance_score",
            "resampling_importance_accuracy",
            "resampling_importance_score",
            "forced_importance_accuracy",
            "forced_importance_score",
        ],
        help="Y-axis metric to plot",
    )
    parser.add_argument(
        "--limit-problems",
        type=int,
        default=None,
        help="Only include first N problems (for quick runs)",
    )

    args = parser.parse_args()

    if px is None:
        raise RuntimeError(
            f"plotly is required for interactive viz: {_PLOTLY_IMPORT_ERROR}"
        )

    rollouts_dir = Path(args.rollouts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_chunk_dataframe(rollouts_dir, limit_problems=args.limit_problems)
    if df.empty:
        raise RuntimeError(f"No labeled chunks found under {rollouts_dir}")

    # Scatter: metric vs chunk index, colored by primary tag.
    fig = px.scatter(
        df,
        x="chunk_idx",
        y=args.metric,
        color="primary_tag",
        hover_data={
            "problem": True,
            "all_tags": True,
            "summary": True,
            "chunk": True,
            "chunk_idx": True,
        },
        title=f"{args.metric} by chunk index",
        template="plotly_white",
    )
    fig.write_html(out_dir / f"chunks_scatter_{args.metric}.html")

    # Box plot: metric distribution by tag.
    fig2 = px.box(
        df,
        x="primary_tag",
        y=args.metric,
        points="all",
        hover_data={
            "problem": True,
            "chunk_idx": True,
            "summary": True,
        },
        title=f"{args.metric} by tag",
        template="plotly_white",
    )
    fig2.update_layout(xaxis_tickangle=-45)
    fig2.write_html(out_dir / f"chunks_box_{args.metric}.html")

    print(f"Wrote interactive HTML to: {out_dir}")


if __name__ == "__main__":
    main()
