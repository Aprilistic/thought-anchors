import sys
import os
import sys
import json
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.attn_funcs import get_avg_attention_matrix
from attention_analysis.receiver_head_funcs import (
    get_problem_text_sentences,
    get_top_k_receiver_heads,
    get_model_rollouts_root,
)
from pytorch_models.model_config import model2layers_heads


def white_to_blues(N: int = 256) -> mcolors.LinearSegmentedColormap:
    """Create a custom colormap that transitions from white to blues."""
    blues = plt.cm.Blues

    blues_colors = blues(np.linspace(0, 1, N))

    colors = [(1, 1, 1), (0, 0, 1)]  # White to Blue
    white_blue_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", colors)
    white_color = white_blue_cmap(np.linspace(0, 1, N))

    white_weights = np.linspace(1, 0, N)[:, np.newaxis]
    blues_weights = 1 - white_weights

    blended_colors = white_weights * white_color + blues_weights * blues_colors

    return mcolors.LinearSegmentedColormap.from_list("WhiteToBlues", blended_colors)


def plot_one_attn_mtx(
    problem_num: int = 4682,
    is_correct: bool = True,
    layer: Optional[int] = 36,
    head: Optional[int] = 6,
    top_k: Optional[int] = None,
    model_name: str = "qwen-14b",
    highlight_tag: Optional[str] = None,
    highlight_axis: str = "both",
    highlight_color: str = "red",
    highlight_alpha: float = 0.18,
    max_size: Optional[int] = 129,
    no_show: bool = False,
    output_dir: str = "plots",
    dpi: int = 300,
) -> None:
    """
    Plot attention matrix for a specific problem and layer/head combination.

    Args:
        problem_num: Problem number
        is_correct: Whether to use correct or incorrect solution
        layer: Layer index (or None if using top_k)
        head: Head index (or None if using top_k)
        top_k: Number of top receiver heads to average (if None, use specific layer/head)
        model_name: Model name
    """
    text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)

    # Load chunk labels (best-effort)
    tag_mask = None
    if highlight_tag:
        dir_root = get_model_rollouts_root(model_name)
        ci = "correct_base_solution" if is_correct else "incorrect_base_solution"
        fp_labels = os.path.join(
            dir_root, ci, f"problem_{problem_num}", "chunks_labeled.json"
        )
        try:
            with open(fp_labels, "r", encoding="utf-8") as f:
                labeled = json.load(f)
            if isinstance(labeled, list):
                tag = highlight_tag.strip().lower()
                tag_mask = np.array(
                    [
                        any(
                            isinstance(t, str) and t.strip().lower() == tag
                            for t in (
                                c.get("function_tags", [])
                                if isinstance(c, dict)
                                else []
                            )
                        )
                        for c in labeled
                    ],
                    dtype=bool,
                )
        except Exception:
            tag_mask = None

    if top_k is not None:
        coords = get_top_k_receiver_heads(
            model_name=model_name,
            top_k=top_k,
            proximity_ignore=4,
            control_depth=False,
        )
    else:
        coords = [(layer, head)]

    avg_matrix_list = []
    for layer_idx, head_idx in coords:
        avg_matrix = get_avg_attention_matrix(
            text,
            model_name,
            layer_idx,
            head_idx,
            sentences=sentences,
        )
        avg_matrix_list.append(avg_matrix)

    if len(avg_matrix_list) > 1:
        avg_matrix = np.nanmean(avg_matrix_list, axis=0)
    else:
        avg_matrix = avg_matrix_list[0]

    # Old pipeline included prompt/output bins at first/last positions.
    # Keep this behavior, but try to keep tag_mask aligned.
    if avg_matrix.shape[0] >= 3:
        avg_matrix = avg_matrix[1:-1, 1:-1]
        if tag_mask is not None and tag_mask.shape[0] == avg_matrix.shape[0] + 2:
            tag_mask = tag_mask[1:-1]

    # Optionally limit size for visualization
    if max_size is not None:
        avg_matrix = avg_matrix[:max_size, :max_size]
        if tag_mask is not None:
            tag_mask = tag_mask[:max_size]

    avg_matrix_tril = np.tril(avg_matrix)

    if top_k is not None:
        vmin = np.nanquantile(avg_matrix_tril, 0.1)
        vmax = np.nanquantile(avg_matrix_tril, 0.9)
    else:
        vmin = np.nanquantile(avg_matrix_tril, 0.01)
        vmax = np.nanquantile(avg_matrix_tril, 0.99)

    white_blue_cmap = white_to_blues()

    plt.imshow(avg_matrix, vmin=0, vmax=vmax, cmap=white_blue_cmap)

    # Overlay taxonomy highlights
    if tag_mask is not None and tag_mask.shape[0] == avg_matrix.shape[0]:
        axis = (highlight_axis or "both").lower()
        for idx, is_hit in enumerate(tag_mask):
            if not is_hit:
                continue
            if axis in ("rows", "both"):
                plt.gca().axhspan(
                    idx - 0.5,
                    idx + 0.5,
                    color=highlight_color,
                    alpha=highlight_alpha,
                    linewidth=0,
                )
            if axis in ("cols", "both"):
                plt.gca().axvspan(
                    idx - 0.5,
                    idx + 0.5,
                    color=highlight_color,
                    alpha=highlight_alpha,
                    linewidth=0,
                )

    if is_correct:
        problem_title = f"Problem: {problem_num} (correct)"
    else:
        problem_title = f"Problem: {problem_num} (incorrect)"

    if top_k is not None:
        title = f"{problem_title}\nTop {top_k} receiver heads"
    else:
        title = f"{problem_title}\nLayer: {layer}, Head: {head}"

    plt.title(title, fontsize=12)

    plt.gca().tick_params(axis="both", labelsize=11)
    xticks = np.arange(0, avg_matrix.shape[1], 25)
    plt.gca().set_xticks(xticks)
    yticks = np.arange(0, avg_matrix.shape[0], 25)
    plt.gca().set_yticks(yticks)
    plt.gca().set_xlim(0, avg_matrix.shape[1] - 1)
    plt.gca().set_ylim(avg_matrix.shape[0] - 1, 0)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.xlabel("Sentence position", fontsize=12)
    plt.ylabel("Sentence position", fontsize=12)

    plt.tight_layout()

    out_base = Path(output_dir)
    if top_k is not None:
        fp_out = (
            out_base
            / "attn_matrix"
            / f"pn_{problem_num}-{is_correct}_head_{layer}-{head}_top{top_k}_{model_name}.png"
        )
    else:
        fp_out = (
            out_base
            / "attn_avg_matrix"
            / f"pn_{problem_num}-{is_correct}_head_{layer}-{head}_{model_name}.png"
        )
    fp_out.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(fp_out, dpi=dpi)
    if not no_show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot attention matrix for a specific problem"
    )
    parser.add_argument(
        "--problem-num", type=int, default=0, help="Problem number to analyze"
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        help="Use correct solution (default: incorrect)",
    )
    parser.add_argument(
        "--layer", type=int, default=None, help="Layer index (ignored if using top-k)"
    )
    parser.add_argument(
        "--head", type=int, default=None, help="Head index (ignored if using top-k)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Number of top receiver heads to average (set to 0 to use specific layer/head)",
    )
    parser.add_argument("--model-name", type=str, default="qwen-15b", help="Model name")
    parser.add_argument(
        "--proximity-ignore",
        type=int,
        default=4,
        help="Proximity ignore for receiver heads",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=129,
        help="Maximum matrix size for visualization",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Output directory for plots"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[3, 3],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display plot, only save"
    )

    parser.add_argument(
        "--highlight-tag",
        type=str,
        default=None,
        help="If set, highlight rows/cols for chunks with this function tag",
    )
    parser.add_argument(
        "--highlight-axis",
        type=str,
        choices=["rows", "cols", "both"],
        default="both",
        help="Whether to highlight rows, cols, or both",
    )
    parser.add_argument(
        "--highlight-color",
        type=str,
        default="red",
        help="Highlight color",
    )
    parser.add_argument(
        "--highlight-alpha",
        type=float,
        default=0.18,
        help="Highlight alpha",
    )

    args = parser.parse_args()

    # Handle top_k vs specific layer/head
    if args.top_k > 0:
        top_k = args.top_k
        layer = None
        head = None
    else:
        top_k = None
        layer = args.layer if args.layer is not None else 36
        head = args.head if args.head is not None else 6

    plt.figure(figsize=tuple(args.figsize))

    plot_one_attn_mtx(
        problem_num=args.problem_num,
        is_correct=args.correct,
        layer=layer,
        head=head,
        top_k=top_k,
        model_name=args.model_name,
        highlight_tag=args.highlight_tag,
        highlight_axis=args.highlight_axis,
        highlight_color=args.highlight_color,
        highlight_alpha=args.highlight_alpha,
        max_size=args.max_size,
        no_show=args.no_show,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
