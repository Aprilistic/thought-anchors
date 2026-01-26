import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_str(val: Any) -> str:
    return val if isinstance(val, str) else ""


def extract_instruction_from_prompt(prompt: str) -> str:
    """Best-effort extraction of user instruction from a manual chat-template prompt."""

    if not prompt:
        return ""

    # vLLM Qwen chat-template style
    s = "<|im_start|>user\n"
    e = "<|im_end|>"
    if s in prompt:
        after = prompt.split(s, 1)[1]
        if e in after:
            return after.split(e, 1)[0].strip()

    # Fallback: older wrapper prompts
    if "Instruction:" in prompt:
        after = prompt.split("Instruction:", 1)[1]
        return after.split("\n", 1)[0].strip()

    return prompt.strip()


def find_problem_dirs(rollouts_dir: Path) -> List[Tuple[str, Path]]:
    """Return (ci, problem_dir) tuples."""

    out: List[Tuple[str, Path]] = []

    if (rollouts_dir / "correct_base_solution").is_dir() or (
        rollouts_dir / "incorrect_base_solution"
    ).is_dir():
        for ci in ("correct_base_solution", "incorrect_base_solution"):
            ci_dir = rollouts_dir / ci
            if not ci_dir.is_dir():
                continue
            for p in sorted(ci_dir.iterdir()):
                if p.is_dir() and p.name.startswith("problem_"):
                    out.append((ci, p))
        return out

    # If user points directly at correct_base_solution/problem_*...
    if rollouts_dir.is_dir() and rollouts_dir.name in (
        "correct_base_solution",
        "incorrect_base_solution",
    ):
        ci = rollouts_dir.name
        for p in sorted(rollouts_dir.iterdir()):
            if p.is_dir() and p.name.startswith("problem_"):
                out.append((ci, p))
        return out

    # If user points at a single problem_ folder
    if rollouts_dir.is_dir() and rollouts_dir.name.startswith("problem_"):
        out.append(("unknown", rollouts_dir))
        return out

    raise ValueError(
        f"Unrecognized rollouts directory layout: {rollouts_dir} (expected a directory containing correct_base_solution/ or problem_*)"
    )


def build_problem_payload(ci: str, problem_dir: Path) -> Dict[str, Any]:
    base_solution_path = problem_dir / "base_solution.json"
    base_solution: Dict[str, Any] = {}
    if base_solution_path.exists():
        data = _read_json(base_solution_path)
        base_solution = data if isinstance(data, dict) else {}

    instruction = ""
    if "instruction" in base_solution:
        instruction = _safe_str(base_solution.get("instruction"))
    if not instruction:
        instruction = extract_instruction_from_prompt(
            _safe_str(base_solution.get("prompt"))
        )

    labeled_path = problem_dir / "chunks_labeled.json"
    chunks_path = problem_dir / "chunks.json"
    labeled = []
    if labeled_path.exists():
        labeled = _read_json(labeled_path)
        if not isinstance(labeled, list):
            labeled = []

    chunks_fallback: List[str] = []
    if chunks_path.exists():
        chunks_json = _read_json(chunks_path)
        if isinstance(chunks_json, dict) and isinstance(
            chunks_json.get("chunks"), list
        ):
            chunks_fallback = [
                _safe_str(x) for x in chunks_json.get("chunks", []) if _safe_str(x)
            ]

    chunks_out: List[Dict[str, Any]] = []
    if labeled:
        for idx, c in enumerate(labeled):
            if not isinstance(c, dict):
                continue
            tags = c.get("function_tags", [])
            if not isinstance(tags, list):
                tags = []

            depends = c.get("depends_on", [])
            if not isinstance(depends, list):
                depends = []

            depends_int: List[int] = []
            for d in depends:
                try:
                    depends_int.append(int(d))
                except Exception:
                    continue

            chunks_out.append(
                {
                    "idx": c.get("chunk_idx", idx),
                    "chunk": _safe_str(c.get("chunk")),
                    "summary": _safe_str(c.get("summary")),
                    "function_tags": [t for t in tags if isinstance(t, str)],
                    "depends_on": depends_int,
                    "metrics": {
                        k: v
                        for k, v in c.items()
                        if isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and math.isfinite(float(v))
                        and (
                            k
                            in (
                                "accuracy",
                                "score",
                                "different_trajectories_fraction",
                                "overdeterminedness",
                            )
                            or k.endswith("_accuracy")
                            or k.endswith("_score")
                            or k.endswith("_kl")
                        )
                    },
                }
            )
    elif chunks_fallback:
        for idx, text in enumerate(chunks_fallback):
            chunks_out.append(
                {
                    "idx": idx,
                    "chunk": text,
                    "summary": "",
                    "function_tags": [],
                    "metrics": {},
                }
            )

    pid = problem_dir.name
    full_cot = _safe_str(base_solution.get("full_cot"))
    if not full_cot:
        # fallback to prompt+solution
        full_cot = _safe_str(base_solution.get("prompt")) + _safe_str(
            base_solution.get("solution")
        )

    # Build edges (sequential + DAG)
    edges: List[Dict[str, Any]] = []
    # sequential edges
    for i in range(len(chunks_out) - 1):
        edges.append(
            {
                "source": int(chunks_out[i]["idx"]),
                "target": int(chunks_out[i + 1]["idx"]),
                "type": "sequential",
            }
        )
    # depends_on edges (dep -> node)
    for c in chunks_out:
        tgt = int(c.get("idx", 0))
        for dep in c.get("depends_on", []) or []:
            try:
                edges.append({"source": int(dep), "target": tgt, "type": "causal"})
            except Exception:
                continue

    return {
        "id": pid,
        "ci": ci,
        "instruction": instruction,
        "nickname": _safe_str(base_solution.get("nickname")),
        "full_cot": full_cot,
        "chunks": chunks_out,
        "edges": edges,
    }


def make_title(problem: Dict[str, Any]) -> str:
    nick = _safe_str(problem.get("nickname"))
    if nick:
        return nick

    instr = _safe_str(problem.get("instruction")).replace("\n", " ").strip()
    if not instr:
        return _safe_str(problem.get("id"))
    if len(instr) <= 60:
        return instr
    return instr[:57] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export rollouts to a web-friendly JSON bundle"
    )
    parser.add_argument(
        "--rollouts-dir",
        type=str,
        required=True,
        help="Path to rollouts root or correct_base_solution directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="web/data",
        help="Where to write web JSON data",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Only export first N problems",
    )

    args = parser.parse_args()
    rollouts_dir = Path(args.rollouts_dir)
    out_dir = Path(args.output_dir)

    problems = find_problem_dirs(rollouts_dir)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    problems_out_dir = out_dir / "problems"
    problems_out_dir.mkdir(parents=True, exist_ok=True)

    index_items: List[Dict[str, Any]] = []
    for ci, pdir in problems:
        payload = build_problem_payload(ci, pdir)
        title = make_title(payload)
        pid = payload["id"]
        out_path = problems_out_dir / f"{pid}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Basic tag histogram for quick filtering
        tag_counts: Dict[str, int] = {}
        for c in payload.get("chunks", []):
            for t in c.get("function_tags", []) or []:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        index_items.append(
            {
                "id": pid,
                "ci": ci,
                "title": title,
                "num_chunks": len(payload.get("chunks", [])),
                "tag_counts": tag_counts,
                "path": f"problems/{pid}.json",
            }
        )

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(index_items),
        "problems": index_items,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(index_items)} problems to {out_dir}")


if __name__ == "__main__":
    main()
