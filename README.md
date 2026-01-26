# Thought Anchors ⚓

## Fork Notes (Custom Pipeline)

This fork modifies the original project to support running experiments on a custom JSONL dataset (e.g. `fortress_prompts.jsonl`) and to track **verbalized evaluation awareness** during reasoning.

### Environment (uv)

```bash
uv sync
```

### Runtime Assumptions

- Generation uses a local OpenAI-compatible vLLM server.
- Judge/classifier uses a local OpenAI-compatible vLLM *chat* server.
- Embeddings are computed locally via `sentence-transformers` (default: `all-MiniLM-L6-v2`).

Env vars:

- `VLLM_GENERATION_BASE_URL` (e.g. `http://0.0.0.0:8001/v1`)
- `VLLM_JUDGE_BASE_URL` (defaults to generation base url)
- `VLLM_API_KEY` (optional; vLLM often ignores it but OpenAI SDK requires a non-empty string)
- `VLLM_GENERATION_MODEL` (default `Qwen/Qwen3-4B-Thinking-2507`)
- `VLLM_JUDGE_MODEL` (default `Qwen/Qwen3-Next-80B-A3B-Instruct`)

### Run: Generate Rollouts From JSONL

`generate_rollouts.py` supports `--dataset jsonl` (default) and reads `--dataset_jsonl`.

Important: For vLLM, this fork uses `/v1/completions` and a manual Qwen chat-template prompt (so rollouts are true continuations).

```bash
uv run python generate_rollouts.py \
  --dataset jsonl \
  --dataset_jsonl fortress_prompts.jsonl \
  --provider vLLM \
  --num_problems 50 \
  --output_dir rollouts
```

Useful filters:

```bash
# Use specific IDs from the JSONL's "problem_idx" field
uv run python generate_rollouts.py --include_problems fortress_2,fortress_10

# Exclude specific IDs
uv run python generate_rollouts.py --exclude_problems fortress_3
```

Note: If your dataset has no ground-truth answers, `--base_solution_type correct|incorrect` is treated as a label only; it will not retry until correctness.

### Run: Label + Analyze

```bash
uv run python analyze_rollouts.py \
  --correct_rollouts_dir rollouts/<MODEL>/temperature_0.6_top_p_0.95/correct_base_solution \
  --output_dir analysis
```

If your dataset has no ground-truth answers, the analysis automatically switches from accuracy-based importance to `different_trajectories_fraction`.

### Whitebox Analysis

Whitebox scripts assume a `rollouts/<model>/temperature_.../correct_base_solution/problem_*` structure.

For Qwen3-4B:

```bash
export WHITEBOX_ROLLOUTS_ROOT="rollouts/Qwen3-4B-Thinking-2507/temperature_0.6_top_p_0.95"

uv run python whitebox-analyses/scripts/prep_attn_cache.py --model qwen3-4b
uv run python whitebox-analyses/scripts/generate_rec_csvs.py --model-name qwen3-4b --data-model qwen3-4b --output-dir csvs
```

### Label Taxonomy Change

The labeling prompt in `prompts.py` replaces `active_computation` with:

- `verbalized_evaluation_awareness`

Downstream plotting/whitebox taxonomy mappings were updated accordingly.

### Interactive Visualization (Rollouts)

This fork adds a Plotly-based interactive viewer for rollout chunk labels/metrics.

```bash
uv run python masking_graphs/interactive_rollout_viz.py \
  --rollouts-dir rollouts/Qwen3-4B-Thinking-2507/temperature_0.6_top_p_0.95/correct_base_solution \
  --output-dir analysis/interactive \
  --metric different_trajectories_fraction
```

### Interactive Web Page (thought-anchors.com-style)

This fork also includes a small static site under `web/` that loads exported JSON and renders a 3-column dashboard (chunk list, circular causal graph from `depends_on`, metric plot).

1) Export data from a rollouts directory:

```bash
uv run python scripts/build_web_data.py \
  --rollouts-dir rollouts/Qwen3-4B-Thinking-2507/temperature_0.6_top_p_0.95/correct_base_solution \
  --output-dir web/data
```

2) Serve the site locally:

```bash
cd web
uv run python -m http.server 8000
```

Open `http://localhost:8000`.

---

We introduce a framework for interpreting the reasoning of large language models by attributing importance to individual sentences in their chain-of-thought. Using black-box, attention-based, and causal methods, we identify key reasoning steps, which we call **thought anchors**, that disproportionately influence downstream reasoning. These anchors are typically planning or backtracking sentences. Our work offers new tools and insights for understanding multi-step reasoning in language models.

See more:
* 📄 Paper: https://arxiv.org/abs/2506.19143
* 🎮 Interface: https://www.thought-anchors.com/
* 💻 Repository for the interface: https://github.com/interp-reasoning/thought-anchors.com
* 📊 Dataset: https://huggingface.co/datasets/uzaymacar/math-rollouts
* 🎥 Video: https://www.youtube.com/watch?v=nCZN09Wjboc&t=1s 

## Get Started

Install dependencies with `uv`:

```bash
uv sync
```

Run scripts via `uv run`, for example:

```bash
uv run python generate_rollouts.py --provider vLLM
uv run python analyze_rollouts.py
```

You can download our [MATH rollout dataset](https://huggingface.co/datasets/uzaymacar/math-rollouts) or resample your own data.

Here's a quick rundown of the main scripts in this repository and what they do:

1. `generate_rollouts.py`: Main script for generating reasoning rollouts. Our [dataset](https://huggingface.co/datasets/uzaymacar/math-rollouts) was created with it.
2. `analyze_rollouts.py`: Processes the generated rollouts and adds `chunks_labeled.json` and other metadata for each reasoning trace. It calculates metrics like **forced answer importance**, **resampling importance**, and **counterfactual importance**.
3. `step_attribution.py`: Computes the sentence-to-sentence counterfactual importance score for all sentences in all reasoning traces.
4. `plots.py`: Generates figures (e.g., the ones in the paper).

Here is what other files do:
* `selected_problems.json`: A list of problems identified in the 25% - 75% accuracy range (i.e., challenging problems). It is sorted in increasing order by average length of sentences (NOTE: We use *chunks*, *steps*, and *sentences* interchangeably through the code).
* `prompts.py`: This includes auto-labeler LLM prompts we used throughout this project. `DAG_PROMPT` is the one we used to generate labels (i.e., function tags or categories, e.g., uncertainty management) for each sentence.
* `utils.py`: Includes utility and helper functions for reasoning trace analysis.
* `misc-experiments/`: This folder includes miscellaneous experiment scripts. Some of them are ongoing work.
* `whitebox-analyses/`: This folder includes the white-box experiments in the paper, including **attention pattern analysis** (e.g., *receiver heads*) and **attention suppression**.

## Citation

Please cite our work if you are using our code or dataset.

```
@misc{bogdan2025thoughtanchorsllmreasoning,
      title={Thought Anchors: Which LLM Reasoning Steps Matter?},
      author={Paul C. Bogdan and Uzay Macar and Neel Nanda and Arthur Conmy},
      year={2025},
      eprint={2506.19143},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.19143},
}
```

## Contact

For any questions, thoughts, or feedback, please reach out to [uzaymacar@gmail.com](mailto:uzaymacar@gmail.com) and [paulcbogdan@gmail.com](mailto:paulcbogdan@gmail.com).


### Miscallenous

To upload the `math_rollouts` dataset to HuggingFace, I ran:

```bash
hf upload-large-folder uzaymacar/math_rollouts --repo-type=dataset math_rollouts
```

But it turns out this is not `dataset` compatible. The `misc-scripts/push_hf_dataset.py` takes care of this instead, creating a `dataset`-compatible data repository on HuggingFace.
