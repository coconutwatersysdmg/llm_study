# LLM Structured Question Answering on Dirty BCNF Databases

Code for Section 4.5 of the paper *Leveraging BCNF Schema Structure for Efficient Subset-Repair Checking*. It produces Fig. 6 (QA accuracy under Clean / Dirty / Valid Repair conditions) together with Table 1 (conflict identification with explicit FD hints) and Table 2 (effect of two families of defective candidate repairs: over-deletion and residual conflicts). The whole pipeline runs with `python scripts/run_formal_pipeline.py`. To recompute every summary from the data in the repository without calling any API, run `scripts/validate_frozen_dataset.py` and then `scripts/summarize_results.py`. Requires Python 3.12 and `pip install -r requirements.txt`, and the repository must sit next to the algorithm repository `experiments` under the same parent directory.

## 1. Mapping to the paper

| Paper item | Content | Output |
|---|---|---|
| §4.5, Fig. 6 | Overall and per-category accuracy, two models × three data conditions | `results/final/exp1_summary.csv` → `../experiments/paper_figures/tif/fig4_llm_data_quality.tif` |
| §4.5, Table 1 | Conflict-identification precision/recall/F1 under Dirty + explicit FD hints | `results/final/exp1_conflict_summary.csv` → `../experiments/paper_tables/table_fd_prompt.csv` |
| §4.5, Table 2 | Effect of over-deletion (Exp2A) and residual conflicts (Exp2B) on affected-question accuracy | `results/final/exp2_*_summary.csv` → `../experiments/paper_tables/table_llm_checker.csv` |

The plotting script for Fig. 6 and the table builders for Tables 1 and 2 live in the `experiments` repository and read `results/final/` here, so the two repositories must be kept side by side.

## 2. Reproduction

```bash
# One configuration, everything else automatic: dataset hash check -> unit tests -> Exp2 artifacts
# -> connectivity check -> smoke test -> formal runs -> summarization
cp .env.example .env      # fill in ZAI_API_KEY / DASHSCOPE_API_KEY
python scripts/run_formal_pipeline.py
```

To recompute every summary without calling an API, using the frozen data and per-request records shipped in the repository:

```bash
python scripts/validate_frozen_dataset.py                                           # verify dataset hashes first
python scripts/summarize_results.py                                                 # rebuild results/final/
cd ../experiments
python scripts/build_llm_paper_tables.py                                            # rebuild Tables 1 and 2
python paper_figures/source/plot_llm_figures.py                                     # regenerate Fig. 6
```

Before spending money, `--dry-run` reports the number of requests and the estimated cost:

```bash
python scripts/run_exp1.py --dry-run
python scripts/run_exp2_overdeletion.py --dry-run
python scripts/run_exp2_residual.py --dry-run
```

## 3. Layout

```
checker/     BCNFRepairChecker adapter (calls the algorithms in experiments, read-only)
clients/     OpenAI-compatible clients for Zhipu and Bailian
generation/  Database / dirty-data / candidate-repair / QA data generation
evaluation/  Answer normalization, paired tests, metric computation
prompts/     Prompt templates (bump PROMPT_VERSION on any change)
scripts/     Dataset construction, connectivity checks, cost estimation, formal runs, summarization
results/     final (summary CSVs), logs (run manifests), raw (per-request records, committed)
dev/         Development-era scripts, not needed for the paper experiments
tests/       13 unit tests
```

## 4. Environment and directory layout

Python 3.12:

```bash
pip install -r requirements.txt
python -m pytest tests -q        # 13 tests
```

`utils/bootstrap.py` imports the frozen `BCNFRepairChecker` from the sibling `experiments/` directory through `sys.path`, so the layout must be:

```
some_dir/
├── experiments/     # algorithm repository
└── llm_study/       # this repository
```

Copying `llm_study` out on its own makes it exit with an import error. This is intentional: there is exactly one implementation of the checker.

## 5. Experimental protocol

| Item | Setting |
|---|---|
| Data size | STUDENT 1000, COURSE 100, ENROLLMENT 15000 tuples |
| Questions | 1200: 360 conflict-free, 360 irrelevant-conflict, 480 answer-critical-conflict |
| Model snapshots | `glm-4-flash-250414`; `qwen3.7-flash-2026-07-15` (`enable_thinking=False`) |
| Sampling | `temperature=0.01`, `max_tokens=32`, single turn, no tools, no web access |
| Generation seeds | `GENERATION_SEED=42`, `EVIDENCE_ORDER_SEED=42` |
| Exp2 perturbations | Error rates 1% / 5% / 10% / 20%, seeds 101 / 202 / 303 |
| Prompt version | `v1` (`prompts/templates.py`) |

Evidence is attached in a controlled way: the records matching each question's metadata are taken directly from the database and passed to the model as input. There is no retriever, no embedding, and no vector store, which separates "the model answered wrongly" from "retrieval failed".

## 6. Frozen data and verifiability

- The frozen dataset used in the paper is committed: `data/qa/questions.jsonl` (1200 questions), `data/qa/evidence.jsonl` (the corresponding evidence), and `data/qa/dataset_manifest.json`.
- `dataset_manifest.json` records `dataset_version`, both generation seeds, table sizes, per-category question counts, the schema definition, and `SHA256_questions` / `SHA256_evidence`. `scripts/validate_frozen_dataset.py` compares the hashes and exits non-zero on any mismatch.
- Per-request records `results/raw/*.jsonl` (about 20 MB) are committed. They contain the condition, the answer, correctness, and the prompt hash of every request, which is enough to recompute all summary tables without calling an API.
- `results/logs/*_manifest.json` records the model snapshot, temperature, prompt version, dataset version, and the list of question IDs.
- `scripts/build_dataset.py` rebuilds the database and QA data from scratch without calling an API, for new scales or new experiments. It overwrites everything under `data/` including the manifest; reproducing the paper does not require it, so use the committed frozen version.

## 7. Cost

`scripts/estimate_cost.py` estimates token cost before a run; `results/final/api_cost_summary.csv` records the actual usage of the reported experiments. Qwen unit prices can be overridden with `QWEN_INPUT_CNY_PER_M` / `QWEN_OUTPUT_CNY_PER_M`, and `MAX_QWEN_BUDGET_CNY` (20 CNY by default) aborts a run that would exceed the budget.

## 8. Citation

If you use this code, please cite:

> Leveraging BCNF Schema Structure for Efficient Subset-Repair Checking. IEEE International Conference on Data Engineering (ICDE), 2027.

## 9. License

MIT, see [`LICENSE`](LICENSE).
