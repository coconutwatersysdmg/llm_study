# dev — development-era scripts

This directory is not part of the paper's experimental pipeline, and nothing here is needed to reproduce the reported results.

| File | Description |
|---|---|
| `retry_failed_glm.py` | Re-runs requests that failed transiently against the Zhipu API, keyed by `request_id`; the formal pipeline now has retries and resume built in |
| `build_paper_tables_from_raw.py` | Builds Tables 1 and 2 directly from `results/raw/*.jsonl`; the formal path is now `summarize_results.py` followed by `experiments/scripts/build_llm_paper_tables.py`, with identical output |
