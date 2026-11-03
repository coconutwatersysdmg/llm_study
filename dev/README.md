# dev — 开发期脚本

本目录不属于论文实验流程，复现论文结果不需要这里的任何文件。

| 文件 | 说明 |
|---|---|
| `retry_failed_glm.py` | 智谱接口偶发失败时按 `request_id` 补跑，正式流程已内置重试与断点续跑 |
| `build_paper_tables_from_raw.py` | 直接从 `results/raw/*.jsonl` 建表1、表2；正式路径改为先 `summarize_results.py` 再 `experiments/scripts/build_llm_paper_tables.py`，两者输出一致 |
