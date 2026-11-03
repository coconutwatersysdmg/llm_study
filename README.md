# BCNF 条件下 LLM 结构化问答实验

本仓库是论文《BCNF条件下的数据库子集修复检查问题研究》第 4.5 节的问答实验代码，产出图6（Clean / Dirty / Valid Repair 三种条件下的问答准确率）以及表1（基于函数依赖提示的冲突识别）和表2（过度删除与残余冲突两类候选修复缺陷的影响）；全流程用 `python scripts/run_formal_pipeline.py` 一次跑完，不调用 API、仅用入库数据重算全部汇总时依次执行 `scripts/validate_frozen_dataset.py` 与 `scripts/summarize_results.py`；环境为 Python 3.12 + `pip install -r requirements.txt`，且必须与算法仓库 `experiments` 放在同一父目录下。

## 1. 对应论文位置

| 论文位置 | 内容 | 产出文件 |
|---|---|---|
| 4.5 节、图6 | 两种模型 × 三种数据条件的总体与分类准确率 | `results/final/exp1_summary.csv` → `experiments/paper_figures/tif/fig4_llm_data_quality.tif` |
| 4.5 节、表1 | Dirty + 显式函数依赖提示下的冲突识别 P/R/F1 | `results/final/exp1_conflict_summary.csv` → `experiments/paper_tables/table_fd_prompt.csv` |
| 4.5 节、表2 | 过度删除（Exp2A）与残余冲突（Exp2B）对受影响问题准确率的影响 | `results/final/exp2_*_summary.csv` → `experiments/paper_tables/table_llm_checker.csv` |

图6 的绘图脚本与表1、表2 的建表脚本都在 `experiments` 仓库中，直接读取本仓库的 `results/final/`，因此两个仓库需要并列存放。

## 2. 复现

```bash
# 一处配置，其余全自动：校验数据哈希 → 单元测试 → 构建 Exp2 工件 → 连通性检查 → 冒烟 → 正式实验 → 汇总
cp .env.example .env      # 填入 ZAI_API_KEY / DASHSCOPE_API_KEY
python scripts/run_formal_pipeline.py
```

不调用 API，直接用已入库的冻结数据与逐请求记录重算全部汇总表：

```bash
python scripts/validate_frozen_dataset.py                                           # 先确认数据哈希
python scripts/summarize_results.py                                                 # 重算 results/final/
cd ../experiments
python scripts/build_llm_paper_tables.py                                            # 重建表1、表2
python paper_figures/source/plot_llm_figures.py                                     # 重出图6
```

运行前可用 `--dry-run` 核对请求数与成本：

```bash
python scripts/run_exp1.py --dry-run
python scripts/run_exp2_overdeletion.py --dry-run
python scripts/run_exp2_residual.py --dry-run
```

## 3. 目录结构

```
checker/     BCNFRepairChecker 适配层（调用 experiments 中的算法，只读不改）
clients/     智谱、百炼两个 OpenAI 兼容客户端
generation/  数据库/脏数据/候选修复/问答数据生成
evaluation/  答案归一化、配对检验等指标计算
prompts/     Prompt 模板（PROMPT_VERSION 变更须升版本）
scripts/     数据构建、连通性检查、成本估算、正式实验、汇总
results/     final（汇总 CSV）、logs（运行清单）、raw（逐请求记录，已入库）
dev/         开发期脚本，非论文实验所需
tests/       13 项单元测试
```

## 4. 环境与目录位置

Python 3.12：

```bash
pip install -r requirements.txt
python -m pytest tests -q        # 13 项测试
```

`utils/bootstrap.py` 通过 `sys.path` 导入同级 `experiments/` 中冻结的 `BCNFRepairChecker`，因此目录结构必须是：

```
some_dir/
├── experiments/     # 算法仓库
└── llm_study/       # 本仓库
```

`llm_study` 单独拷出会因找不到 `experiments/` 直接报错退出，这是有意为之：检查器只有一份实现。

## 5. 实验协议

| 项 | 设置 |
|---|---|
| 数据规模 | STUDENT 1000、COURSE 100、ENROLLMENT 15000 条 |
| 问题数量 | 1200：无冲突 360、无关冲突 360、答案关键冲突 480 |
| 模型快照 | `glm-4-flash-250414`；`qwen3.7-flash-2026-07-15`（`enable_thinking=False`） |
| 采样参数 | `temperature=0.01`、`max_tokens=32`、单轮、无 tools、无联网 |
| 生成种子 | `GENERATION_SEED=42`、`EVIDENCE_ORDER_SEED=42` |
| Exp2 扰动 | 错误率 1% / 5% / 10% / 20%，种子 101 / 202 / 303 |
| Prompt 版本 | `v1`（`prompts/templates.py`） |

证据采用受控附着（controlled evidence）：按问题元数据直接取对应数据库记录作为模型输入，不设检索器、嵌入与向量库，目的是把“回答错误”与“检索失败”区分开。

## 6. 数据冻结与可核对性

- 论文使用的冻结数据已入库：`data/qa/questions.jsonl`（1200 条）、`data/qa/evidence.jsonl`（对应证据），以及 `data/qa/dataset_manifest.json`。
- `dataset_manifest.json` 记录 `dataset_version`、两个生成种子、各表规模、三类问题计数、模式定义，以及 `SHA256_questions` / `SHA256_evidence`；`scripts/validate_frozen_dataset.py` 比对哈希，不一致即以非零码退出。
- 逐请求记录 `results/raw/*.jsonl`（约 20 MB）入库，含每次请求的条件、答案、正确性与 Prompt 哈希，可在不调用 API 的情况下重算全部汇总表。
- `results/logs/*_manifest.json` 记录模型快照、温度、Prompt 版本、数据集版本与题号清单。
- `scripts/build_dataset.py` 用于从零重建数据库与问答数据（不调用 API），供更换规模或做新实验时使用；它会覆盖 `data/` 下的数据与清单，复现论文结果不需要运行它，请以入库的冻结版本为准。

## 7. 成本

`scripts/estimate_cost.py` 在调用前估算 token 费用；`results/final/api_cost_summary.csv` 是本次实验的实际用量。Qwen 单价可用 `QWEN_INPUT_CNY_PER_M` / `QWEN_OUTPUT_CNY_PER_M` 覆盖，上限由 `MAX_QWEN_BUDGET_CNY`（默认 20 元）控制，超限即中止。

## 8. 引用

若使用本仓库代码，请引用论文《BCNF条件下的数据库子集修复检查问题研究》，计算机研究与发展。

## 9. 许可

MIT 许可，见 [`LICENSE`](LICENSE)。
