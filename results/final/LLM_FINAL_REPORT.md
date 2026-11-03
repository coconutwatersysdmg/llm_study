# LLM Final Report

## 1. Dataset
- dataset_version: `bcnf_llm_v1`
- question_count: 1200
- category_counts: {'Irrelevant-Conflict': 360, 'Answer-Critical-Conflict': 480, 'No-Conflict': 360}
- SHA256 questions: `08b09471af99fca05a97288eb29246eecd5ef113b778fe1cdc4bb3298faf3b42`
- SHA256 evidence: `7c0a9650516a94b7aafb27e18c860b96e490a2af15c7945e61d09dec559a6cc6`
- prompt_version: `v1`

## 2. Models
- zhipu: `glm-4-flash-250414`
- bailian: `qwen3.7-flash-2026-07-15` (enable_thinking=False)
- bailian base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`

## 3. Exp1
- zhipu clean: acc=1.000000 unknown=0.000000 n=1200
- zhipu dirty: acc=0.813333 unknown=0.069167 n=1200
- zhipu valid_repair: acc=1.000000 unknown=0.000000 n=1200
- bailian clean: acc=1.000000 unknown=0.000000 n=1200
- bailian dirty: acc=0.735833 unknown=0.256667 n=1200
- bailian valid_repair: acc=1.000000 unknown=0.000000 n=1200

Dirty+FD conflict metrics: see `exp1_conflict_summary.csv`.

## 4. Exp2A
- zhipu rate=0.01 seed=101 candidate: acc=0.000000 n=16
- zhipu rate=0.01 seed=101 checked: acc=1.000000 n=16
- zhipu rate=0.01 seed=202 candidate: acc=0.000000 n=16
- zhipu rate=0.01 seed=202 checked: acc=1.000000 n=16
- zhipu rate=0.01 seed=303 candidate: acc=0.000000 n=19
- zhipu rate=0.01 seed=303 checked: acc=1.000000 n=19
- zhipu rate=0.05 seed=101 candidate: acc=0.000000 n=32
- zhipu rate=0.05 seed=101 checked: acc=1.000000 n=32
- zhipu rate=0.05 seed=202 candidate: acc=0.000000 n=36
- zhipu rate=0.05 seed=202 checked: acc=1.000000 n=36
- zhipu rate=0.05 seed=303 candidate: acc=0.000000 n=31
- zhipu rate=0.05 seed=303 checked: acc=1.000000 n=31
- zhipu rate=0.1 seed=101 candidate: acc=0.000000 n=70
- zhipu rate=0.1 seed=101 checked: acc=1.000000 n=70
- zhipu rate=0.1 seed=202 candidate: acc=0.000000 n=71
- zhipu rate=0.1 seed=202 checked: acc=1.000000 n=71
- zhipu rate=0.1 seed=303 candidate: acc=0.000000 n=67
- zhipu rate=0.1 seed=303 checked: acc=1.000000 n=67
- zhipu rate=0.2 seed=101 candidate: acc=0.000000 n=139
- zhipu rate=0.2 seed=101 checked: acc=1.000000 n=139
- zhipu rate=0.2 seed=202 candidate: acc=0.000000 n=145
- zhipu rate=0.2 seed=202 checked: acc=1.000000 n=145
- zhipu rate=0.2 seed=303 candidate: acc=0.000000 n=142
- zhipu rate=0.2 seed=303 checked: acc=1.000000 n=142
- bailian rate=0.01 seed=101 candidate: acc=0.000000 n=16
- bailian rate=0.01 seed=101 checked: acc=1.000000 n=16
- bailian rate=0.01 seed=202 candidate: acc=0.000000 n=16
- bailian rate=0.01 seed=202 checked: acc=1.000000 n=16
- bailian rate=0.01 seed=303 candidate: acc=0.000000 n=19
- bailian rate=0.01 seed=303 checked: acc=1.000000 n=19
- bailian rate=0.05 seed=101 candidate: acc=0.000000 n=32
- bailian rate=0.05 seed=101 checked: acc=1.000000 n=32
- bailian rate=0.05 seed=202 candidate: acc=0.000000 n=36
- bailian rate=0.05 seed=202 checked: acc=1.000000 n=36
- bailian rate=0.05 seed=303 candidate: acc=0.000000 n=31
- bailian rate=0.05 seed=303 checked: acc=1.000000 n=31
- bailian rate=0.1 seed=101 candidate: acc=0.000000 n=70
- bailian rate=0.1 seed=101 checked: acc=1.000000 n=70
- bailian rate=0.1 seed=202 candidate: acc=0.000000 n=71
- bailian rate=0.1 seed=202 checked: acc=1.000000 n=71
- bailian rate=0.1 seed=303 candidate: acc=0.000000 n=67
- bailian rate=0.1 seed=303 checked: acc=1.000000 n=67
- bailian rate=0.2 seed=101 candidate: acc=0.000000 n=139
- bailian rate=0.2 seed=101 checked: acc=1.000000 n=139
- bailian rate=0.2 seed=202 candidate: acc=0.000000 n=145
- bailian rate=0.2 seed=202 checked: acc=1.000000 n=145
- bailian rate=0.2 seed=303 candidate: acc=0.000000 n=142
- bailian rate=0.2 seed=303 checked: acc=1.000000 n=142

## 5. Exp2B
- zhipu rate=0.01 seed=101: detection=1.0 unchecked_affected_acc=0.666667 checked_exposure=0.0
- zhipu rate=0.01 seed=202: detection=1.0 unchecked_affected_acc=0.333333 checked_exposure=0.0
- zhipu rate=0.01 seed=303: detection=1.0 unchecked_affected_acc=0.000000 checked_exposure=0.0
- zhipu rate=0.05 seed=101: detection=1.0 unchecked_affected_acc=0.666667 checked_exposure=0.0
- zhipu rate=0.05 seed=202: detection=1.0 unchecked_affected_acc=0.625000 checked_exposure=0.0
- zhipu rate=0.05 seed=303: detection=1.0 unchecked_affected_acc=0.333333 checked_exposure=0.0
- zhipu rate=0.1 seed=101: detection=1.0 unchecked_affected_acc=0.660000 checked_exposure=0.0
- zhipu rate=0.1 seed=202: detection=1.0 unchecked_affected_acc=0.488372 checked_exposure=0.0
- zhipu rate=0.1 seed=303: detection=1.0 unchecked_affected_acc=0.818182 checked_exposure=0.0
- zhipu rate=0.2 seed=101: detection=1.0 unchecked_affected_acc=0.567308 checked_exposure=0.0
- zhipu rate=0.2 seed=202: detection=1.0 unchecked_affected_acc=0.500000 checked_exposure=0.0
- zhipu rate=0.2 seed=303: detection=1.0 unchecked_affected_acc=0.484211 checked_exposure=0.0
- bailian rate=0.01 seed=101: detection=1.0 unchecked_affected_acc=0.000000 checked_exposure=0.0
- bailian rate=0.01 seed=202: detection=1.0 unchecked_affected_acc=0.000000 checked_exposure=0.0
- bailian rate=0.01 seed=303: detection=1.0 unchecked_affected_acc=0.000000 checked_exposure=0.0
- bailian rate=0.05 seed=101: detection=1.0 unchecked_affected_acc=0.181818 checked_exposure=0.0
- bailian rate=0.05 seed=202: detection=1.0 unchecked_affected_acc=0.437500 checked_exposure=0.0
- bailian rate=0.05 seed=303: detection=1.0 unchecked_affected_acc=0.166667 checked_exposure=0.0
- bailian rate=0.1 seed=101: detection=1.0 unchecked_affected_acc=0.460000 checked_exposure=0.0
- bailian rate=0.1 seed=202: detection=1.0 unchecked_affected_acc=0.232558 checked_exposure=0.0
- bailian rate=0.1 seed=303: detection=1.0 unchecked_affected_acc=0.290909 checked_exposure=0.0
- bailian rate=0.2 seed=101: detection=1.0 unchecked_affected_acc=0.355769 checked_exposure=0.0
- bailian rate=0.2 seed=202: detection=1.0 unchecked_affected_acc=0.328947 checked_exposure=0.0
- bailian rate=0.2 seed=303: detection=1.0 unchecked_affected_acc=0.157895 checked_exposure=0.0

## 6. Statistical significance
See `exp1_pairwise.csv` and `exp2_overdeletion_pairwise.csv`.
- Exp1 zhipu clean_vs_dirty: diff=0.186667 CI=[0.164167,0.209167] p=7.41841e-68
- Exp1 zhipu dirty_vs_valid_repair: diff=-0.186667 CI=[-0.209167,-0.164167] p=7.41841e-68
- Exp1 zhipu clean_vs_valid_repair: diff=0.000000 CI=[0.000000,0.000000] p=1
- Exp1 bailian clean_vs_dirty: diff=0.264167 CI=[0.239167,0.289167] p=7.49068e-96
- Exp1 bailian dirty_vs_valid_repair: diff=-0.264167 CI=[-0.289167,-0.239167] p=7.49068e-96
- Exp1 bailian clean_vs_valid_repair: diff=0.000000 CI=[0.000000,0.000000] p=1

## 7. API usage
- smoke__zhipu.jsonl: req=400 ok=399 fail=1 retries=16 tokens=72202
- smoke__bailian.jsonl: req=400 ok=400 fail=0 retries=0 tokens=81389
- exp1_zhipu.jsonl: req=4800 ok=4800 fail=0 retries=139 tokens=868305
- exp1_bailian.jsonl: req=4800 ok=4800 fail=0 retries=0 tokens=976100
- exp2_overdeletion_zhipu.jsonl: req=3136 ok=3136 fail=0 retries=125 tokens=464718
- exp2_overdeletion_bailian.jsonl: req=3136 ok=3136 fail=0 retries=0 tokens=532081
- exp2_residual_zhipu.jsonl: req=2072 ok=2072 fail=0 retries=46 tokens=171871
- exp2_residual_bailian.jsonl: req=2072 ok=2072 fail=0 retries=0 tokens=198023

## 8. Cost
- zhipu smoke__zhipu.jsonl: CNY≈0.000000
- bailian smoke__bailian.jsonl: CNY≈0.016628
- zhipu exp1_zhipu.jsonl: CNY≈0.000000
- bailian exp1_bailian.jsonl: CNY≈0.199623
- zhipu exp2_overdeletion_zhipu.jsonl: CNY≈0.000000
- bailian exp2_overdeletion_bailian.jsonl: CNY≈0.108961
- zhipu exp2_residual_zhipu.jsonl: CNY≈0.000000
- bailian exp2_residual_bailian.jsonl: CNY≈0.040428

## 9. Automated research audit
- [x] 1200 questions unchanged
- [x] dataset SHA unchanged
- [x] prompt version unchanged
- [x] no ground truth leakage
- [x] both API model IDs recorded
- [x] Qwen enable_thinking=False
- [x] web/tool use disabled
- [x] Exp1 same evidence across models
- [x] Exp2A checker did not access Ground Truth
- [x] Exp2B checker never selected a true tuple
- [x] all raw responses preserved
- [x] no API key in repository
