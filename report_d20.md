# nanochat training report

Generated: 2026-01-18 11:22:11

## Environment

### Git Information
- Branch: master
- Commit: f403858 (dirty)
- Message: Revert "Merge pull request #1 from codeandcodes/feat/fp8-blackwell"

### Hardware
- Platform: Linux
- CPUs: 16 cores (24 logical)
- Memory: 31.1 GB
- GPUs: 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- GPU Memory: 95.0 GB total
- CUDA Version: 12.8
- Hourly Rate: $2.00/hour

### Software
- Python: 3.10.12
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 435,645
- Lines: 10,713
- Files: 55
- Tokens (approx): 108,911
- Dependencies (uv.lock lines): 2,641

Run started: 2026-01-18 11:22:11

---

## Base model evaluation
timestamp: 2026-01-18 11:46:47

- Model: base_model (step 21400)
- CORE metric: 0.1982
- hellaswag_zeroshot: 0.2638
- jeopardy: 0.0680
- bigbench_qa_wikidata: 0.5169
- arc_easy: 0.5331
- arc_challenge: 0.1047
- copa: 0.3200
- commonsense_qa: 0.1104
- piqa: 0.3743
- openbook_qa: 0.1360
- lambada_openai: 0.3905
- hellaswag: 0.2646
- winograd: 0.2308
- winogrande: 0.0450
- bigbench_dyck_languages: 0.1330
- agi_eval_lsat_ar: 0.0815
- bigbench_cs_algorithms: 0.3871
- bigbench_operators: 0.1762
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.2413
- coqa: 0.1982
- boolq: -0.3914
- bigbench_language_identification: 0.1774


## Chat evaluation mid
timestamp: 2026-01-18 12:11:24

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d20
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4495
- ARC-Challenge: 0.3328
- MMLU: 0.3346
- GSM8K: 0.0371
- HumanEval: 0.0549
- SpellingBee: 0.9922
- ChatCORE metric: 0.2622


## Chat evaluation sft
timestamp: 2026-01-18 12:32:20

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d20
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.4853
- ARC-Challenge: 0.3353
- MMLU: 0.3423
- GSM8K: 0.0553
- HumanEval: 0.0366
- SpellingBee: 0.9844
- ChatCORE metric: 0.2711


## Summary

- Characters: 435,645
- Lines: 10,713
- Files: 55
- Tokens (approx): 108,911
- Dependencies (uv.lock lines): 2,641

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.1982   | -        | -        | -        |
| ARC-Challenge   | -        | 0.3328   | 0.3353   | -        |
| ARC-Easy        | -        | 0.4495   | 0.4853   | -        |
| GSM8K           | -        | 0.0371   | 0.0553   | -        |
| HumanEval       | -        | 0.0549   | 0.0366   | -        |
| MMLU            | -        | 0.3346   | 0.3423   | -        |
| ChatCORE        | -        | 0.2622   | 0.2711   | -        |

Total wall clock time: 1h10m
