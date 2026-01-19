# nanochat training report

Generated: 2026-01-18 09:36:24

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
- Characters: 436,698
- Lines: 10,731
- Files: 55
- Tokens (approx): 109,174
- Dependencies (uv.lock lines): 2,641

Run started: 2026-01-18 09:36:24

---

## Base model evaluation
timestamp: 2026-01-18 10:14:04

- Model: base_model (step 40640)
- CORE metric: 0.2708
- hellaswag_zeroshot: 0.3672
- jeopardy: 0.2277
- bigbench_qa_wikidata: 0.5803
- arc_easy: 0.5920
- arc_challenge: 0.2184
- copa: 0.4200
- commonsense_qa: 0.1667
- piqa: 0.4222
- openbook_qa: 0.1253
- lambada_openai: 0.4500
- hellaswag: 0.3705
- winograd: 0.3626
- winogrande: 0.1097
- bigbench_dyck_languages: 0.0980
- agi_eval_lsat_ar: 0.0761
- bigbench_cs_algorithms: 0.3848
- bigbench_operators: 0.2048
- bigbench_repeat_copy_logic: 0.0625
- squad: 0.3478
- coqa: 0.2604
- boolq: -0.0679
- bigbench_language_identification: 0.1789


## Chat evaluation mid
timestamp: 2026-01-18 10:40:45

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d26
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.5614
- ARC-Challenge: 0.3908
- MMLU: 0.3711
- GSM8K: 0.0705
- HumanEval: 0.1098
- SpellingBee: 0.9883
- ChatCORE metric: 0.3222


## Chat evaluation sft
timestamp: 2026-01-18 11:22:09

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: d26
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.5875
- ARC-Challenge: 0.4078
- MMLU: 0.3740
- GSM8K: 0.0804
- HumanEval: 0.1037
- SpellingBee: 0.9883
- ChatCORE metric: 0.3330


## Summary

- Characters: 436,698
- Lines: 10,731
- Files: 55
- Tokens (approx): 109,174
- Dependencies (uv.lock lines): 2,641

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2708   | -        | -        | -        |
| ARC-Challenge   | -        | 0.3908   | 0.4078   | -        |
| ARC-Easy        | -        | 0.5614   | 0.5875   | -        |
| GSM8K           | -        | 0.0705   | 0.0804   | -        |
| HumanEval       | -        | 0.1098   | 0.1037   | -        |
| MMLU            | -        | 0.3711   | 0.3740   | -        |
| ChatCORE        | -        | 0.3222   | 0.3330   | -        |

Total wall clock time: 1h45m
