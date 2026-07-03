"""
CoT (Chain-of-Thought) Reasoning Module

Strategy: Direct LLM reasoning instead of Neuro-Symbolic constraint extraction.

Pipeline:
  1. generate_cot_data.py  - Use DeepSeek API to generate CoT reasoning chains
  2. train_cot_model.py    - LoRA fine-tune Qwen2.5-7B on reasoning data
  3. run_cot_inference.py  - Generate answers for test set using fine-tuned model

Background: The Neuro-Symbolic approach (LLM extracts constraints → symbolic solver
→ answer verifier) failed because DeepSeek-V3/R1 could not extract structured
constraints with sufficient precision (only 3.6% correct). The CoT approach
leverages the LLM's natural strength in step-by-step reasoning.
"""
