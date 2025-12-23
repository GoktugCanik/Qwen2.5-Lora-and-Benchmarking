LoRA Deep vs Diverse Fine-Tuning on Qwen2.5-Coder

This repository presents a comparative study of LoRA-based fine-tuning on the Qwen/Qwen2.5-Coder-1.5B-Instruct model using two distinct instruction-style datasets: Deep Instruction and Diverse Instruction.

The objective is to analyze how dataset characteristics affect code generation performance and to identify the best-performing checkpoint using the LiveCodeBench framework.
Project Overview

    Base Model: Qwen/Qwen2.5-Coder-1.5B-Instruct

    Fine-tuning Method: LoRA (Low-Rank Adaptation) via Unsloth

    Training Variants:

        deep_instruction: Focused on step-by-step logical reasoning.

        diverse_instruction: Focused on variety in problem formulations.

    Evaluation Benchmark: LiveCodeBench

    Evaluation Subset: AtCoder – Easy (41 problems, date range: 2408-2502)

    Primary Metric: Pass@1

Both models were fine-tuned using identical hyperparameters to ensure an objective comparison of the data influence itself.
Methodology
Training Data

    Deep Dataset: Naholav/CodeGen-Deep-5K

    Diverse Dataset: Naholav/CodeGen-Diverse-5K

    Split: 80% Train, 10% Validation, 10% Test.

Hyperparameters

    Learning Rate: 2e-4

    Batch Size: 4 (with Gradient Accumulation 4, Total=16)

    Epochs: 3

    LoRA Config: r=16,α=32, targeting all linear layers.

Training Analysis (Loss Curves)

We monitored both Training Loss and Validation Loss every 20 steps to ensure the model was learning effectively without extreme overfitting.

    Deep Instruction: Showed a very smooth and consistent decline in loss (ending at ~0.26), suggesting a high-quality, easily learnable data structure.

    Diverse Instruction: Exhibited more fluctuations in training loss and a higher plateau in validation loss (~0.88), indicating that the diversity of the data makes the optimization landscape more complex.

Benchmarking Results

The models were evaluated on 41 AtCoder Easy problems. The Pass@1 metric was used to determine the success rate of the first generated solution.
Performance Summary

Model,Best Checkpoint,Pass@1 (%),Problems Solved

Deep_Instruction,Step-100 / 200 / 600 / 700,26.83,11 / 41

Diverse_Instruction,Step-300,24.39,10 / 41


# Comparison Report: Deep (Step 600) vs Diverse (Step 300)

## Common Solved Questions

- abc302_a
- abc315_a
- abc320_b
- abc322_a

## Questions Solved ONLY by Deep (Step 600)

- abc305_a
- abc307_a
- abc319_b
- abc322_b
- abc323_a
- abc323_b
- abc324_a

## Questions Solved ONLY by Diverse (Step 300)

- abc306_b
- abc307_b
- abc308_a
- abc311_a
- abc312_a
- abc324_b
