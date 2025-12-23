import os
import torch
import gc
from google.colab import drive
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Drive ve Model Hazırlığı
drive.mount('/content/drive')

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True

def get_dataset_splits(dataset_id):
    ds = load_dataset(dataset_id, split="train")
    train_testval = ds.train_test_split(test_size=0.2, seed=42)
    test_val = train_testval["test"].train_test_split(test_size=0.5, seed=42)
    return {
        "train": train_testval["train"],
        "val": test_val["train"],
        "test": test_val["test"]
    }

def formatting_func(examples):
    instructions = []
    for i, o in zip(examples["input"], examples["solution"]):
        text = f"<|im_start|>system\nYou are an expert Python programmer...<|im_end|>\n" \
               f"<|im_start|>user\n{i}<|im_end|>\n" \
               f"<|im_start|>assistant\n{o}<|im_end|>"
        instructions.append(text)
    return {"text": instructions}

def run_fine_tuning(dataset_id, output_dir):
    print(f"🚀 Başlatılıyor: {dataset_id}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
    )

    splits = get_dataset_splits(dataset_id)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = splits["train"].map(formatting_func, batched=True),
        eval_dataset = splits["val"].map(formatting_func, batched=True),
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 20,
            eval_strategy = "steps",
            eval_steps = 20,
            save_strategy = "steps",
            save_steps = 100,
            output_dir = output_dir,
            optim = "adamw_8bit",
        ),
    )

    trainer.train()
    # Bellek temizliği
    del model; del tokenizer; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    # Deep Eğitimini Başlat
    run_fine_tuning("Naholav/CodeGen-Deep-5K", "/content/drive/MyDrive/nlp_project/deep_instruction")
    
    # Diverse Eğitimini Başlat
    run_fine_tuning("Naholav/CodeGen-Diverse-5K", "/content/drive/MyDrive/nlp_project/diverse_instruction")