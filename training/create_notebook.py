import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Qwen 2.5 3B - Invoice Extraction Fine-Tuning\n",
    "\n",
    "This notebook will fine-tune the **Qwen 2.5 3B Instruct** model on our synthetic invoice dataset using Unsloth. Unsloth makes training 2x faster and uses 70% less memory.\n",
    "\n",
    "### Instructions:\n",
    "1. In the left sidebar, click the **Folder** icon to open the file browser.\n",
    "2. Drag and drop your `train_formatted.jsonl` and `val_formatted.jsonl` files into the files area.\n",
    "3. Go to the top menu: **Runtime -> Change runtime type** and ensure **T4 GPU** is selected.\n",
    "4. Go to **Runtime -> Run all** (or press Ctrl+F9).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture\n",
    "!pip install unsloth\n",
    "!pip install --force-reinstall --no-cache-dir --no-deps triton\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from unsloth import FastLanguageModel\n",
    "import torch\n",
    "\n",
    "max_seq_length = 4096 # Supports up to 32k\n",
    "dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+\n",
    "load_in_4bit = True # Use 4bit quantization to reduce memory usage\n",
    "\n",
    "# Load Model\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name = \"unsloth/Qwen2.5-3B-Instruct\",\n",
    "    max_seq_length = max_seq_length,\n",
    "    dtype = dtype,\n",
    "    load_in_4bit = load_in_4bit,\n",
    ")\n",
    "\n",
    "# Configure LoRA Adapters\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model,\n",
    "    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128\n",
    "    target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
    "                      \"gate_proj\", \"up_proj\", \"down_proj\",],\n",
    "    lora_alpha = 16,\n",
    "    lora_dropout = 0, # Supports any, but = 0 is optimized\n",
    "    bias = \"none\",    # Supports any, but = \"none\" is optimized\n",
    "    use_gradient_checkpointing = \"unsloth\", # True or \"unsloth\" for very long context\n",
    "    random_state = 3407,\n",
    "    use_rslora = False,\n",
    "    loftq_config = None,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "# Load our formatted JSONL datasets\n",
    "train_dataset = load_dataset(\"json\", data_files=\"train_formatted.jsonl\", split=\"train\")\n",
    "val_dataset = load_dataset(\"json\", data_files=\"val_formatted.jsonl\", split=\"train\")\n",
    "\n",
    "print(f\"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import SFTTrainer\n",
    "from transformers import TrainingArguments\n",
    "from unsloth import is_bfloat16_supported\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model = model,\n",
    "    tokenizer = tokenizer,\n",
    "    train_dataset = train_dataset,\n",
    "    eval_dataset = val_dataset,\n",
    "    dataset_text_field = \"text\",\n",
    "    max_seq_length = max_seq_length,\n",
    "    dataset_num_proc = 2,\n",
    "    packing = False, # Can make training 5x faster for short sequences.\n",
    "    args = TrainingArguments(\n",
    "        per_device_train_batch_size = 2,\n",
    "        gradient_accumulation_steps = 4,\n",
    "        warmup_steps = 10,\n",
    "        num_train_epochs = 3, # Standard for fine-tuning\n",
    "        learning_rate = 2e-4,\n",
    "        fp16 = not is_bfloat16_supported(),\n",
    "        bf16 = is_bfloat16_supported(),\n",
    "        logging_steps = 5,\n",
    "        eval_strategy = \"steps\",\n",
    "        eval_steps = 25,\n",
    "        optim = \"adamw_8bit\",\n",
    "        weight_decay = 0.01,\n",
    "        lr_scheduler_type = \"linear\",\n",
    "        seed = 3407,\n",
    "        output_dir = \"outputs\",\n",
    "        report_to = \"none\", # Change to wandb if you want to track metrics\n",
    "    ),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start training!\n",
    "trainer_stats = trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the resulting LoRA adapters\n",
    "model.save_pretrained(\"lora_model\") # Local saving\n",
    "tokenizer.save_pretrained(\"lora_model\")\n",
    "print(\"Training Complete! Your LoRA model is saved in the 'lora_model' folder.\")\n",
    "print(\"You can right click the 'lora_model' folder in the left sidebar and click 'Download'.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(os.path.join(os.path.dirname(__file__), "Unsloth_Finetune_Qwen2_5_3B.ipynb"), "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
