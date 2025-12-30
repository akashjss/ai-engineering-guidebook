import torch
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import re

# 1. Load the model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct" # Example model
max_seq_length = 1024
load_in_4bit = True

PatchFastRL("GRPO", FastLanguageModel)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    fast_inference = True,
)

# 2. Define LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Create and format the dataset
SYSTEM_PROMPT = """
Respond in the following format:
<thought>
...
</thought>
<answer>
...
</answer>
"""

def format_dataset(examples):
    questions = examples["question"]
    answers = examples["answer"]
    formatted_prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        for question in questions
    ]
    return {"prompt": formatted_prompts, "answer": answers}

# Loading a subset of math dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(format_dataset, batched=True)

# 4. Define Reward Functions
def format_reward_func(completions, **kwargs):
    """Reward for matching the <thought>...<answer> format."""
    pattern = r"^<thought>\n.*?\n</thought>\n<answer>\n.*?\n</answer>$"
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for response in responses:
        res = re.match(pattern, response, re.DOTALL)
        rewards.append(1.0 if res else 0.0)
    return rewards

def correctness_reward_func(completions, answer, **kwargs):
    """Reward for matching the correct answer."""
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for response, gold_answer in zip(responses, answer):
        # Very simple extraction of answer from <answer> tag
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # GSM8K answers usually end with "#### 123"
            gold_clean = gold_answer.split("####")[-1].strip()
            rewards.append(2.0 if extracted == gold_clean else 0.0)
        else:
            rewards.append(0.0)
    return rewards

# 5. Initialize GRPO Config and Trainer
training_args = GRPOConfig(
    output_dir = "outputs/grpo",
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 8, # Number of responses per prompt (Group Size)
    max_prompt_length = 256,
    max_completion_length = 512,
    num_train_epochs = 1,
    save_steps = 100,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [format_reward_func, correctness_reward_func],
    args = training_args,
    train_dataset = dataset,
)

# 6. Start Training
# trainer.train()

print("GRPO Training script initialized. Uncomment trainer.train() to start.")

