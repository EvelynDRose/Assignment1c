# Imports
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig, AutoTokenizer
import numpy as np
import evaluate
import torch
from peft import LoraConfig
from trl import SFTTrainer
from tabulate import tabulate
from statistics import mean 

# dataset
dataset = load_dataset('json', data_files='alpaca_data.json', split='train')

print("DATASET: ", dataset)

# models and tokeizers
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Models
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token="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", quantization_config=bnb_config, device_map="auto")
tokenizer = tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_eos_token=True, add_bos_token=True)

# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", quantization_config=bnb_config, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", add_eos_token=True)

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# model = AutoModelForCausalLM.from_pretrained("results_llama\checkpoint-8500", quantization_config=bnb_config, device_map="auto")
# tokenizer = tokenizer = AutoTokenizer.from_pretrained("results_llama\checkpoint-8500", add_eos_token=True, add_bos_token=True)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# mapping
def formatting_func(example):
    text = f"### Instruction: {example['instruction']}\nInput: {example['input']}\n### Answer: {example['output']}"
    return text
def generate_and_tokenize_prompt(prompt):
    result = tokenizer(formatting_func(prompt), truncation=True, max_length=512, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

# Evaluate
metric_b = evaluate.load("bleu")
metric_r = evaluate.load('rouge')
metric_be = evaluate.load("bertscore")
b, r, be = [],[],[]
def compute_metrics(eval_pred):
    preds, decoded_preds = [], []

    for i in range(len(eval_pred)):
        print(i)
        text = f"Given an instruction and input, return the correct answer. \nInstruction: {eval_pred['instruction'][i]}\nInput: {eval_pred['input'][i]}\nAnswer:"
        text2 = f"Given an instruction and input, return the correct answer. \nInstruction: {eval_pred['instruction'][i]}\nInput: {eval_pred['input'][i]}\nAnswer:{eval_pred['output'][i]}"
        preds.append(text2)
        model_input = tokenizer(text, return_tensors="pt").to("cuda")
        decoded_preds.append(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

    b.append(metric_b.compute(predictions=preds, references=decoded_preds))
    r.append(metric_r.compute(predictions=preds, references=decoded_preds))
    be.append(metric_be.compute(predictions=preds, references=decoded_preds, lang="en"))

# Evaluations set for metrics
eval_set = dataset.shuffle(seed=42).select(range(20))
print(eval_set['instruction'])

# tokenize dataset
tokenized_dataset = dataset.map(generate_and_tokenize_prompt)

print(tokenized_dataset)

# params
training_params = TrainingArguments(
    output_dir="./results_phi",
    num_train_epochs=1,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=500,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=peft_params,
    dataset_text_field="instruction",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# task 1
trainer.train()

# task 2
# compute_metrics(eval_set)

# # table
# head =  [      "BLEU",      "Rogue-L",              "BERTScore"]
# row =   [b[0]['bleu'], r[0]['rougeL'], mean(be[0]["precision"])]

# table = [row]

# print(tabulate(table, headers=head, tablefmt="grid"))


# Task 3
k = [1,3,5,10]
beam = [1,3,5,10]
t = [.3,.5,.8,1]
def task3(eval_pred):
    for j in range(len(k)):
        preds, decoded_preds_k, decoded_preds_beam, decoded_preds_t = [], [], [], []
        for i in range(len(eval_pred)):
            print(i)
            text = f"Given an instruction and input, return the correct answer. \nInstruction: {eval_pred['instruction'][i]}\nInput: {eval_pred['input'][i]}\nAnswer:"
            text2 = f"Given an instruction and input, return the correct answer. \nInstruction: {eval_pred['instruction'][i]}\nInput: {eval_pred['input'][i]}\nAnswer:{eval_pred['output'][i]}"
            preds.append(text2)
            model_input = tokenizer(text, return_tensors="pt").to("cuda")
            decoded_preds_k.append(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True, top_k = k[j]))
            decoded_preds_beam.append(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True, beam_size = beam[j]))
            decoded_preds_t.append(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True, temperature = t[j]))

        b.append([metric_b.compute(predictions=preds, references=decoded_preds_k), metric_b.compute(predictions=preds, references=decoded_preds_beam), metric_b.compute(predictions=preds, references=decoded_preds_t)])
        r.append([metric_r.compute(predictions=preds, references=decoded_preds_k), metric_r.compute(predictions=preds, references=decoded_preds_beam), metric_r.compute(predictions=preds, references=decoded_preds_t)])
        be.append([metric_be.compute(predictions=preds, references=decoded_preds_k, lang="en"), metric_be.compute(predictions=preds, references=decoded_preds_beam, lang="en"), metric_be.compute(predictions=preds, references=decoded_preds_t, lang="en")])

b, r, be = [],[],[]
task3(eval_set)

print(b)
print(r)
print(be)

for i in range(len(b)):
    # table
    head =  ["",      "BLEU",      "Rogue-L",              "BERTScore"]
    row_k =   ["k="+str(k[i]), b[i][0]['bleu'], r[i][0]['rougeL'], mean(be[i][0]["precision"])]
    row_b =   ["b="+str(b[i]), b[i][1]['bleu'], r[i][1]['rougeL'], mean(be[i][1]["precision"])]
    row_t =   ["t="+str(t[i]), b[i][2]['bleu'], r[i][2]['rougeL'], mean(be[i][2]["precision"])]

    table = [row_k,row_b,row_t]

    print(tabulate(table, headers=head, tablefmt="grid"))


# References
# https://huggingface.co/docs/transformers/en/training
# https://www.datacamp.com/tutorial/fine-tuning-llama-2
# https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config
# https://github.com/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb