import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import unicodedata
import re
import arabic_reshaper
from bidi.algorithm import get_display
import os

# List of models to evaluate
model_names = [
    'bigscience/bloom-560m',
    'akhooli/gpt2-small-arabic',
    'aubmindlab/aragpt2-large'
]

def normalize_text(text):
    return unicodedata.normalize("NFC", text)

def prepare_rtl_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def compute_surprisal(model, tokenizer, prefix, target_word):
    """
    Computes surprisal of the entire target_word given a prefix.
    Surprisal = -log2(P(target | prefix))
    """
    prefix = normalize_text(prefix)
    target_word = normalize_text(target_word)

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)

    input_ids = prefix_ids + target_ids

    with torch.no_grad():
        inputs = torch.tensor([input_ids])
        outputs = model(inputs)
        logits = outputs.logits

    surprisal = 0.0
    for i in range(len(prefix_ids), len(input_ids)):
        token_logits = logits[0, i - 1]
        token_id = input_ids[i]
        prob = torch.softmax(token_logits, dim=-1)[token_id].item()
        surprisal += -math.log2(prob)

    return surprisal

# Load stimuli
file_name = "full_stimuli.txt"
try:
    with open(file_name, "r", encoding="utf-8") as file:
        stimuli = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found.")
    exit()

if len(stimuli) % 4 != 0:
    print("Error: Stimuli file must contain a multiple of 4 lines.")
    exit()

conditions = [
    "SS: singular attractor, singular verb",
    "PS: plural attractor, singular verb",
    "SP: singular attractor, plural verb",
    "PP: plural attractor, plural verb"
]

# Process each model
for model_name in model_names:
    print(f"\n===== Running: {model_name} =====")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    output_lines = []
    num_stimuli = len(stimuli) // 4
    print(f"Total stimuli detected: {num_stimuli}")

    for i in range(num_stimuli):
        stimulus_num = i + 1
        output_lines.append(f"Stimulus {stimulus_num}:\n")
        print(f"  Processing Stimulus {stimulus_num}...")

        for j in range(4):
            sentence = normalize_text(stimuli[i * 4 + j])
            rtl_display = prepare_rtl_text(sentence)
            print(f"    RTL Display: {rtl_display}")

            try:
                prefix, rest = sentence.split("،")
                target_word = rest.strip().split()[0]
            except Exception:
                print(f" Failed to extract prefix/verb from: {sentence}")
                output_lines.append(f"  {conditions[j]}: Skipped due to parsing error.\n")
                continue

            try:
                surprisal = compute_surprisal(model, tokenizer, prefix + "،", target_word)
                output_lines.append(f"  {conditions[j]}: Surprisal Score = {surprisal:.4f}\n")
            except Exception:
                print(f" Failed to compute surprisal for: {target_word}")
                output_lines.append(f"  {conditions[j]}: Surprisal calculation failed for '{target_word}'.\n")

        output_lines.append("\n")

    # Prepare result filename
    model_short_name = model_name.split("/")[-1]
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"Surprise_scores_prefix_{model_short_name}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f" Results saved to: {output_path}")
