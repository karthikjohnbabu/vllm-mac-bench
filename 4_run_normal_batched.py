import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_folder = "models/SmolLM2-360M-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

num_requests = 100   # same as the vLLM script
max_tokens = 100     # same limit as the other scripts


def sync():
    if device == "mps":
        torch.mps.synchronize()


# ---------- load model ----------
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_folder)
tokenizer.padding_side = "left"   # batched generation needs padding on the left
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_folder, dtype=torch.bfloat16).to(device)
model.eval()
sync()
load_time = time.time() - start

# ---------- the same 100 questions the vLLM script uses ----------
texts = []
for i in range(num_requests):
    messages = [{"role": "user", "content": "Give me one fun fact number " + str(i + 1)}]
    texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
input_len = inputs["input_ids"].shape[1]

# ---------- warm-up (not timed in the results) ----------
model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
sync()


def run_batch(fixed_length):
    extra = {"min_new_tokens": max_tokens} if fixed_length else {}
    start = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        **extra,
    )
    sync()
    elapsed = time.time() - start

    # count the real tokens of each answer (stop counting after the first end-of-answer token)
    total_tokens = 0
    for row in output:
        new = row[input_len:]
        ends = (new == tokenizer.eos_token_id).nonzero()
        total_tokens += (ends[0].item() + 1) if len(ends) > 0 else len(new)
    answer = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return elapsed, total_tokens, answer


for fixed_length in (False, True):
    elapsed, total_tokens, answer = run_batch(fixed_length)
    label = "every answer forced to exactly " + str(max_tokens) + " tokens" if fixed_length else "answers stop naturally"
    print("\n===== MULTI LOAD (normal Transformers, " + str(num_requests) + " requests together, batched; " + label + ") =====")
    print("Device               :", device)
    print("Model load time      :", round(load_time, 2), "sec")
    print("Total requests       :", num_requests)
    print("Total time           :", round(elapsed, 2), "sec")
    print("Total output tokens  :", total_tokens)
    print("Tokens per second    :", round(total_tokens / elapsed, 2), " (throughput)")
    print("Requests per second  :", round(num_requests / elapsed, 2))
    print("Avg time per request :", round(elapsed / num_requests, 3), "sec")
    print("\nSample answer from request 1:")
    print(answer)
