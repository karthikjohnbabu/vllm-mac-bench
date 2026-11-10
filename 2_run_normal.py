import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_folder = "models/SmolLM2-360M-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"


def sync():
    if device == "mps":
        torch.mps.synchronize()


def ask(model, tokenizer, question, max_new_tokens):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer, input_len, len(new_tokens)


# ---------- load model ----------
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForCausalLM.from_pretrained(model_folder, dtype=torch.bfloat16).to(device)
model.eval()
sync()
load_time = time.time() - start

# ---------- warm-up (not timed in the results) ----------
ask(model, tokenizer, "Say hello.", 5)
sync()

# ---------- single request ----------
question = "Explain what a GPU is in simple words."
start = time.time()
answer, input_tokens, output_tokens = ask(model, tokenizer, question, 200)
sync()
gen_time = time.time() - start

print("\n===== ANSWER =====")
print(answer)

print("\n===== DETAILS (normal Transformers, single request) =====")
print("Device           :", device)
print("Model load time  :", round(load_time, 2), "sec")
print("Input tokens     :", input_tokens)
print("Output tokens    :", output_tokens)
print("Response time    :", round(gen_time, 2), "sec  (latency)")
print("Tokens per second:", round(output_tokens / gen_time, 2))

# ---------- many requests, one after another ----------
num_requests = 10
total_tokens = 0
start = time.time()
for i in range(num_requests):
    _, _, n = ask(model, tokenizer, "Give me one fun fact number " + str(i + 1), 100)
    total_tokens += n
sync()
total_time = time.time() - start

print("\n===== MULTI LOAD (normal Transformers, " + str(num_requests) + " requests one by one) =====")
print("Total requests       :", num_requests)
print("Total time           :", round(total_time, 2), "sec")
print("Total output tokens  :", total_tokens)
print("Tokens per second    :", round(total_tokens / total_time, 2), " (throughput)")
print("Requests per second  :", round(num_requests / total_time, 2))
print("Avg time per request :", round(total_time / num_requests, 3), "sec")
