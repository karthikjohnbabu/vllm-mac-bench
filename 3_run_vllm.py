import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
from vllm import LLM, SamplingParams

model_folder = "models/SmolLM2-360M-Instruct"

# ---------- load model ----------
start = time.time()
llm = LLM(model=model_folder, max_model_len=2048)
load_time = time.time() - start

# ---------- warm-up (not timed in the results) ----------
llm.chat([{"role": "user", "content": "Say hello."}], SamplingParams(temperature=0, max_tokens=5), use_tqdm=False)

# ---------- single request ----------
settings = SamplingParams(temperature=0, max_tokens=200)
question = [{"role": "user", "content": "Explain what a GPU is in simple words."}]

start = time.time()
result = llm.chat(question, settings, use_tqdm=False)
gen_time = time.time() - start

input_tokens = len(result[0].prompt_token_ids)
output_tokens = len(result[0].outputs[0].token_ids)

print("\n===== ANSWER =====")
print(result[0].outputs[0].text)

print("\n===== DETAILS (vLLM, single request) =====")
print("Model load time  :", round(load_time, 2), "sec")
print("Input tokens     :", input_tokens)
print("Output tokens    :", output_tokens)
print("Response time    :", round(gen_time, 2), "sec  (latency)")
print("Tokens per second:", round(output_tokens / gen_time, 2))

# ---------- 10 requests, one after another (same test as the normal script) ----------
settings = SamplingParams(temperature=0, max_tokens=100)
num_seq = 10
total_tokens = 0
start = time.time()
for i in range(num_seq):
    q = [{"role": "user", "content": "Give me one fun fact number " + str(i + 1)}]
    r = llm.chat(q, settings, use_tqdm=False)
    total_tokens += len(r[0].outputs[0].token_ids)
total_time = time.time() - start

print("\n===== MULTI LOAD (vLLM, " + str(num_seq) + " requests one by one) =====")
print("Total time           :", round(total_time, 2), "sec")
print("Total output tokens  :", total_tokens)
print("Tokens per second    :", round(total_tokens / total_time, 2), " (throughput)")
print("Requests per second  :", round(num_seq / total_time, 2))

# ---------- 100 requests together ----------
num_requests = 100
all_questions = []
for i in range(num_requests):
    all_questions.append([{"role": "user", "content": "Give me one fun fact number " + str(i + 1)}])

start = time.time()
results = llm.chat(all_questions, settings, use_tqdm=False)
total_time = time.time() - start

total_tokens = 0
for r in results:
    total_tokens += len(r.outputs[0].token_ids)

print("\n===== MULTI LOAD (vLLM, " + str(num_requests) + " requests together) =====")
print("Total requests       :", num_requests)
print("Total time           :", round(total_time, 2), "sec")
print("Total output tokens  :", total_tokens)
print("Tokens per second    :", round(total_tokens / total_time, 2), " (throughput)")
print("Requests per second  :", round(num_requests / total_time, 2))
print("Avg time per request :", round(total_time / num_requests, 3), "sec")

print("\nSample answer from request 1:")
print(results[0].outputs[0].text)
