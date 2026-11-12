import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
from vllm import LLM, SamplingParams

model_folder = "models/SmolLM2-360M-Instruct"
num_requests = 100   # same as the other scripts
max_tokens = 100     # same limit as the other scripts

# ---------- load model ----------
start = time.time()
llm = LLM(model=model_folder, max_model_len=2048)
load_time = time.time() - start

# ---------- the same 100 questions ----------
all_questions = []
for i in range(num_requests):
    all_questions.append([{"role": "user", "content": "Give me one fun fact number " + str(i + 1)}])

# ---------- warm-up (not timed in the results) ----------
llm.chat(all_questions, SamplingParams(temperature=0, max_tokens=5), use_tqdm=False)

for fixed_length in (False, True):
    if fixed_length:
        settings = SamplingParams(temperature=0, max_tokens=max_tokens, min_tokens=max_tokens)
        label = "every answer forced to exactly " + str(max_tokens) + " tokens"
    else:
        settings = SamplingParams(temperature=0, max_tokens=max_tokens)
        label = "answers stop naturally"

    start = time.time()
    results = llm.chat(all_questions, settings, use_tqdm=False)
    elapsed = time.time() - start

    total_tokens = 0
    for r in results:
        total_tokens += len(r.outputs[0].token_ids)

    print("\n===== MULTI LOAD (vLLM, " + str(num_requests) + " requests together; " + label + ") =====")
    print("Model load time      :", round(load_time, 2), "sec")
    print("Total requests       :", num_requests)
    print("Total time           :", round(elapsed, 2), "sec")
    print("Total output tokens  :", total_tokens)
    print("Tokens per second    :", round(total_tokens / elapsed, 2), " (throughput)")
    print("Requests per second  :", round(num_requests / elapsed, 2))
    print("Avg time per request :", round(elapsed / num_requests, 3), "sec")
    print("\nSample answer from request 1:")
    print(results[0].outputs[0].text)
