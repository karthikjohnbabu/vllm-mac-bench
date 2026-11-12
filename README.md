# vllm-mac-bench

A small, honest benchmark: **plain Hugging Face Transformers vs vLLM on an Apple Silicon Mac**, using the same model, the same questions and the same settings.

I wanted to see what vLLM's continuous batching actually gives on a laptop GPU, so I measured it myself.

## Setup

- Machine: MacBook Pro, Apple M5 Pro, 24 GB memory, macOS 27.2.1
- Model: SmolLM2-360M-Instruct (about 360 million parameters, bfloat16)
- Normal run: Transformers 5.22.2 with PyTorch 2.15.0 on the Mac GPU (`mps`)
- vLLM run: vLLM 0.33.0 with the vllm-metal plugin (MLX on Metal), installed with Homebrew
- Settings: greedy decoding (`temperature=0`), max 200 new tokens for the single question, max 100 for the fun-fact questions, context window 2048
- Each script does one tiny warm-up request before timing starts

## What each script does

| File | What it does |
| --- | --- |
| `1_download_model.py` | Downloads the model into a local `models/` folder |
| `2_run_normal.py` | Plain Transformers: 1 question, then 10 questions one after another |
| `3_run_vllm.py` | vLLM: 1 question, 10 questions one after another, then 100 questions together |
| `4_run_normal_batched.py` | Plain Transformers: the same 100 questions together in one batch, in two modes (answers stop naturally, and every answer forced to 100 tokens) |
| `5_run_vllm_matched.py` | vLLM: the same 100 questions together, in the same two modes |

## Results: first session

Transformers one request at a time vs vLLM. Median of 3 runs, with the range in brackets. Higher is better.

| Measurement | Normal Transformers | vLLM |
| --- | --- | --- |
| Single request, tokens/sec | 182 (181 to 188) | 192 (123 to 222) |
| 10 requests one by one, tokens/sec | 181 (180 to 186) | 203 (197 to 213) |
| 10 requests one by one, requests/sec | 4.0 (4.0 to 4.1) | 5.1 (5.0 to 5.4) |
| 100 requests together, tokens/sec | not run | 4,133 (2,647 to 5,571) |
| 100 requests together, requests/sec | not run | 83 (53 to 112) |

Raw numbers for every run are in `results/summary.csv`.

## What this shows

- **One request at a time:** vLLM and plain Transformers are close. This is a tie, and one vLLM run was actually slower.
- **10 requests one by one:** vLLM was consistently a little ahead, about 12 percent on throughput.
- **100 requests together:** vLLM's median throughput was about 23 times that of plain Transformers handling requests one by one (range about 15 to 31 times). **This mostly measures batching, not the engine**, because the Transformers side was not batched. See the matched comparison below.

## Matched comparison: batched Transformers vs vLLM

A reader pointed out that the first 100-request result compared batched vLLM with one-by-one Transformers. So I added a batched Transformers baseline (`4_run_normal_batched.py`) and a matching vLLM script (`5_run_vllm_matched.py`). Both use the same 100 questions, greedy decoding and a 100-token limit. Each runs in two modes: answers stop naturally, and every answer is forced to exactly 100 tokens, so the work on both sides is identical.

Median of 3 runs, with the range in brackets. Higher is better.

| Measurement (100 requests together) | Batched Transformers | vLLM |
| --- | --- | --- |
| Tokens/sec, answers stop naturally | 2,454 (2,413 to 2,472) | 5,727 (4,464 to 6,170) |
| Requests/sec, answers stop naturally | 46.1 (45.3 to 46.5) | 113.9 (88.8 to 122.7) |
| Tokens/sec, forced to 100 tokens each | 3,820 (3,669 to 4,496) | 6,619 (6,463 to 6,646) |
| Requests/sec, forced to 100 tokens each | 38.2 (36.7 to 45.0) | 66.2 (64.6 to 66.5) |

### What this shows

- Batching alone took plain Transformers from 181 tokens/sec (10 requests one by one) to about 2,450, roughly 13.6 times, with no vLLM involved.
- With identical work (10,000 tokens on both sides), vLLM was about 1.7 times faster than batched Transformers. Its slowest run was above the fastest batched Transformers run.
- With natural lengths the gap was about 2.3 times. The engines produced slightly different token totals (5,030 for vLLM, 5,322 for Transformers), so the forced-length pair is the cleaner comparison.
- The gap is bigger when answer lengths vary. That fits vLLM letting finished requests leave the batch early, but this benchmark does not prove it.

### Caveats for this comparison

- The backend differs as well as the engine: Transformers ran on PyTorch MPS, vLLM ran through vllm-metal (MLX on Metal). Some of the gap may come from the backend and not from vLLM's scheduling. I have not separated the two.
- In vLLM run 1 the log reported that the machine had paged memory out and sized the KV cache down, and that run was the slowest at natural length. Close other apps before running.
- The forced-length Transformers runs slowed down across the three runs (4,496, 3,820, 3,669 tokens/sec). I have not found out why.
- I ran the batched script once to check it before the three saved runs, and left that run out.

## Caveats

- Three runs per setup, on one Mac, with one small model. Treat the numbers as a snapshot, not a verdict.
- The 100-request test varied by about 2 times between runs. I did not isolate the cause (background apps, memory pressure or heat are possible).
- The two engines produce slightly different answers, so output lengths differ. Compare tokens per second and requests per second, not total time.
- In the first session the normal script handled requests one by one, so its 100-request number mixes batching with engine speed. The matched comparison above separates the two.
- The vllm-metal plugin is alpha software. Results may change with newer versions.

## Reproduce it

1. Install vLLM for Apple Silicon:

        brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
        brew install vllm-project/vllm-metal/vllm-metal

2. Make a Python 3.12 environment for the normal run and install the packages:

        conda create -n vllm-mac python=3.12 -y
        conda activate vllm-mac
        pip install vllm-metal

3. Download the model and run the normal script (inside the environment):

        python 1_download_model.py
        python 2_run_normal.py

4. Run the vLLM script with the Python that Homebrew installed:

        conda deactivate
        $(brew --prefix vllm-metal)/libexec/bin/python 3_run_vllm.py

5. Run the matched comparison:

        conda activate vllm-mac
        python 4_run_normal_batched.py
        conda deactivate
        $(brew --prefix vllm-metal)/libexec/bin/python 5_run_vllm_matched.py

## Author

Karthik Babu, [karthikbabu.in](https://karthikbabu.in)

## Licence

MIT
