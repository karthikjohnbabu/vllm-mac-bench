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

## Results

Median of 3 runs, with the range in brackets. Higher is better.

| Test | Normal Transformers | vLLM |
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
- **100 requests together:** this is where vLLM pulls away. The median throughput was about 23 times the normal run's, and the range was about 15 to 31 times.

## Caveats

- Three runs per setup, on one Mac, with one small model. Treat the numbers as a snapshot, not a verdict.
- The 100-request test varied by about 2 times between runs. I did not isolate the cause (background apps, memory pressure or heat are possible).
- The two engines produce slightly different answers, so output lengths differ. Compare tokens per second and requests per second, not total time.
- The normal script was not run with 100 requests, because it handles requests one by one and the fair comparison is on throughput.
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

## Author

Karthik Babu, [karthikbabu.in](https://karthikbabu.in)

## Licence

MIT
