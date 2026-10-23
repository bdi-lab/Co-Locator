# Ask KG Agent: A Multi-Agent Framework for Code Localization Using Code and Knowledge Graphs

This is the official implementation of the paper **"Ask KG Agent: A Multi-Agent Framework for Code Localization Using Code and Knowledge Graphs"** accepted to the 40th Annual Conference on Neural Information Processing Systems (NeurIPS 2026).

All codes are written by **Chanyoung Chung** ([chanyoung.chung@kaist.ac.kr](mailto:chanyoung.chung@kaist.ac.kr)).

If you use this repository, please cite our paper.
```bibtex
@inproceedings{colocator,
  author={Chanyoung Chung and Jeemin Kim and Joyce Jiyoung Whang},
  title={Ask KG Agent: A Multi-Agent Framework for Code Localization Using Code and Knowledge Graphs},
  year={2026},
  booktitle={Proceedings of the 40th Annual Conference on Neural Information Processing Systems},
}
```

## Environmental Settings

We used Python 3.10.19, CUDA 12.8, and PyTorch 2.4.0. Install the remaining dependencies using the following command:

```
pip install -r requirements.txt
```

## Steps to run Co-Locator

### Step 1. Code Graph Construction

Execute the following commands `./cg/` to construct code graphs:

```
# SWE-bench-Live (lite)
python main.py -d SWE-bench-Live/SWE-bench-Live -s lite
python embed.py -d SWE-bench-Live/SWE-bench-Live -s lite

# LocBench
python main.py -d czlll/Loc-Bench_v1 -s test
python embed.py -d czlll/Loc-Bench_v1 -s test
```

### Step 2. Knowledge Graph Construction

Execute the following commands in `./kg/` to construct knowledge graphs:

```
# SWE-bench-Live (lite)
python save_issues.py -d SWE-bench-Live/SWE-bench-Live -s lite
python extract_kg.py -d SWE-bench-Live/SWE-bench-Live -s lite
python save_kg.py -d SWE-bench-Live/SWE-bench-Live -s lite
python embed.py -d SWE-bench-Live/SWE-bench-Live -s lite

# LocBench
python save_issues.py -d czlll/Loc-Bench_V1 -s test
python extract_kg.py -d czlll/Loc-Bench_V1 -s test
python save_kg.py -d czlll/Loc-Bench_V1 -s test
python embed.py -d czlll/Loc-Bench_V1 -s test
```

To run `extract_kg.py`, replace `GITHUB_TOKEN` and `OPENAI_API_KEY` with the actual token and key.

### Step 3. Run Co-Locator⁻

You can serve Qwen2.5-Coder-32B-Instruct with vLLM by the following command:

```
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --gpu_memory_utilization 0.95 --port 8000 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 131072
```

For the Retriever agent, run `python server.py` in a separate window.

To run Co-Locator⁻ on SWE-bench-Live (lite), use the following commands in `./CoLocator/`:

```
# Initial predictions with Retriever
python pred_with_problem.py -d SWE-bench-Live/SWE-bench-Live -s lite

# Collaborative code localization
python run.py -d SWE-bench-Live/SWE-bench-Live -s lite

# Post-retrieval and Evaluation
python postprocessing.py -d SWE-bench-Live/SWE-bench-Live -s lite
```

To run Co-Locator⁻ on LocBench, use the following commands in `./CoLocator/`:

```
# Initial predictions with Retriever
python pred_with_problem.py -d czlll/Loc-Bench_V1 -s test

# Collaborative code localization
python run.py -d czlll/Loc-Bench_V1 -s test

# Post-retrieval and Evaluation
python postprocessing.py -d czlll/Loc-Bench_V1 -s test
```

You should replace `OPENAI_API_KEY` in `main.py` with the actual OpenAI API key to use GPT-4o.

### Step 4. Run Co-Locator⁺

Use the following commands in `./CoLocator_Rerank/` to run Co-Locator⁺:

```
# SWE-bench-Live (lite)
python run.py -d SWE-bench-Live/SWE-bench-Live -s lite

# LocBench
python run.py -d czlll/Loc-Bench_V1 -s test
```

You can evaluate Co-Locator⁺ by the following commands:

```
# SWE-bench-Live (lite)
python postprocessing.py -d SWE-bench-Live/SWE-bench-Live -s lite

# LocBench
python postprocessing.py -d czlll/Loc-Bench_V1 -s test
```

You should replace `OPENAI_API_KEY` in `main.py` with the actual OpenAI API key to use GPT-4o.
