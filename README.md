# Entrospect
Official Python implementation of the paper _Entrospect: Information Theoretic Self-Reflection Elicits Better Self-Correction of Language Model_

## The Pipeline Visualized

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a858a53-c83d-42f0-9fcf-f6d172bc5ef4" alt="img" width="800">
</p>

The pipeline of our Entrospect prompt-driven framework, extending the original Self-Refine structure with an **Optimal Reflective Pattern Selector (ORPS)** and a universal **semantic similarity-based stopping condition**. The framework requires no supervised pre-training or access to the model's internal parameters, granting it to be generalizable to various language models and reasoning tasks.

## How It Differs from Self-Refine, Its Predecessor

<p align="center">
  <img width="410" alt="image" src="https://github.com/user-attachments/assets/a2345206-3b04-4519-ad77-11498726136c">
</p>

The single-round refinement of an initial response for the same query, comparing Self-Refine and our proposed _Entrospect_. Self-Refine fully relies on the model's self-reflected feedback, where any biases introduced during self-reflection are directly carried over into the refinement stage, hindering constructive improvements. On the other hand, our Entrospect identifies the optimal revision suggestion from an itemized ouput of the self-reflection, enabling Entrospect to achieve more robust and reliable response refinement.

## Prerequisites
1. Install PyTorch from [the official website](https://pytorch.org/get-started/previous-versions/).
2. Install essential python dependencies by running ```pip install --no-cache-dir transformers einops numpy<2```.
3. The default demo comes with [_Qwen 2.5 7B Instruct (4bit GPTQ quantized)_](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4), which relies on ```pip install --no-cache-dir optimum>=1.20.0``` and [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).

## Run the Math Reasoning Demo
Simply run ```python main.py``` after all the required dependencies are adequately deployed for your Python environment. You can also specify your own query in the [**main.py**](https://github.com/henryyantq/Entrospect/blob/main/main.py) file following the guidance of dedicated comments. **Notably, the default demo setting requires approximately 10.5GB of storage (1x language model + 1x [embedding model](https://huggingface.co/jinaai/jina-embeddings-v3) for long-text semantic similarity calculation).**
