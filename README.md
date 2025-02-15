# Entrospect
Official Python implementation of the paper _Entrospect: Information Theoretic Self-Reflection Elicits Better Self-Correction of Language Model_

![fig_main](https://github.com/user-attachments/assets/8cc0e646-172f-4d76-adba-80e8108678cc)

## Prerequisites
1. Install PyTorch from [the official website](https://pytorch.org/get-started/previous-versions/).
2. Install essential python dependencies by running ```pip install --no-cache-dir transformers einops numpy<2```.
3. The default demo comes with [_Qwen 2.5 7B Instruct (4bit GPTQ quantized)_](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4), which relies on ```pip install --no-cache-dir optimum>=1.20.0``` and [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).

## Run the Math Reasoning Demo
Simply run ```python main.py``` after all the required dependencies are adequately deployed for your Python environment. You can also specify your own query in the [**main.py**](https://github.com/henryyantq/Entrospect/blob/main/main.py) file following the guidance of dedicated comments. **Notably, the default demo setting requires approximately 10.5GB of storage (1x language model + 1x [embedding model](https://huggingface.co/jinaai/jina-embeddings-v3) for long-text semantic similarity calculation).**
