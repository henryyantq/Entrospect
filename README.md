# Entrospect
Official Python implementation of the paper _Entrospect: Information Theoretic Self-Reflection Elicits Better Self-Correction of Language Model_

## Prerequisites
1. Install PyTorch from [the official website](https://pytorch.org/get-started/previous-versions/).
2. Install essential python dependencies by running ```pip install --no-cache-dir transformers einops numpy<2```.
3. The default demo comes with _Qwen 2.5 7B Instruct (4bit GPTQ quantized)_, which relies on ```pip install --no-cache-dir optimum>=1.20.0``` and [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).

### Run the Demo ```
