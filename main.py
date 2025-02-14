import re
import ast
import torch
import warnings
import numpy as np
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


warnings.filterwarnings("ignore")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def chat_hf(model, tokenizer, inst, max_new_tokens=8192):
    if type(inst) == str:
        message = [{"role": "user", "content": inst}]
    else:
        message = inst
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=None,
        top_p=None,
        temperature=None
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def entropy(model, tokenizer, inst):
    message = [{"role": "user", "content": inst}]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**model_inputs, labels=model_inputs["input_ids"])
    return outputs.loss.item()


def mutual_info(model, tokenizer, query, prompt):
    marginal_entropy = entropy(model, tokenizer, query)
    conditional_entropy = entropy(model, tokenizer, f"{query}\n{prompt}")
    return marginal_entropy - conditional_entropy


def get_embedding(text, model, adapter="text-matching"):
    embedding = model.encode([text], task=adapter)
    return embedding[0]


def similarity(model, candidate_1, candidate_2):
    embedding_1 = get_embedding(candidate_1, model)
    embedding_2 = get_embedding(candidate_2, model)
    return embedding_1 @ embedding_2.T


def entrospect(
    model, tokenizer, similarity_model,
    query="Solve the following quadratic equation for x: 3x^2 - 12x + 9 = 0",
    prompt="Think step-by-step.",
    max_n_search=5,
    similarity_thresh=0.9
):
    inst = f"{query}\n{prompt}"
    initial_mi = mutual_info(model, tokenizer, query, prompt)
    prev_rsp = chat_hf(model, tokenizer, inst)
    initial_rsp = prev_rsp
    selected_rsp_d = prompt
    n_search = 0
    while True:
        prev_best_mi = initial_mi
        n_search += 1
        inst = f"[Problem]\n{query}\n\n[Solution]\n{prev_rsp}\n\nPlease analyze the above solution to the given math problem. Your task is to identify any deficiencies or errors in the solution. Please follow these steps:\n\n1. **Understand the Problem**: Carefully read and comprehend the math problem to grasp what is being asked.\n2. **Review the Solution**: Examine the provided solution step by step.\n3. **Identify Deficiencies**: Look for errors in calculations, logical reasoning, or assumptions. Note any steps that are missing, incorrect, or insufficiently justified.\n4. **Assess Clarity and Completeness**: Evaluate the explanation for clarity and whether it fully addresses the problem."
        rsp_d = chat_hf(model, tokenizer, inst)
        inst = f"Format all independent revision suggestions that can be extracted from the following in a **Python List of Strings**:\n{rsp_d}"
        print(f">>> Iter {n_search} - Reflecting... <<<")
        rsp_d = chat_hf(model, tokenizer, inst)
        pattern = r"(\[.*?\])"
        rsp_d = re.search(pattern, rsp_d, re.DOTALL).group(1)
        rsp_d = ast.literal_eval(rsp_d)
        flag = 0
        print(f">>> Iter {n_search} - Conducting ORPS... <<<")
        for i in range(len(rsp_d)):
            q = f"[Problem]\n{query}\n\n[Solution]\n{prev_rsp}\n\nPlease provide a refined solution for the problem given the previous solution."
            mi = mutual_info(model, tokenizer, q, rsp_d[i])
            if mi >= prev_best_mi:
                flag = 1
                prev_best_mi = mi
                selected_rsp_d = rsp_d[i]
        if flag == 0:
            rsp = prev_rsp
            break
        inst = f"{q}\n{selected_rsp_d}"
        print(f">>> Iter {n_search} - Refining... <<<")
        rsp = chat_hf(model, tokenizer, inst)
        print(f">>> Iter {n_search} - Convergence checking... <<<")
        s = similarity(similarity_model, prev_rsp, rsp)
        print(f">>> Iter {n_search} - Similarity calculated: {s} <<<")
        if s >= similarity_thresh:
            break
        prev_rsp = rsp
        if n_search == max_n_search:
            break
    return initial_rsp, rsp


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    similarity_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", 
        trust_remote_code=True
    )
    y_0, y_pred = entrospect(
        model, tokenizer, similarity_model,
        # query=""
        # Specify the ``query'' parameter with your own query.
    )
    print(f"\n[INITIAL ANSWER]\n{y_0}")
    print(f"\n[REFINED ANSWER]\n{y_pred}")