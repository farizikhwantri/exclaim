import os
import csv
import random
import pandas as pd
import argparse
import logging
from tqdm import tqdm

from typing import List, Dict, Callable
# from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
# from accelerate import init_empty_weights, infer_auto_device_map

from utils_prompt import build_prompt
from utils_prompt import build_prompt_curried
from utils_prompt import zero_shot_prompt
from utils_prompt import multi_hop_prompt
from utils_prompt import multi_hop_prompt_curried

# check device
if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal LLM for NLI entailment using in-context learning / zero-shot classification")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset file (optional)")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero-shot classification")
    parser.set_defaults(zero_shot=False)
    parser.add_argument("--multi_hop", action="store_true", help="Use multi-hop classification")
    parser.set_defaults(multi_hop=False)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the predictions")
    parser.add_argument("--model_type", type=str, default="causal", choices=["causal", "seq2seq"], help="Model type: causal or seq2seq")
    parser.add_argument("--model_name", type=str, default="gpt2-xl", help="Causal LLM model name (>1B parameters)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--max_length", type=int, default=64, help="Max generation length")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of examples per batch (processing sequentially)")
    # load_in_4bit or load_in_8bit
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.set_defaults(load_in_8bit=False)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.set_defaults(load_in_4bit=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    return args

def load_csv_dataset(csv_path):
    dataset = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expecting keys: premise, hypothesis, label (optional)
            instance ={
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row.get("label", "").lower()  # can be 'entailment' or other
            }
            # copy the rest of the row
            for key, value in row.items():
                if key not in instance:
                    instance[key] = value
            dataset.append(instance)
    return dataset


def classify_entailment(generated_text):
    # Postprocess: if generated text (lowercase) contains 'entail', mark as entailment.
    # Otherwise choose not entailment.
    text = generated_text.strip().lower()
    if "not" in text or "not entail" in text:
        return "not_entailment"
    elif "ent" in text or "entail" in text:
        return "entailment"
    return "not_entailment"

def evaluate_model(model, tokenizer, dataset, max_length, bulild_prompt_func=build_prompt, model_type="causal", debug=False):
    model.eval()
    all_predictions = []
    all_labels = []
    eval_res = []
    for example in tqdm(dataset, desc="Evaluating"):
        # prompt = bulild_prompt_func(example["premise"], example["hypothesis"])
        prompt = bulild_prompt_func(example)
        if debug or random.random() < 0.01:
            print("------------------------")
            print(prompt)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if debug or random.random() < 0.01:
            print("------------------------")
            print(generated)

        # extract generated answer after the prompt
        # answer = generated[len(prompt):].strip().split()[0]  # first token word

        if model_type == "causal":
            answer = generated[len(prompt):].strip().split()
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = "not entailment"
        elif model_type == "seq2seq":  
            answer = generated
        pred = classify_entailment(answer)
        all_predictions.append(pred)
        if example["label"]:
            all_labels.append(example["label"])
        eval_res.append({
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "prediction": pred,
            "label": example["label"],
            "generated": generated,
        })
    return all_predictions, all_labels, eval_res

def load_large_model(model_name_or_path, load_in_8bit=False, load_in_4bit=False):
    """
    Load a large language model across multiple GPUs
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Set specific parameters for large models
    kwargs = {
        "device_map": "auto",       # Automatically distribute across GPUs
        "torch_dtype": torch.float16,  # Use half precision
        "trust_remote_code": True,
    }
    # remove device map if it mps
    if torch.backends.mps.is_available():
        kwargs.pop("device_map", None)
        print("Using MPS device, removing device_map")
    
    # Add quantization options if requested
    if load_in_8bit:
        print("Using 8-bit quantization")
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        print("Using 4-bit quantization")
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_quant_type"] = "nf4"
        kwargs["bnb_4bit_compute_dtype"] = torch.float16
    
    # Load the model with the specified parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs
    )
    
    # Print memory usage per GPU
    print("\nMemory usage after loading:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    
    return model, tokenizer

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Arguments: %s", args)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    logger.info("Loading dataset from: %s", args.dataset_path)
    dataset = load_csv_dataset(args.dataset_path)
    logger.info("Dataset size: %d", len(dataset))
    
    logger.info("Loading model: %s", args.model_name)
    model = None
    # if args.model_type == "causal":
    #     model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    # elif args.model_type == "seq2seq":
    #     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(DEVICE)

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name,
    #                                           trust_remote_code=True,
    #                                           revision="main")
    model, tokenizer = load_large_model(args.model_name, 
                                        load_in_4bit=args.load_in_4bit, 
                                        load_in_8bit=args.load_in_8bit)

    # Ensure the tokenizer has an eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token

    build_prompt_func = build_prompt

    if args.train_path:
        train_dataset = load_csv_dataset(args.train_path)
        # choose randomly two examples from the training dataset of positive and negative labels
        # random sample two examples
        pos_example, neg_example = None, None
        while True:
            pos_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
            if pos_example["label"] == "entailment":    
                pos_example = (pos_example["premise"], pos_example["hypothesis"])
                break
        while True:
            neg_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
            if neg_example["label"] != "entailment":   
                neg_example = (neg_example["premise"], neg_example["hypothesis"]) 
                break
        build_prompt_func = build_prompt_curried(pos_example, neg_example)
    if args.zero_shot:
        build_prompt_func = zero_shot_prompt

    if args.multi_hop:
        build_prompt_func = multi_hop_prompt
        if args.train_path:
            train_dataset = load_csv_dataset(args.train_path)
            # choose randomly two examples from the training dataset of positive and negative labels
            # random sample two examples
            pos_example, neg_example = None, None
            while True:
                pos_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
                if pos_example["label"] == "entailment":    
                    pos_example = (pos_example["premise"], pos_example["hypothesis"])
                    break
            while True:
                neg_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
                if neg_example["label"] != "entailment":   
                    neg_example = (neg_example["premise"], neg_example["hypothesis"]) 
                    break
            build_prompt_func = multi_hop_prompt_curried(pos_example, neg_example)

    not_entailment_length = tokenizer("not entailment", return_tensors="pt").input_ids.shape[1]
    entailment_length = tokenizer("entailment", return_tensors="pt").input_ids.shape[1]

    max_length = max(not_entailment_length, entailment_length) 
    # max_length = args.max_length

    # if debug use only 10 examples
    if args.debug:
        # dataset = dataset[:10]
        # get 10% number of examples from the dataset
        num_samples = int(len(dataset) * 0.1) if args.num_samples is None else args.num_samples
        if args.multi_hop:
            # pick examples with multi-hop > 1
            # assert int(dataset[0].get("hop", 1)) >= 1, "dataset should have hop > 1"
            # dataset = [ex for ex in dataset if int(ex.get("hop", 1)) > 1]
            # check if premise contains separator, meaning it has multi-hop
            dataset = [ex for ex in dataset if "||" in ex["premise"]]
            logger.info("Debug mode: using only examples with multi-hop > 1")
            # print(len(dataset))
            random_indices = torch.randint(0, len(dataset), (num_samples,))
            dataset = [dataset[i] for i in random_indices]
        else:
            # pick randomly 10 examples
            random_indices = torch.randint(0, len(dataset), (num_samples,))
            dataset = [dataset[i] for i in random_indices]
            logger.info("Debug mode: using only examples")

    predictions, labels, eval_res = evaluate_model(model, tokenizer, dataset, max_length, build_prompt_func, 
                                                   model_type=args.model_type, debug=args.debug)
    
    # If labels exist, print a simple report
    if labels:
        from sklearn.metrics import classification_report
        report = classification_report(labels, predictions, zero_division=0)
        print("Classification Report:\n", report)
    else:
        logger.info("Zero-shot predictions (first 10): %s", predictions[:10])

    # Save predictions to output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "predictions.csv")

        # with open(output_file, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["premise", "hypothesis", "prediction", "label"])
        #     for example, pred in zip(dataset, predictions):
        #         writer.writerow([example["premise"], example["hypothesis"], pred, example.get("label", "")])

        df = pd.DataFrame(eval_res)
        # remove the 'generated' column if it exists
        df = df.drop(columns=["generated"], errors='ignore')
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info("eval saved to: %s", output_file)
        logger.info("Predictions saved to: %s", output_file)
    
if __name__ == "__main__":
    main()
