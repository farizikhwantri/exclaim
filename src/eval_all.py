# import argparse
import os

import logging

# import time
# import random
from typing import Tuple

from tqdm import tqdm

# import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator

from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import precision_recall_fscore_support, classification_report

from sklearn.metrics import roc_auc_score

from utils_cli import parse_args
from pipeline import construct_model, get_csv_dataset
from metric import f2_score

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")



def eval_parse_args():
    eval_parser = parse_args("Evaluate a model on a CSV dataset.")

    # eval_parser.add_argument("--num_labels", type=int, help="Number of labels in the dataset")
    
    # Load LoRA model arguments
    eval_parser.add_argument("--load_lora", action="store_true", help="Load LoRA model")
    eval_parser.set_defaults(load_lora=False)

    eval_parser.add_argument("--target_modules", type=str, nargs="+", default=None, help="Target modules to apply LoRA")

    eval_parser.add_argument("--filter_key", type=str, default="model_name",
                                help="Key to filter documents in the dataset")
    eval_parser.add_argument("--filter_value", type=str, default=None,
                                help="Value to filter documents in the dataset")

    args = eval_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def evaluate_model(model: nn.Module, dataset: data.Dataset, batch_size: int) -> Tuple[float, dict]:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=default_data_collator
    )

    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    # print dataset size
    print(f"Dataset size: {len(dataset)}")
    
    for batch in tqdm(dataloader, "Evaluating", len(dataloader)):
        with torch.no_grad():
            batch_eval = {
                "input_ids": batch["input_ids"].to(device=DEVICE),
                "attention_mask": batch["attention_mask"].to(device=DEVICE),
            }
            if "token_type_ids" in batch:
                batch_eval["token_type_ids"] = batch["token_type_ids"].to(device=DEVICE)
            logits = model(**batch_eval).logits
            labels = batch["labels"].to(device=DEVICE)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").detach()
            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    eval_metric = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    report = classification_report(all_labels, all_predictions)
    # eval_metric['classification_report'] = report
    print(report)
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # roc_auc
    try:
        roc_auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr')
        eval_metric['roc_auc'] = roc_auc
        print(f"ROC AUC Score: {roc_auc}")
    except ValueError as e:
        print(f"Error computing ROC AUC: {e}")

    return total_loss.item() / len(dataloader.dataset), eval_metric


def load_lora_model(model, target_modules: list, task_type=TaskType.SEQ_CLS):
    # #If only targeting attention blocks of the model
    # target_modules = ["q_proj", "v_proj"]

    # #If targeting all linear layers
    # target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    default_target_modules = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense"]
    if target_modules is None:
        target_modules = default_target_modules

    for name, _ in model.named_modules():
        print(name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, lora_config).to(DEVICE)
    model.print_trainable_parameters()

    return model


def main():
    args = eval_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    if args.seed is not None:
        set_seed(args.seed)

    # filter the dataset with a custom filter function if provided
    def filter_function(doc):
        # write a filter function to filter out documents with model=='human'
        return doc.get(args.filter_key, '') != args.filter_value

    filter_fun = filter_function if args.filter_value is not None else None

    print("start loading the dataset", "path:", args.dataset_path)
    dataset = get_csv_dataset(data_name=args.dataset_name, 
                              model_name=args.model_name, 
                              path=args.dataset_path, split="all",
                              label_key=args.label_key,
                              use_fast=args.fast_tokenizer,
                              filter_function=filter_fun)
    print("finished loading the dataset")

    model_name = args.model_name
    label_key = args.label_key
    num_labels = args.num_labels
    if args.num_labels is None:
        # num_labels = len(dataset.features["label"].names)
        num_labels = len(dataset.features[label_key].names)

    print("start constructing the model", "model_name:", model_name)
    model = construct_model(model_name=model_name, num_labels=num_labels).to(DEVICE)
    print("finished constructing the model")

    if args.load_lora:
        model = load_lora_model(model=model, target_modules=args.target_modules, task_type=TaskType.SEQ_CLS)
    
    print(args.checkpoint_dir)
    if args.checkpoint_dir is not None:
        # load the model from the checkpoint
        state_dict = torch.load(os.path.join(args.checkpoint_dir, "model.pth"), map_location=DEVICE)
        # debug state_dict
        model.load_state_dict(state_dict=state_dict)
    else:
        logger.info("No checkpoint is provided, zero-shot evaluation")
    

    loss, perf_metrics = evaluate_model(model=model, dataset=dataset, batch_size=args.eval_batch_size)
    logger.info(f"loss: {loss}, metrics : {perf_metrics}")

    # group the validation dataset using target from eval dataset to get the classification report
    # if 'target' in dataset:
    for target in sorted(set(dataset['target'])):
        print(f"Processing target {target}")
        # filter the eval_train dataset by target
        dataset_target = [item for item in dataset if item['target'] == target]
        loss, perf_metrics = evaluate_model(model=model, dataset=dataset_target, batch_size=args.eval_batch_size)
        logger.info(f"cls loss: {loss}, metrics : {perf_metrics}")


if __name__ == "__main__":
    main()