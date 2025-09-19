# import argparse
import os

import logging

# import time
# import random
from typing import Tuple

# import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator

from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import precision_recall_fscore_support, classification_report

from utils_cli import parse_args
from pipeline import construct_model, get_csv_dataset

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
    
    for batch in dataloader:
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
    
    return total_loss.item() / len(dataloader.dataset), eval_metric


def load_lora_model(model, target_modules: list, task_type=TaskType.SEQ_CLS):
    # #If only targeting attention blocks of the model
    # target_modules = ["q_proj", "v_proj"]

    # #If targeting all linear layers
    # target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    default_target_modules = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense"]
    if target_modules is None:
        target_modules = default_target_modules

    # for name, _ in model.named_modules():
    #     print(name)

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

    print("start loading the dataset", "path:", args.dataset_path)
    train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                    model_name=args.model_name, 
                                    path=args.dataset_path, split="train")
    print("finished loading the dataset")

    model_name = args.model_name

    num_labels = len(train_dataset.features["label"].names)

    print("start constructing the model", "model_name:", model_name)
    model = construct_model(model_name=model_name, num_labels=num_labels).to(DEVICE)
    print("finished constructing the model")

    if args.load_lora:
        model = load_lora_model(model=model, target_modules=args.target_modules, task_type=TaskType.SEQ_CLS)
    
    print(args.checkpoint_dir)
    if args.checkpoint_dir is not None:
        # load the model from the checkpoint
        state_dict = torch.load(os.path.join(args.checkpoint_dir, "model.pth"))
        # debug state_dict
        model.load_state_dict(state_dict=state_dict)
    else:
        logger.info("No checkpoint is provided, zero-shot evaluation")
    

    eval_train_dataset = get_csv_dataset(data_name=args.dataset_name, model_name=args.model_name, 
                                         path=args.dataset_path, split="val")
    train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train : {train_acc}")

    # group the validation dataset using target from eval dataset to get the classification report
    if eval_train_dataset['target'] is not None:
        for target in set(eval_train_dataset['target']):
            print(f"Processing target {target}")
            # filter the eval_train dataset by target
            eval_train_dataset_target = [item for item in eval_train_dataset if item['target'] == target]
            train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset_target, batch_size=args.eval_batch_size)
            logger.info(f"Train loss: {train_loss}, Train : {train_acc}")

    print("start evaluating the model on the test/validation dataset")
    print("=============================================") 

    eval_dataset = get_csv_dataset(data_name=args.dataset_name, model_name=args.model_name, 
                                   path=args.dataset_path, split="test")
    eval_loss, eval_acc = evaluate_model(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation : {eval_acc}")
    # group the validation dataset using target from eval dataset to get the classification report

    if eval_dataset['target'] is not None:
        for target in set(eval_dataset['target']):
            print(f"Processing target {target}")
            # filter the eval dataset by target
            eval_dataset_target = [item for item in eval_dataset if item['target'] == target]
            eval_loss, eval_acc = evaluate_model(model=model, dataset=eval_dataset_target, batch_size=args.eval_batch_size)
            logger.info(f"Evaluation loss: {eval_loss}, Evaluation : {eval_acc}")


if __name__ == "__main__":
    main()