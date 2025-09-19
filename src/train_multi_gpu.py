import argparse
import logging
import os
import time
import random
from typing import Tuple

from accelerate import Accelerator

# import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import precision_recall_fscore_support, classification_report

from utils_cli import parse_args
from pipeline import construct_model, get_csv_dataset

accelerator = Accelerator()


def multi_gpu_parse_args():
    parser = parse_args("Train a model on a CSV dataset with Multi-GPU.")
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def accelerate_train(
    model_name: str,
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_labels: int = None,
    num_train_steps: int = 0,
    tokenizer: AutoTokenizer = None,
    grad_accumulation_steps: int = 1
) -> nn.Module:
    # print basic parameters with simple object
    logging.info(f"Model Name: {model_name}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Number of Training Epochs: {num_train_epochs}")
    logging.info(f"Learning Rate: {learning_rate}")
    logging.info(f"Weight Decay: {weight_decay}")
    logging.info(f"Number of Labels: {num_labels}")
    logging.info(f"Number of Training Steps: {num_train_steps}")
    logging.info(f"Gradient Accumulation Steps: {grad_accumulation_steps}")

    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    if num_labels is None:
        num_labels = len(dataset.features["label"].names)

    assert num_labels >= len(dataset.features["label"].names), \
        "Number of labels must be equal or greater than the number of labels in the dataset"

    print("start constructing the model", "model_name:", model_name)
    model = construct_model(model_name=model_name, num_labels=num_labels,
                            tokenizer=tokenizer)
    print("finished constructing the model")

    steps = num_train_steps if num_train_steps > 0 else num_train_epochs * len(dataset) // batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=steps
    )

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)

    start_time = time.time()
    model.train()

    step = 0

    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_train = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]
            }
            if "token_type_ids" in batch:
                batch_train["token_type_ids"] = batch["token_type_ids"]

            loss = model(**batch_train).loss

            # loss.backward()
            accelerator.backward(loss)

            # optimizer.step()
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach().float()
            if random.random() < 0.01:
                print(f"batch loss: {loss.detach().float()}")

            step += 1
            if num_train_steps > 0 and step >= num_train_steps:
                break

        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")

        if num_train_steps > 0 and step >= num_train_steps:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")

    return model


def evaluate_model(model: nn.Module, dataset: data.Dataset, batch_size: int) -> Tuple[float, dict]:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=default_data_collator
    )

    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    model, dataloader = accelerator.prepare(model, dataloader)
    
    for batch in dataloader:
        with torch.no_grad():
            batch_eval = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            if "token_type_ids" in batch:
                batch_eval["token_type_ids"] = batch["token_type_ids"]

            logits = model(**batch_eval).logits

            # labels = batch["labels"].to(device=DEVICE)
            labels = batch["labels"]

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


def main():
    args = multi_gpu_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    if args.seed is not None:
        set_seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, 
                                              use_fast=True, 
                                              trust_remote_code=True)

    print("start loading the dataset", "path:", args.dataset_path)
    train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                    model_name=args.model_name, 
                                    path=args.dataset_path, split="train",
                                    label_key=args.label_key, 
                                    tokenizer=tokenizer,)
    print("finished loading the dataset")

    print("start training the model")
    model = accelerate_train(
        model_name=args.model_name,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_labels=args.num_labels,
        num_train_steps=args.num_train_steps,
        tokenizer=tokenizer,
        grad_accumulation_steps=args.grad_accumulation_steps
    )
    print("finished training the model")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))

    eval_train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                         model_name=args.model_name, 
                                         path=args.dataset_path, split="train",
                                         label_key=args.label_key,
                                         tokenizer=tokenizer)
    train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train : {train_acc}")

    # group the validation dataset using target from eval dataset to get the classification report
    if 'target' in eval_train_dataset:
        for target in set(eval_train_dataset['target']):
            print(f"Processing target {target}")
            # filter the eval_train dataset by target
            eval_train_dataset_target = [item for item in eval_train_dataset if item['target'] == target]
            train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset_target, batch_size=args.eval_batch_size)
            logger.info(f"Train loss: {train_loss}, Train : {train_acc}")

    print("start evaluating the model on the test/validation dataset")
    print("=============================================")  
    eval_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                   model_name=args.model_name, 
                                   path=args.dataset_path, split="validation",
                                   label_key=args.label_key)
    
    eval_loss, eval_acc = evaluate_model(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation : {eval_acc}")
    # group the validation dataset using target from eval dataset to get the classification report

    if 'target' in eval_dataset: 
        for target in set(eval_dataset['target']):
            print(f"Processing target {target}")
            # filter the eval dataset by target
            eval_dataset_target = [item for item in eval_dataset if item['target'] == target]
            eval_loss, eval_acc = evaluate_model(model=model, dataset=eval_dataset_target, batch_size=args.eval_batch_size)
            logger.info(f"Evaluation loss: {eval_loss}, Evaluation : {eval_acc}")

    # if args.checkpoint_dir is not None:
    #     torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
    