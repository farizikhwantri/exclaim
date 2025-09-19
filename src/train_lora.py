# import argparse
import logging
import os
import time
import random
# from typing import Tuple

from accelerate.utils import set_seed

# import evaluate
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils import data
from transformers import default_data_collator

from peft import LoraConfig, get_peft_model, TaskType

# from sklearn.metrics import precision_recall_fscore_support, classification_report

from utils_cli import parse_args
from utils_torch import trainer_by_step, trainer_by_epochs
from pipeline import construct_model, get_csv_dataset
from eval_split import evaluate_model

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")



def train_parse_args():
    train_parser = parse_args("Train text classification models from CSV dataset")

    train_parser.add_argument("--target_modules", type=str, nargs="+", 
                              default=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense"], 
                              help="Target modules to apply LoRA")

    args = train_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def train(
    model_name: str,
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    target_modules: list,
    num_train_steps: int = 0,
    num_labels: int = None,
    task_type: str = TaskType.SEQ_CLS,
) -> nn.Module:

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
    base_model = construct_model(model_name=model_name, num_labels=num_labels)
    print("finished constructing the model")

    # #If only targeting attention blocks of the model
    # target_modules = ["q_proj", "v_proj"]

    # #If targeting all linear layers
    # target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    default_target_modules = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense"]
    if target_modules is None:
        target_modules = default_target_modules

    for name, _ in base_model.named_modules():
        print(name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(base_model, lora_config).to(DEVICE)
    model.print_trainable_parameters()
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # start_time = time.time()
    # model.train()
    # for epoch in range(num_train_epochs):
    #     total_loss = 0.0
    #     for batch in train_dataloader:
    #         optimizer.zero_grad(set_to_none=True)
    #         # handle if model needs token_type_ids
    #         batch_train = {
    #             "input_ids": batch["input_ids"].to(device=DEVICE),
    #             "attention_mask": batch["attention_mask"].to(device=DEVICE),
    #             "labels": batch["labels"].to(device=DEVICE),
    #         }
    #         if "token_type_ids" in batch:
    #             batch_train["token_type_ids"] = batch["token_type_ids"].to(device=DEVICE)

    #         loss = model(**batch_train).loss
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.detach().float()
    #         if random.random() < 0.01:
    #             print(f"batch loss: {loss.detach().float()}")
    #     logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # logging.info(f"Completed training in {elapsed_time:.2f} seconds.")

    if num_train_steps > 0:
        model = trainer_by_step(model, num_train_steps, 
                                dataloader=train_dataloader, 
                                optimizer=optimizer, device=DEVICE)
    else: 
        model = trainer_by_epochs(model, num_train_epochs, 
                                  dataloader=train_dataloader, 
                                  optimizer=optimizer, dataset=dataset, 
                                  device=DEVICE)

    return model


def main():
    args = train_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    if args.seed is not None:
        set_seed(args.seed)

    print("start loading the dataset", "path:", args.dataset_path)
    train_dataset = get_csv_dataset(data_name=args.dataset_name, model_name=args.model_name, path=args.dataset_path, split="train")
    print("finished loading the dataset")

    print("start training the model")
    model = train(
        model_name=args.model_name,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_labels=args.num_labels,
        target_modules=args.target_modules,
        num_train_steps=args.num_train_steps
    )
    print("finished training the model")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))
    

    eval_train_dataset = get_csv_dataset(data_name=args.dataset_name, model_name=args.model_name, path=args.dataset_path, split="train")
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
    eval_dataset = get_csv_dataset(data_name=args.dataset_name, model_name=args.model_name, path=args.dataset_path, split="validation")
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
