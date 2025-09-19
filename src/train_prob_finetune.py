import argparse
import logging
import os
import time
import random
from typing import Tuple

# import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator

from utils_cli import parse_args
from utils_torch import trainer_by_epochs, trainer_by_step
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

logging.info(f"Using device: {DEVICE}")



def train_parse_args():
    train_parser = parse_args("Train text classification models from CSV dataset")
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
    num_labels: int = None,
    probe_epochs: int = None,
    finetune_epochs: int = None,
    num_train_steps: int = 0
) -> nn.Module:

    # how to make pytorch consume max memory available?
    # Pre-allocate GPU memory
    # dummy_tensor = torch.empty((10000, 10000), device=DEVICE)
    # del dummy_tensor

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
    model = construct_model(model_name=model_name, num_labels=num_labels).to(DEVICE)
    print("finished constructing the model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    # Probe the model
    # Freeze all layers except the classification head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


    start_time = time.time()
    model.train()

    if probe_epochs is None:
        probe_epochs = num_train_epochs

    # model = trainer_by_epochs(model=model, num_train_epochs=probe_epochs,
    #                          dataloader=train_dataloader, optimizer=optimizer,
    #                          dataset=dataset, device=DEVICE)

    if num_train_steps > 0:
        model = trainer_by_step(model, num_train_steps, 
                                dataloader=train_dataloader, 
                                optimizer=optimizer, device=DEVICE)
    else: 
        model = trainer_by_epochs(model, num_train_epochs, 
                                  dataloader=train_dataloader, 
                                  optimizer=optimizer, dataset=dataset, 
                                  device=DEVICE)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training probe in {elapsed_time:.2f} seconds.")

    eval_loss, eval_acc = evaluate_model(model=model, dataset=dataset, batch_size=batch_size)
    logging.info(f"Probe loss: {eval_loss}, Probe accuracy: {eval_acc}")


    # Finetune the model
    # Unfreeze all layers
    for name, param in model.named_parameters():
        # print the parameters name
        print(name, param.requires_grad)
        param.requires_grad = True

    finetune_start_time = time.time()
    model.train()
    
    if finetune_epochs is None:
        finetune_epochs = num_train_epochs

    ft_train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    # model = trainer_by_epochs(model, num_train_epochs=finetune_epochs, 
    #                           dataloader=ft_train_dataloader, optimizer=optimizer, 
    #                           dataset=dataset, device=DEVICE)

    if num_train_steps > 0:
        model = trainer_by_step(model, num_train_steps, 
                                dataloader=ft_train_dataloader, 
                                optimizer=optimizer, device=DEVICE)
    else: 
        model = trainer_by_epochs(model, num_train_epochs, 
                                  dataloader=ft_train_dataloader, 
                                  optimizer=optimizer, dataset=dataset, 
                                  device=DEVICE)
    
    ft_loss, ft_acc = evaluate_model(model=model, dataset=dataset, batch_size=batch_size)
    logging.info(f"Finetune loss: {ft_loss}, Finetune accuracy: {ft_acc}")

    end_finetune_time = time.time()

    elapsed_finetune_time = end_finetune_time - finetune_start_time
    logging.info(f"Completed finetuning in {elapsed_finetune_time:.2f} seconds.")

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
    train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                    model_name=args.model_name, 
                                    path=args.dataset_path, split="train",
                                    padding="max_length")
    print("finished loading the dataset")

    logging.info("start training the model")
    model = train(
        model_name=args.model_name,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_labels=args.num_labels,
        num_train_steps=args.num_train_steps
    )
    logging.info("finished training the model")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))
    

    eval_train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                         model_name=args.model_name, 
                                         path=args.dataset_path, split="train",
                                         padding="max_length")
    train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train : {train_acc}")

    if 'target' in eval_train_dataset:
        # group the validation dataset using target from eval dataset to get the classification report
        for target in set(eval_train_dataset['target']):
            print(f"Processing target {target}")
            # filter the eval_train dataset by target
            eval_train_dataset_target = [item for item in eval_train_dataset if item['target'] == target]
            train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset_target, batch_size=args.eval_batch_size)
            logger.info(f"Train loss: {train_loss}, Train : {train_acc}")

     
    eval_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                   model_name=args.model_name, 
                                   path=args.dataset_path, 
                                   split="validation")
    eval_loss, eval_acc = evaluate_model(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation : {eval_acc}")
    # group the validation dataset using target from eval dataset to get the classification report

    if 'target' in eval_dataset:
        print("start evaluating the model on the test/validation dataset")
        print("=============================================")  
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
