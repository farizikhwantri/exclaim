# import argparse
import logging
import os
import time
import random
# from typing import Tuple

# import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from torch.utils.data import ConcatDataset
# from transformers import DataCollatorWithPadding
from transformers import default_data_collator

# from sklearn.metrics import precision_recall_fscore_support, classification_report

from utils_cli import parse_args
from utils_torch import trainer_by_step, trainer_by_epochs
from pipeline import construct_model, get_csv_dataset, get_glue_dataset
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


def custom_data_collator(features):
    # Identify keys that are common in all examples
    common_keys = set.intersection(*(set(f.keys()) for f in features))
    # Option 1: Remove keys not present in all examples
    filtered_features = [{k: f[k] for k in common_keys} for f in features]
    return default_data_collator(filtered_features)



def train_parse_args():
    train_parser = parse_args("Train text classification models from CSV dataset")

    train_parser.add_argument("--continue_training", action="store_true",
                              help="Continue training from a checkpoint")
    train_parser.set_defaults(continue_training=False)

    # Add an argument for an glue dataset name
    train_parser.add_argument("--aux_dataset_name", type=str, default=None,
                              help="Name of the auxiliary dataset for training")
    
    # Add an argument for auxillary dataset path for synthetic data
    train_parser.add_argument("--aux_dataset_path", type=str, default=None,
                              help="Path to the auxiliary dataset for training")
    
    train_parser.add_argument("--filter_key", type=str, default="model_name",
                                help="Key to filter documents in the dataset")
    train_parser.add_argument("--filter_value", type=str, default=None,
                                help="Value to filter documents in the dataset")

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
    num_train_steps: int=0,
    num_labels: int = None,
    grad_accumulation_steps: int = 1,
    continue_training: bool = False,
    checkpoint_dir: str = None,
) -> nn.Module:

    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_data_collator,
    )

    if num_labels is None:
        num_labels = len(dataset.features["labels"].names)

    # print("num_labels:", num_labels)
    print(dataset)
    # assert num_labels >= len(dataset.features["label"].names), \
    #     "Number of labels must be equal or greater than the number of labels in the dataset"

    print("start constructing the model", "model_name:", model_name)
    model = construct_model(model_name=model_name, num_labels=num_labels).to(DEVICE)
    print("finished constructing the model")

    if continue_training and checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        else:
            print(f"Checkpoint {checkpoint_path} does not exist. Starting training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if num_train_steps > 0:
        logging.info(f"Training for {num_train_steps} steps")
        model = trainer_by_step(model, num_train_steps, 
                                dataloader=train_dataloader, 
                                optimizer=optimizer, device=DEVICE,
                                grad_accumulation_steps=grad_accumulation_steps)
    else: 
        model = trainer_by_epochs(model, num_train_epochs, 
                                  dataloader=train_dataloader, 
                                  optimizer=optimizer, dataset=dataset, 
                                  device=DEVICE,
                                  grad_accumulation_steps=grad_accumulation_steps)

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
    #             # print(f"batch loss: {loss.detach().float()}")
    #             logging.info(f"batch loss: {loss.detach().float()}")
    #     logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # logging.info(f"Completed training in {elapsed_time:.2f} seconds.")

    return model


def main():
    args = train_parse_args()
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
    train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                    model_name=args.model_name, 
                                    path=args.dataset_path, split="train", 
                                    label_key=args.label_key,
                                    padding="max_length",
                                    filter_function=filter_fun)
    print("finished loading the dataset")

    # if glue dataset is specified, load and combine them
    if args.aux_dataset_name:
        print("start loading the auxiliary dataset", "name:", args.aux_dataset_name)
        aux_dataset = get_glue_dataset(data_name=args.aux_dataset_name, 
                                       model_name=args.model_name, 
                                       split="train")
        print("finished loading the auxiliary dataset")
        # Combine the datasets
        combined_dataset = ConcatDataset([train_dataset, aux_dataset])
        train_dataset = combined_dataset

    # If auxiliary dataset is specified, load and combine them
    if args.aux_dataset_path:
        print("start loading the auxiliary dataset", "path:", args.aux_dataset_path)
        aux_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                      model_name=args.model_name, 
                                      path=args.aux_dataset_path, split="train", 
                                      label_key=args.label_key,
                                      padding="max_length")
        print("finished loading the auxiliary dataset")
        # Combine the datasets
        combined_dataset = ConcatDataset([train_dataset, aux_dataset])
        train_dataset = combined_dataset

    # args.num_labels = len(train_dataset.features[args.label_key].names)
    # if args.num_labels is None:
    #     args.num_labels = len(train_dataset.features[args.label_key].names)

    # print object attributes
    # print("train_dataset attributes:", train_dataset.datasets.__dict__.keys())

    if args.num_labels is None:
        try:
            if isinstance(train_dataset, ConcatDataset):
                # Use the first underlying dataset to obtain features
                base_ds = train_dataset.datasets[0]
                args.num_labels = len(base_ds.features[args.label_key].names)
            else:
                args.num_labels = len(train_dataset.features[args.label_key].names)
        except AttributeError:
            # Fallback if features attribute is not available; 
            # e.g., compute unique labels from examples
            print("No features attribute found, computing unique labels from examples.")
            unique_labels = set(example[args.label_key] for example in train_dataset)
            args.num_labels = len(unique_labels)

    print("start training the model")
    model = train(
        model_name=args.model_name,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_labels=args.num_labels,
        num_train_steps=args.num_train_steps,
        grad_accumulation_steps=args.grad_accumulation_steps,
        continue_training=args.continue_training,
        checkpoint_dir=args.checkpoint_dir,
    )
    print("finished training the model")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))

    eval_train_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                         model_name=args.model_name, 
                                         path=args.dataset_path, 
                                         split="train",
                                         label_key=args.label_key,
                                         padding="max_length",
                                         filter_function=filter_fun)
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

     
    eval_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                   model_name=args.model_name, 
                                   path=args.dataset_path, 
                                   split="validation",
                                   label_key=args.label_key,
                                   padding="max_length",
                                   filter_function=filter_fun)
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