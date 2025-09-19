import os
import inspect

import logging
import json
import pickle

# import time
import random
from typing import List, Dict

import numpy as np

# import evaluate
import torch
# import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator
from transformers import AutoTokenizer, PreTrainedTokenizer

# tested with captum 0.7.0
from captum.attr import LayerIntegratedGradients, \
                        configure_interpretable_embedding_layer, \
                        remove_interpretable_embedding_layer

# tested with captum 0.7.0
from captum.attr import InputXGradient, IntegratedGradients, ShapleyValueSampling, \
                        LayerGradientXActivation, \
                        LayerConductance, \
                        DeepLift, Occlusion

from ferret import LIMEExplainer, SHAPExplainer, GradientExplainer, IntegratedGradientExplainer
from ferret import Benchmark

# from sklearn.metrics import precision_recall_fscore_support, classification_report

from utils_cli import parse_args
from utils_torch import get_nested_attr
from pipeline import construct_model, get_csv_dataset
from eval_split import load_lora_model, TaskType

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# print(f"Using device: {DEVICE}")

attributions_map = {
    "grad": InputXGradient,
    "ig": IntegratedGradients,
    "shapley": ShapleyValueSampling,
    "deeplift": DeepLift,
    "lig": LayerIntegratedGradients,
    "occ:": Occlusion,
    "LayerGradientXActivation": LayerGradientXActivation,
    # "layer_deeplift": LayerDeepLift,
    "LayerConductance": LayerConductance,
}

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             token_type_ids=token_type_ids)
        return outputs.logits

# Now you can use wrapped_model for DeepLIFT attribution

def get_attribution_method(method: str):
    if method not in attributions_map:
        raise ValueError(f"Attribution method {method} not supported, \
                         available methods: {attributions_map.keys()}")

    return attributions_map[method]


def function_accepts_argument(func, arg_name):
    signature = inspect.signature(func)
    return arg_name in signature.parameters



def attr_parse_args():
    attr_parser = parse_args("Run and Evaluate a model attribution on a custom CSV dataset.")

    # attr_parser.add_argument("--num_labels", type=int, help="Number of labels in the dataset")

    # Add attribution arguments
    attr_parser.add_argument("--attr_method", type=str, default="lig", 
                             help="Attribution method to use")
    attr_parser.add_argument("--embeddings", type=str, default="embeddings", 
                             help="embedding layer name")
    attr_parser.add_argument("--attr_target_layer", type=str, default="embeddings", 
                             help="Target layer for attribution")

    # Load LoRA model arguments
    attr_parser.add_argument("--load_lora", action="store_true", 
                             help="Load LoRA model")
    attr_parser.set_defaults(load_lora=False)

    attr_parser.add_argument("--target_modules", type=str, nargs="+", default=None, 
                             help="Target modules to apply LoRA")
    
    attr_parser.add_argument("--output_attr_dir", type=str, default=None,)

    args = attr_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.output_attr_dir is not None:
        os.makedirs(args.output_attr_dir, exist_ok=True)

    return args


def summarize_attributions(attributions: np.ndarray):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    # compute the normalized attributions using numpy
    # attributions = attributions.sum(axis=-1).squeeze(0)
    # attributions = attributions / np.linalg.norm(attributions)
    return attributions


def construct_whole_embeddings(embeddings, input_ids, ref_input_ids, token_type_ids=None, 
                               ref_token_type_ids=None, position_ids=None, 
                               ref_position_ids=None):
    input_embeddings = embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    ref_input_embeddings = embeddings(ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids)
    
    return input_embeddings, ref_input_embeddings


def interpret_model(model, tokenizer, dataset, attr_method, target_layers=None, \
                    embedding_name=None, batch_size: int=8, debug: bool=False, label_key:str = 'label') -> List[Dict]:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, 
        collate_fn=default_data_collator
    )

    model.eval()

    def forward_func(inputs, attention_mask=None, token_type_ids=None, target=None):
        logits = model(inputs,attention_mask=attention_mask,
                       token_type_ids=token_type_ids).logits
        # print("logits shape", logits.shape)
        if target is not None:
            return logits[:, target].values
        # pred = logits.max(dim=-1).values
        # print("pred", pred)
        # return pred
        return logits
    
    def forward_func_emb(inputs_embs, attention_mask=None, token_type_ids=None, embeds=True):   
        # print("inputs_embs", inputs_embs.shape, embeds) 
        # batched the inputs_embs   
        if embeds: 
            outputs = model(inputs_embeds=inputs_embs,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        else:
            outputs = model(inputs_embs,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        logits = outputs.logits
        # access model 
        return logits
    
    if attr_method == DeepLift:
        # DeepLift needs accept model
        deep_lift_model = ModelWrapper(model)
        attr = attr_method(deep_lift_model)
    elif function_accepts_argument(attr_method, "layer"):
        print("attr_method", attr_method, "needs layer")
        # if attr_method == LayerDeepLift:
        #     # DeepLift needs accept model
        #     deep_lift_model = ModelWrapper(model)
        #     attr = attr_method(deep_lift_model, target_layers)
        # else:
        attr = attr_method(forward_func_emb, target_layers)
    else:
        attr = attr_method(forward_func)

    ref_ids = tokenizer.pad_token_id 

    attr_outputs = []

    for batch in dataloader:
        model.zero_grad()
        # create baseline input using the pad token from tokenizer
        baseline = torch.ones_like(batch["input_ids"]) * ref_ids
        baseline = baseline.to(device=DEVICE, dtype=torch.long)
        # print("baseline", baseline)

        input_ids = batch["input_ids"].to(device=DEVICE, dtype=torch.long)
        attention_mask = batch["attention_mask"].to(device=DEVICE, dtype=torch.long)
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(device=DEVICE, dtype=torch.long)
        else:
            token_type_ids = None
        
        # Forward pass
        output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if hasattr(output, "logits"):
            logits = output.logits
            preds = logits.argmax(dim=-1)
        else:
            print("output", output)
        # preds = torch.zeros(input_ids.shape[0], dtype=torch.long)


        attr_output = None
        # check if accept 3 arguments
        # if function_accepts_argument(attr.attribute, "baselines"):
        if isinstance(attr, (LayerIntegratedGradients)):
            attr_output = attr.attribute(input_ids, 
                                         additional_forward_args=(attention_mask, token_type_ids, False),
                                         baselines=baseline,
                                         target=preds,
                                         n_steps=10)
            # print(len(attr_output[0]))
            attr_output = attr_output[0].detach().cpu()
            attr_output = summarize_attributions(attr_output)
        else:
            # only working for grad and ig for now
            interpretable_embedding = configure_interpretable_embedding_layer(model,
                                                                              embedding_name)
            input_embs = interpretable_embedding.indices_to_embeddings(
                    input_ids).to(device=DEVICE, dtype=torch.float32)

            # print("attr.attribute", attr.attribute)
            if function_accepts_argument(attr.attribute, "baselines"):
                baseline_embs = interpretable_embedding.indices_to_embeddings(
                        baseline).to(device=DEVICE, dtype=torch.float32)
                print("input_embs", input_embs.shape, "baseline_embs", baseline_embs.shape)
                attr_output = attr.attribute(inputs=input_embs, 
                                             baselines=baseline_embs,
                                             additional_forward_args=(attention_mask, token_type_ids, True),
                                             target=preds)
            else:
                attr_output = attr.attribute(inputs=input_embs, 
                                             additional_forward_args=(attention_mask, token_type_ids),
                                             target=preds)
                
            # remove interpretable embedding layer
            remove_interpretable_embedding_layer(model, interpretable_embedding)
            attr_output = attr_output[0].detach().cpu()
            # check if still include the embedding dimension
            if len(attr_output.shape) > 2:
                # print(attr_output.shape, len(input_ids))
                attr_output = summarize_attributions(attr_output)
                # print(attr_output.shape, len(input_ids))

        # print(attr_output.shape, len(input_ids))
        labels = batch["labels"].detach().cpu().numpy()

        # print(attr_output, len(attr_output[0]))

        # prediction to cpus
        preds = preds.detach().cpu().numpy()

        for idx, (toks, pred) in enumerate(zip(input_ids, preds)):
            input_tokens = tokenizer.convert_ids_to_tokens(toks)
            # remove padding tokens
            input_tokens = [str(t) for t in input_tokens if not (t in tokenizer.all_special_tokens)]

            # convert pred and label index to str class
            # print(pred, len(dataset.features[label_key].names))
            # pred = str(dataset.features[label_key].names[pred])
            pred = str(pred)
            # label = str(dataset.features[label_key].names[labels[idx]])
            label = str(labels[idx])

            # print(idx, len(list(zip(input_ids, preds))), len(attr_output))
            attr_out = attr_output[idx]
            attr_out = attr_out[:len(input_tokens)]

            # print(type(attr_out), len(attr_out), attr_out.shape, len(input_tokens))

            # attr_out = summarize_attributions(attr_out)
            attr_out = attr_out.tolist()

            attr_res = {
                "tokens": input_tokens,
                "pred": pred,
                label_key: label,
                "attr": attr_out
            }
            if debug and random.random() < 0.01:
                print(attr_res)
                # pass
            attr_outputs.append(attr_res)
    
    return attr_outputs

def ferret_interpret_model(model, tokenizer, dataset, 
                           batch_size: int=8, debug: bool=False,
                           label_key: str='label',) -> List:

    ig = IntegratedGradientExplainer(model, tokenizer, multiply_by_inputs=True)
    g = GradientExplainer(model, tokenizer, multiply_by_inputs=True)
    l = LIMEExplainer(model, tokenizer)
    s = SHAPExplainer(model, tokenizer)

    bench = Benchmark(model, tokenizer,explainers=[ig, g, l, s])
    num_labels = model.config.num_labels

    eval_results = []
    for instance in dataset:
        text = tokenizer.decode(instance["input_ids"], skip_special_tokens=False)
        # print('Text:', text, 'Label:', instance[label_key], num_labels)
        instance_result = {
            "text": text,
            label_key: instance[label_key],
            "correct_results": [],
            "incorrect_results": []
        }
        # get model prediction
        # print("Instance:", instance)
        instance = default_data_collator([instance])
        inp = {
            "input_ids": instance["input_ids"].to(device=DEVICE),
            "attention_mask": instance["attention_mask"].to(device=DEVICE),
        }
        if "token_type_ids" in instance:
            inp["token_type_ids"] = instance["token_type_ids"].to(device=DEVICE)

        logits = model(**inp).logits
        # labels = instance["labels"].to(device=DEVICE)
        prediction = logits.argmax(dim=-1)

        # classes_to_eval = list(range(num_labels))

        # # Determine which classes to explain:
        # if num_labels > 5:
        #     # Select top-5 classes by using torch.topk on the logits.
        #     # Here we assume logits has shape [1, num_labels].
        #     topk = 5
        #     classes_to_eval = torch.topk(logits, k=topk, dim=-1).indices.squeeze(0).tolist()

        # Now iterate over the selected classes:
        # for class_idx in classes_to_eval:
        # # for class_idx in range(num_labels):
        #     # print('Class:', class_idx)
        class_idx = prediction.item()
        result = bench.explain(text, class_idx)
        # print("Result:", result)

        evals_result = bench.evaluate_explanations(result, class_idx)
        if random.random() < 0.5:
            print("Bench results:", evals_result)

        if instance_result[label_key] == class_idx:
            instance_result["correct_results"].append(evals_result)
        else:
            instance_result["incorrect_results"].append(evals_result)
        eval_results.append(instance_result)
    
    return eval_results


def seralize_attributions(attributions: List[Dict]):
    # check if the attributions is a list of dictionaries and each element is serializable
    for attr in attributions:
        if not isinstance(attr, dict):
            raise ValueError("Attributions should be a list of dictionaries")
        for key, value in attr.items():
            if not isinstance(value, (int, float, str, list)):
                raise ValueError("Attributions should be serializable")
    return True



def save_attributions(attributions: List[Dict], path: str):
    # check serializability of the attributions
    if seralize_attributions(attributions):
        with open(path, "w") as f:
            json.dump(attributions, f)


def main():
    args = attr_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    if args.seed is not None:
        set_seed(args.seed)

    print("start loading the dataset", "path:", args.dataset_path)
    all_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                  model_name=args.model_name, 
                                  path=args.dataset_path, split="all",
                                  label_key=args.label_key, 
                                  use_fast=args.fast_tokenizer,)
    print("finished loading the dataset")

    model_name = args.model_name

    label_key = args.label_key

    num_labels = len(all_dataset.features[label_key].names)
    if args.num_labels is not None:
        num_labels = args.num_labels

    print("start constructing the model", "model_name:", model_name)
    model = construct_model(model_name=model_name, num_labels=num_labels).to(DEVICE)
    print("finished constructing the model")

    if args.load_lora:
        model = load_lora_model(model=model, 
                                target_modules=args.target_modules, 
                                task_type=TaskType.SEQ_CLS)


    attr_method = get_attribution_method(args.attr_method)


    print(model)

    # get model layer 
    layers = args.attr_target_layer.split(",")
    print("Layers:", layers, "embedding_name:", args.embeddings)
    target_layers = []
    for layer in layers:
        # get the layer from the model
        target_layer = get_nested_attr(model, layer)
        # set the target layer required gradient
        target_layer.requires_grad = True
        print("Target layer:", target_layer)
        if target_layer is None:
            raise ValueError(f"Layer {layer} not found in the model")
        target_layers.append(target_layer)

    print(attr_method, target_layers)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=args.fast_tokenizer,)

    
    print(args.checkpoint_dir)

    # if args.checkpoint_dir is not None:
    # check if the model checkpoint exists
    expected_checkpoint = os.path.join(args.checkpoint_dir, "model.pth")
    if args.checkpoint_dir is not None and os.path.exists(expected_checkpoint):
        # load the model from the checkpoint
        state_dict = torch.load(expected_checkpoint, map_location=DEVICE)
        # debug state_dict
        model.load_state_dict(state_dict=state_dict)
    else:
        logger.info("No checkpoint is provided, zero-shot evaluation")


    # interpret model
    # attr_outs = interpret_model(model, tokenizer, all_dataset, 
    #                             attr_method, target_layers, 
    #                             embedding_name=args.embeddings, 
    #                             batch_size=args.eval_batch_size, 
    #                             debug=True)

    # for attr_out in train_attr_outs:
    #     print(attr_out)

    # create the filepath
    # filename = f"train_attributions_{args.attr_method}-layer_{args.attr_target_layer}.json"
    # attributions_path = os.path.join(args.output_attr_dir, filename)

    eval_outputs = ferret_interpret_model(model, tokenizer, all_dataset, 
                                          batch_size=args.eval_batch_size, debug=True,
                                          label_key=label_key)
    
    # get path basename from the dataset path
    dataset_basename = os.path.basename(args.dataset_path)

    # save pickle file
    filepath = os.path.join(args.output_attr_dir, f"ferret-output-{dataset_basename}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(eval_outputs, f)

    # save_attributions(attr_outs, attributions_path)
    # TODO: create top-k faithfulness evaluation by masking the tokens with the highest attributions


if __name__ == "__main__":
    main()
