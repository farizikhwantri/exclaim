import copy
import numpy as np
from statistics import mean

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inseq import load_model

# Import the evaluation classes
from explainer_llm import LLMAttribution
from faithfulness_lm import AOPC_Comprehensiveness_LLM_Evaluation, AOPC_Sufficiency_LLM_Evaluation
from ferret.explainers.explanation import Explanation

# Main demonstration
def eval_without_explainer_wrapper():
    model_name = "gpt2"
    # Load causal model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # For causal models, set pad token to EOS if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    # Load inseq attribution model (using Integrated Gradients here)
    inseq_model = load_model(model_name, attribution_method="integrated_gradients")

    # Define the input text
    text = "The Eiffel Tower is located in Paris, which is the capital of France."
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    length = inputs["input_ids"].shape[1]
    
    # Obtain attribution using inseq (the API returns an object with a list of step attributions)
    attribution = inseq_model.attribute(
        text,
        attribution_args={"n_steps": 20},
        step_scores=["logit"],
        include_eos_baseline=True,
        output_step_attributions=True,
    )

    # Create our helper object
    # helper = LLMHelper(model, tokenizer)
    
    # Initialize evaluation classes and attach required components.
    aopc_compr_eval = AOPC_Comprehensiveness_LLM_Evaluation(model, tokenizer)
    # aopc_compr_eval.helper = helper
    # aopc_compr_eval.tokenizer = tokenizer

    aopc_suff_eval = AOPC_Sufficiency_LLM_Evaluation(model, tokenizer)
    # aopc_suff_eval.helper = helper
    # aopc_suff_eval.tokenizer = tokenizer

    # Define removal arguments (for demonstration, use threshold based approach and masking)
    removal_args = {
        # "based_on": "th",       # using a threshold method
        # "thresholds": [0.5],    # a list of threshold values
        "remove_tokens": False  # set False to use masking (mask token) instead of deletion
    }
    evaluation_args = {"removal_args": removal_args, "remove_first_last": False, "only_pos": True}

    # scores = attribution.step_attributions[0].target_attributions
    # # take norm of the scores
    # scores = np.linalg.norm(scores, axis=-1)
    # scores = scores / np.sum(scores)  # normalize the scores
    # print("Attribution scores:", scores, scores.shape)
    # # squeeze the scores to match the input length
    # scores = np.squeeze(scores)
    # print("Squeezed attribution scores:", scores, scores.shape)

    # # Define the target token to monitor (e.g., "France")
    # # target_token = "France"
    # print("Attribution step attributions:", attribution)
    # target_token = attribution.step_attributions[0].target # first token in the list
    # print("Target token:", target_token)
    # target_token = target_token[0][0].token.replace("Ġ", "")  # remove the special character

    # print("Target token:", target_token)
    
    # # Construct a simple Explanation object using the inseq output.
    # # Provide the additional required arguments "explainer" and "target".
    # explanation = Explanation(
    #     text=text,
    #     scores=scores,  # soft scores from inseq
    #     tokens=attribution.step_attributions[0].prefix[0],            # tokens from the prefix
    #     explainer="inseq_integrated_gradients",                       # identifier for the explainer
    #     target=target_token                                           # target that we care about
    # )

    # # Compute AOPC Comprehensiveness evaluation:
    # compr_evaluation = aopc_compr_eval.compute_evaluation(explanation, target_token, token_position=None, **evaluation_args)
    # print("AOPC Comprehensiveness Evaluation:", compr_evaluation)

    # # Compute AOPC Sufficiency evaluation:
    # suff_evaluation = aopc_suff_eval.compute_evaluation(explanation, target_token, token_position=None, **evaluation_args)
    # print("AOPC Sufficiency Evaluation:", suff_evaluation)

    for step in attribution.step_attributions:
        print("Step attributions:", step)
        
        scores = step.target_attributions
        # take norm of the scores
        scores = np.linalg.norm(scores, axis=-1)
        scores = scores / np.sum(scores)
        print("Attribution scores:", scores, scores.shape)
        # squeeze the scores to match the input length
        scores = np.squeeze(scores)

        print("Squeezed attribution scores:", scores, scores.shape)

        # get the target token based on the step
        target_token_step = step.target[0][0].token.replace("Ġ", "")
        print("Target token:", target_token_step)
        # Construct a simple Explanation object using the inseq output.
        # Provide the additional required arguments "explainer" and "target".
        tokenized = step.prefix[0]
        tokenized_id = [token.id for token in tokenized]
        text = tokenizer.decode(tokenized_id, skip_special_tokens=True)
        explanation = Explanation(
            text=text,
            scores=scores,  # soft scores from inseq
            tokens=step.prefix[0],  # tokens from the prefix
            explainer="inseq_integrated_gradients",  # identifier for the explainer
            target=target_token_step  # target that we care about
        )

        # Compute AOPC Comprehensiveness evaluation:
        compr_evaluation = aopc_compr_eval.compute_evaluation(explanation, target_token_step, token_position=None, **evaluation_args)
        print("AOPC Comprehensiveness Evaluation:", compr_evaluation)
        # Compute AOPC Sufficiency evaluation:
        suff_evaluation = aopc_suff_eval.compute_evaluation(explanation, target_token_step, token_position=None, **evaluation_args)
        print("AOPC Sufficiency Evaluation:", suff_evaluation)

# Example usage:
def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # For causal models, set pad token to EOS if necessary.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    attribution_method = "input_x_gradient"  # or "integrated_gradients"
    
    # Create LLMAttribution explainer.
    explainer = LLMAttribution(model, tokenizer, attribution_method=attribution_method)
    
    # Define input text.
    text = "The Eiffel Tower is located in Paris, which is the capital of France."
    # Compute feature importance. The wrapper returns a list of Explanation objects (one per step).
    explanations = explainer.compute_feature_importance(text, target=1, 
                                                        step_scores=["logit"],
                                                        include_eos_baseline=True, 
                                                        output_step_attributions=True)
    
    # compute comprehensiveness and sufficiency scores
    aopc_compr_eval = AOPC_Comprehensiveness_LLM_Evaluation(model, tokenizer)
    aopc_suff_eval = AOPC_Sufficiency_LLM_Evaluation(model, tokenizer)

    # Define removal arguments (for demonstration, use threshold based approach and masking)
    removal_args = {
        # "based_on": "th",       # using a threshold method
        # "thresholds": [0.5],    # a list of threshold values
        "remove_tokens": False  # set False to use masking (mask token) instead of deletion
    }
    evaluation_args = {"removal_args": removal_args, "remove_first_last": False, "only_pos": True}
    
    # Iterate over the explanations and print details.
    for i, expl in enumerate(explanations):
        print(f"--- Explanation Step {i+1} ---")
        print("Text:", expl.text)
        print("Tokens:", [t for t in expl.tokens])
        print("Scores shape:", np.array(expl.scores))
        print("Explainer:", expl.explainer)
        print("Target token:", expl.target)
        print("\n")

        # Compute comprehensiveness and sufficiency scores
        comprehensiveness_score = aopc_compr_eval.compute_evaluation(expl, target_token=expl.target, **evaluation_args)
        sufficiency_score = aopc_suff_eval.compute_evaluation(expl, target_token=expl.target, **evaluation_args)

        print("Comprehensiveness Score:", comprehensiveness_score)
        print("Sufficiency Score:", sufficiency_score)


if __name__ == "__main__":
    main()