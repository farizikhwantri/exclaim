import argparse

import logging

from collections import defaultdict

import json
import pandas as pd

import torch
import transformers
from accelerate import Accelerator, infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the Accelerator
accelerator = Accelerator()

sac_structure = """
An assurance case is a structured, compelling argument, supported by evidence, justifying that a system has 
some postulated properties in a specific context and environment.
Goals specify properties of the system that are to be demonstrated by the assurance case. They have to be 
specified to start development of an argument.

The Argument contains an explicit and verifiable reasoning supported by evidence which demonstrates that 
the specified goals are achieved. It is the core element of an assurance case.
The argument is to be supported by evidence. Evidence is a verifiable and auditable information in any form 
(documents, data, photos, video, statistics) which describes the system, its components, characteristics, 
properties or events including data on the system operation. To be valid, the evidence needs to be consistent 
with the reality and up to date.
In assurance case terms, an argument takes the form of a series of connected claims.
The argument starts with an overall claim about the system or object of interest, which is then decomposed into
a series of sub-claims intended to support the overall claim. Depending on the complexity of the object or system 
being argued for, these claims often need to be decomposed into further supporting sub-claims until eventually 
the claims are supported by evidence justifying those claims.
"""

# sac_cae_json_schema={"MainClaim":{"description":"The primary claim that is being argued or discussed.","SubClaims":[{"SubClaim1":{"description":"The first sub-claim supporting the main claim.","ArgumentClaims":[{"ArgumentClaim":{"description":"The argument that supports SubClaim1.","ArgumentSubClaims":[{"ArgumentSubClaim1":{"description":"The first sub-claim within the argument for SubClaim1.","Evidences":[{"Evidence1":{"description":"Description of Evidence 1 supporting ArgumentSubClaim1.","type":"","source":""}},{"Evidence2":{"description":"Description of Evidence 2 supporting ArgumentSubClaim1.","type":"","source":""}}]}},{"ArgumentSubClaim2":{"description":"The second sub-claim within the argument for SubClaim1.","Evidences":[{"Evidence1":{"description":"Description of Evidence 1 supporting ArgumentSubClaim2.","type":"","source":""}},{"Evidence2":{"description":"Description of Evidence 2 supporting ArgumentSubClaim2.","type":"","source":""}}]}}]}}]}},{"SubClaim2":{"description":"The second sub-claim supporting the main claim.","ArgumentClaims":[{"ArgumentClaim":{"description":"The argument that supports SubClaim2.","ArgumentSubClaims":[{"ArgumentSubClaim1":{"description":"The first sub-claim within the argument for SubClaim2.","Evidences":[{"Evidence1":{"description":"Description of Evidence 1 supporting ArgumentSubClaim1.","type":"","source":""}},{"Evidence2":{"description":"Description of Evidence 2 supporting ArgumentSubClaim1.","type":"","source":""}},{"Evidence3":{"description":"Description of Evidence 3 supporting ArgumentSubClaim1.","type":"","source":""}},{"Evidence4":{"description":"Description of Evidence 4 supporting ArgumentSubClaim1.","type":"","source":""}}]}},{"ArgumentSubClaim2":{"description":"The second sub-claim within the argument for SubClaim2.","Evidences":[{"Evidence1":{"description":"Description of Evidence 1 supporting ArgumentSubClaim2.","type":"","source":""}},{"Evidence2":{"description":"Description of Evidence 2 supporting ArgumentSubClaim2.","type":"","source":""}}]}}]}}]}}]}}

sac_cae_json_schema = {
    "MainClaim": {
        "description": "The primary claim that is being argued or discussed.",
        "SubClaims": [
            {
                "SubClaim1": {
                    "description": "The first sub-claim supporting the main claim.",
                    "ArgumentClaims": [
                        {
                            "ArgumentClaim": {
                                "description": "The argument that supports SubClaim1.",
                                "ArgumentSubClaims": [
                                    {
                                        "ArgumentSubClaim1": {
                                            "description": "The first sub-claim within the argument for SubClaim1.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    {
                                        "ArgumentSubClaim2": {
                                            "description": "The second sub-claim within the argument for SubClaim1.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "SubClaim2": {
                    "description": "The second sub-claim supporting the main claim.",
                    "ArgumentClaims": [
                        {
                            "ArgumentClaim": {
                                "description": "The argument that supports SubClaim2.",
                                "ArgumentSubClaims": [
                                    {
                                        "ArgumentSubClaim1": {
                                            "description": "The first sub-claim within the argument for SubClaim2.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence3": {
                                                        "description": "Description of Evidence 3 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence4": {
                                                        "description": "Description of Evidence 4 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    {
                                        "ArgumentSubClaim2": {
                                            "description": "The second sub-claim within the argument for SubClaim2.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}

cae_simple_json_schema = {
    "MainClaim": {
        "description": "The primary claim that is being argued or discussed.",
        "Claims": [
            {
                "Claim1": {
                    "description": "The first sub-claim supporting the main claim.",
                    "Arguments": [
                        {
                            "Argument1": {
                                "description": "The argument that supports SubClaim1.",
                                "Claims": [
                                    {
                                        "Claim1": {
                                            "description": "The first sub-claim within the argument for SubClaim1.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    {
                                        "Claim2": {
                                            "description": "The second sub-claim within the argument for SubClaim1.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "Claim2": {
                    "description": "The second sub-claim supporting the main claim.",
                    "Arguments": [
                        {
                            "Argument1": {
                                "description": "The argument that supports SubClaim2.",
                                "Claims": [
                                    {
                                        "Claim1": {
                                            "description": "The first sub-claim within the argument for SubClaim2.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence3": {
                                                        "description": "Description of Evidence 3 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence4": {
                                                        "description": "Description of Evidence 4 supporting ArgumentSubClaim1.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    {
                                        "Claim2": {
                                            "description": "The second sub-claim within the argument for SubClaim2.",
                                            "Evidences": [
                                                {
                                                    "Evidence1": {
                                                        "description": "Description of Evidence 1 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                },
                                                {
                                                    "Evidence2": {
                                                        "description": "Description of Evidence 2 supporting ArgumentSubClaim2.",
                                                        "type": "",
                                                        "source": "",
                                                        # "date": ""
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
}

no_sys_role_list = ["gemma", "olmo", "Saul"]

# Filtering requirements

def construct_prompt(requirement_id, requirement_name, requirement_description, requirement_rationale, format):
    template = """

    The requirement is {requirement_id}: {requirement_name}

    # Requirement Description
    {requirement_description}
    # Rationale and supplemental guidance
    {requirement_rationale}


    Give the output in {format}.
    """.format(requirement_id=requirement_id, requirement_name=requirement_name, 
               requirement_description=requirement_description, 
               requirement_rationale=requirement_rationale, format=format)
    return template

# def get_max_memory():
#     """Get the maximum memory available for the current GPU for loading models."""
#     free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
#     print(f"Free memory: {free_in_GB}GB")
#     max_memory = f'{free_in_GB-6}GB'
#     n_gpus = torch.cuda.device_count()
#     max_memory = {i: max_memory for i in range(n_gpus)}
#     return max_memory

def get_max_memory():
    """Get the maximum memory available for loading models, checking CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        # CUDA is available
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
        max_memory = f'{free_in_GB-6}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        return {"CUDA": max_memory}
    
    elif torch.backends.mps.is_available():
        # MPS is available (Apple Silicon)
        # Currently, MPS does not have a direct method to get free memory
        print("MPS is available. Please provide the maximum memory available for MPS.")
        memory_in_GB = 16  # Example: replace with an actual estimation method if possible
        max_memory = f'{memory_in_GB-2}GB'
        return {"mps": max_memory}
    
    else:
        # Fallback for CPU-only systems
        import psutil
        memory_in_GB = int(psutil.virtual_memory().total / 1024**3)
        max_memory = f'{memory_in_GB-2}GB'
        return {"CPU": max_memory}

def main(args):
    # Initialize the model and tokenizer
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info(f"Using device: {accelerator.device}")
    logger.info('args: %s', args)

    def_model_name = "microsoft/phi-4"
    model_name = args.model_name if args.model_name is not None else def_model_name

    # sys_msg = """You are a security assurance engineer. Your duty is to produce a thorough Security Assurance Case
    # from IEC 62443 requirements. {sac_structure}""".format(sac_structure=sac_structure)

    # def_sys_msg = """You are a legal expert in privacy and security issue. Your duty is to produce a thorough Data Processing Agreement Assurance Case
    # from General Data Protection Regulation legal requirements. {sac_structure}""".format(sac_structure=sac_structure)

    def_sys_msg = """You are a legal expert in privacy and security issue. Your duty is to produce a thorough Security Assurance Case
    from Cybersecurity Resilience Act requirements. {sac_structure}""".format(sac_structure=sac_structure)

    # read arguments system message from args of json file
    if args.system_msg is not None:
        with open(args.system_msg, 'r') as f:
            # write a function to read the system message according to file type e.g. json, txt, etc.
            if args.system_msg.endswith('.json'):
                sys_msg = json.load(f).get('system_message', def_sys_msg)['content']
                sys_msg = sys_msg.strip() if sys_msg else def_sys_msg
            elif args.system_msg.endswith('.txt'):
                sys_msg = f.read().strip()
    else:
        # use default system message
        sys_msg = def_sys_msg
    
    # If the txt file doesn't contain a system message, use the default
    if not sys_msg:
        sys_msg = def_sys_msg

    # sys_msg = args.system_msg if args.system_msg is not None else def_sys_msg

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


    # Get optimal memory split
    # device_map = infer_auto_device_map(model_name, dtype=torch.bfloat16)

    if torch.backends.mps.is_available():
        dtype = torch.float16  # or torch.float32 for full precision
    else:
        dtype = torch.bfloat16
    

    # Load the model with the inferred device map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        # device_map=device_map,
        # torch_dtype=torch.bfloat16,
        torch_dtype=dtype,
        trust_remote_code=True,
        max_memory=get_max_memory(),

    )

    model.eval()

    # Prepare the model and tokenizer with the accelerator
    # model, tokenizer = accelerator.prepare(model, tokenizer)

    # Create the pipeline
    # pipe = transformers.pipeline(
    #     "text-generation",
    #     model=model_name,
    #     model_kwargs={"load_in_8bit": True},
    #     device_map='auto'
    # )

    results_dict = defaultdict(dict)
    ncalls = args.ncalls

    # load the input file
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith('.json'):
        df = pd.read_json(args.input_file)
    else:
        raise ValueError("Input file should be either csv or json")
    
    # count failed json
    failed_json = 0

    # format_prompt = f"such a way that the data matches the follwing schema: {sac_cae_json_schema}. Give just the json without any explanation. \
    #     Follow the JSON schema key convention naming. Do not copy the schema or obvious example. Print the json in a single line, do not write a new line."

    format_prompt = f"such a way that the data matches the follwing schema: {sac_cae_json_schema}. Give just the json without any explanation."

    if args.user_prompt is not None:
        # read the user prompt from the file
        with open(args.user_prompt, 'r') as f:
            user_prompt = f.read().strip()
        format_prompt = f"{user_prompt}"
    
    # check if the model name contains have no system role chat template
    sys_role_flag = True
    if any(role in model_name for role in no_sys_role_list):
        sys_role_flag = False

    for i, row in df.iterrows():
        req_id = row['requirement_id']
        req_name = row['requirement_name']
        req_desc = row['requirement_description']
        # req_rationale = row['requirement_rationale']
        req_rationale = row['manual_rationale'] if 'manual_rationale' in row else row['requirement_rationale']

        results_dict[row['requirement_id']] = defaultdict(list)

        prompt = construct_prompt(req_id, req_name, req_desc, req_rationale, 
                                  format_prompt)

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
        if not sys_role_flag:
            # contate system message to the prompt
            prompt = sys_msg + ' ' + prompt
            messages = [
                {"role": "user", "content": prompt},
            ]

        # tokenize the messages
        # make the messages a list of strings
        # messages_tokens = [msg['content'] for msg in messages]
        # inputs = tokenizer(messages_tokens, return_tensors="pt", padding=True, truncation=True)

        # print input length
        # print(inputs['input_ids'].shape)
        print(messages)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print(f"Processing requirement {req_id}")

        for i in range(ncalls):
            # outputs = pipe(messages, max_new_tokens=10000)
            # print(outputs[0]['generated_text'])

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10000,
                # no_repeat_ngram_size=2,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


            # how to retrieve the generated text
            
            logger.info(f"Generated text for requirement {req_id} in call {i}: {response}")

            if i % ncalls == 0:
                print(f"Processed {i} requirements")

            print(f"Generated text for requirement {req_id} in call {i}: {response}")

            # save the results to a json file
            model_base_name = model_name.split('/')[-1]
            with open(f"{args.output_dir}/{model_base_name}_{req_id}_{i}.out", "w") as file:
                json.dump(response, file, indent=4)


    
    # # save the results to a json file
    # with open(f"{args.output_dir}/{model_name}.json", "w") as file:
    #     json.dump(results_dict, file, indent=4)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate claim_argument_evidence structure from requirements")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON or csv file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--model_name", type=str, default=None, help="Model name to use for generation")
    parser.add_argument("--ncalls", type=int, default=5, help="Number of calls to the model")
    parser.add_argument("--system_msg", type=str, default=None, help="System message to use for generation")
    parser.add_argument("--user_prompt", type=str, default=None, help="User prompt to use for generation")
    args = parser.parse_args()
    main(args)
    