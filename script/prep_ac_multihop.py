import random
import argparse
import json

import re
import math

# from collections import OrderedDict
import os

# import itertools
import pandas as pd

def generate_threehop_inferences(claims):
    inferences = {"one_hop": [], "two_hop": [], "three_hop": []}

    # Helper function to get the parent description
    def get_parent_description(claim_id):
        return claims[claim_id]["parent_description"] if "parent_description" in claims[claim_id] else None

    hop_statistic = {"one_hop": 0, "two_hop": 0, "three_hop": 0}

    # Generate one-hop inferences
    for claim_id, claim in claims.items():
        parent_description = get_parent_description(claim_id)
        if parent_description:
            inferences["one_hop"].append({
                "premise": parent_description,
                "hypothesis": claim["description"],
                "meta": {
                    "current_node": claim_id,
                    "parent_node": claim["parent"]
                }
            })
            hop_statistic["one_hop"] += 1

    # Generate two-hop inferences
    for claim_id, claim in claims.items():
        parent_description = get_parent_description(claim_id)
        if parent_description:
            grandparent_description = get_parent_description(claim["parent"])
            if grandparent_description:
                inferences["two_hop"].append({
                    "premise": grandparent_description,
                    "hypothesis": claim["description"],
                    "meta": {
                        "current_node": claim_id,
                        "parent_node": claim["parent"],
                        "grandparent_node": claims[claim["parent"]]["parent"],
                        "one_hop": {
                            "premise": parent_description,
                            "hypothesis": claim["description"]
                        }
                    }
                })
                hop_statistic["two_hop"] += 1
                

    # Generate three-hop inferences
    for claim_id, claim in claims.items():
        parent_description = get_parent_description(claim_id)
        if parent_description:
            grandparent_description = get_parent_description(claim["parent"])
            if grandparent_description:
                great_grandparent_description = get_parent_description(claim["parent"])
                if great_grandparent_description:
                    inferences["three_hop"].append({
                        "premise": great_grandparent_description,
                        "hypothesis": claim["description"],
                        "meta": {
                            "current_node": claim_id,
                            "parent_node": claim["parent"],
                            "grandparent_node": claims[claim["parent"]]["parent"],
                            "great_grandparent_node": claims[claims[claim["parent"]]["parent"]]["parent"],
                            "one_hop": {
                                "premise": parent_description,
                                "hypothesis": claim["description"]
                            },
                            "two_hop": {
                                "premise": grandparent_description,
                                "hypothesis": claim["description"]
                            }
                        }
                    })
                    hop_statistic["three_hop"] += 1            

    # print(hop_statistic)
    
    return inferences, hop_statistic


def get_level(claim_id, claims):
    level = 0
    current_id = claim_id
    while current_id in claims and claims[current_id].get("parent") != "root":
        current_id = claims[current_id]["parent"]
        level += 1
    return level

def group_by_level(claims):
    levels = {}
    for claim_id, claim in claims.items():
        level = get_level(claim_id, claims)
        if level not in levels:
            levels[level] = []
        levels[level].append(claim_id)
    return levels

def create_negative_samples(claims):
    levels = group_by_level(claims)
    negative_samples = []

    for level, nodes in levels.items():
        for node in nodes:
            non_connected_nodes = [n for n in nodes if n != node and \
                                    claims[n].get("parent") != claims[node].get("parent")]
            if non_connected_nodes:
                negative_node = random.choice(non_connected_nodes)
                node_description = claims[node]["description"]
                negative_node_description = claims[negative_node]["description"]
                # check if the pair is already in the negative samples
                if not any(sample["meta"]["current_node"] == node and \
                           sample["meta"]["negative_node"] == negative_node \
                            for sample in negative_samples) and \
                            node_description != negative_node_description:
                    negative_samples.append({
                        "premise": claims[node]["description"],
                        "hypothesis": claims[negative_node]["description"],
                        "meta": {
                            "current_node": node,
                            "negative_node": negative_node,
                            "parent_node": claims[node]["parent"],
                            # "root": root
                            "hop": -1,
                            # "target": "negative"
                        },
                        'negative_node': negative_node
                    })

    return negative_samples


def get_root(claim_id, claims):
    current_id = claim_id
    while current_id in claims and claims[current_id].get("parent") != "root":
        current_id = claims[current_id]["parent"]
    return current_id

def create_negative_samples_non_shared_root(claims):
    roots = {}
    for claim_id, claim in claims.items():
        root = get_root(claim_id, claims)
        if root not in roots:
            roots[root] = []
        roots[root].append(claim_id)

    negative_samples = []

    for root, nodes in roots.items():
        for node in nodes:
            non_connected_nodes = [n for r, ns in roots.items() if r != root for n in ns]
            if non_connected_nodes:
                negative_node = random.choice(non_connected_nodes)
                # check if the pair is already in the negative samples
                if not any(sample["meta"]["current_node"] == node and sample["meta"]["negative_node"] \
                           == negative_node for sample in negative_samples):
                    negative_samples.append({
                        "premise": claims[node]["description"],
                        "hypothesis": claims[negative_node]["description"],
                        "meta": {
                            "current_node": node,
                            "negative_node": negative_node,
                            "parent_node": claims[node]["parent"],
                            # "root": root
                            "hop": -1,
                            "target": "negative" # set same as hop
                        }
                    })

    return negative_samples


def generate_n_hop_inferences(claims):
    # Recursive function to get the parent description of a claim until Main Claim
    def get_parent_description(claim_id):
        # check if the claim id is exist in the claims
        if claim_id not in claims:
            return None
        return claims[claim_id]["parent_description"] if "parent_description" in claims[claim_id] else None
    
    n = 5

    inferences = {f"{i}_hop": [] for i in range(1, n+1)}
    hop_statistic = {f"{i}_hop": 0 for i in range(1, n+1)}

    # recursive function to trace from leaf (evidence) to root (main claim)
    def trace_to_main_claim(claim, claim_id, parent, trace, local_inferences, local_hop_statistic, hop):
        assert len(trace) == hop - 1
        # base case if the parent is root
        if "MainClaim" in parent:
            # add the main claim to the leaf 
            local_inferences[f"{hop}_hop"].append({
                "premise": claims[parent]["description"],
                "hypothesis": claim["description"],
                "meta": {
                    "parent_node": parent,
                    "current_node": claim_id,
                    "hop": hop,
                    "target": f"{hop}_hop"
                },
                'trace': trace.copy()
            })
            local_hop_statistic[f"{hop}_hop"] += 1
            return
        # get the parent description
        parent_description = get_parent_description(claim_id)
        if parent_description:
            local_inferences[f"{hop}_hop"].append({
                "premise": claims[parent]["description"],
                "hypothesis": claim["description"],
                "meta": {
                    "parent_node": parent,
                    "current_node": claim_id,
                    "hop": hop, 
                    "target": f"{hop}_hop"
                },
                'trace': trace.copy()
            })
            local_hop_statistic[f"{hop}_hop"] += 1
            # trace[parent] = claims[parent]["description"]
            trace.append({parent: claims[parent]["description"]})
            # trace[parent] = parent_description
            # now update claim and claim_id into the parent
            if "parent" in claims[parent]:
                grandparent = claims[parent]["parent"]
                trace_to_main_claim(claim, claim_id, grandparent, trace, local_inferences, local_hop_statistic, hop+1)
            else:
                return
        else:
            return
    
    # trace from all claims-argument-evidence nodes to main claim
    for claim_id, claim in claims.items():
        # check if the claim contain evidence type
        # if "evidence_type" in claim and claim["evidence_type"] == "Evidence":
        #     claim["description"] = f"{claim["evidence_type"]} show that {claim["description"]}"
        local_inferences = {f"{i}_hop": [] for i in range(1, n+1)}
        local_hop_statistic = {f"{i}_hop": 0 for i in range(1, n+1)}
        parent_description = get_parent_description(claim_id)
        if parent_description:
            # trace = OrderedDict()
            trace = []
            # trace from parent to main claim
            trace_to_main_claim(claim, claim_id, claim['parent'], trace, local_inferences, local_hop_statistic, 1)

        # print(local_hop_statistic)

        # update the inferences and hop_statistic
        for key, value in local_inferences.items():
            inferences[key].extend(value)
        
        for key, value in local_hop_statistic.items():
            hop_statistic[key] += value
    
    return inferences, hop_statistic


def assert_inference_structure(inferences, claims):
    # assert that each premise and hypothesis {i}_hop is inferences have correct id
    for key, value in inferences.items():
        for instance in value:
            curr_id = instance['meta']['current_node']
            parent_id = instance['meta']['parent_node']
            curr_text = claims[curr_id]['description']
            parent_text = claims[parent_id]['description']
            assert instance['premise'] == parent_text, \
                f"premise: {instance['premise']} != {parent_text}, {parent_id}"
            assert instance['hypothesis'] == curr_text, \
                f"hypothesis: {instance['hypothesis']} != {curr_text}"

            # assert the trace is correct
            assert len(instance['trace']) == instance['meta']['hop'] - 1
            trace = instance['trace']
            # for trace_id, trace_text in trace.items():
            for trace_dict in trace:
                trace_id = list(trace_dict.keys())[0]
                trace_text = list(trace_dict.values())[0]
                assert claims[trace_id]['description'] == trace_text, \
                    f"trace: {claims[trace_id]['description']} != {trace_text}, {trace_id}"



def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_numeric_suffix(s):
    # Use regex to match the non-numeric part and the numeric suffix
    match = re.match(r'(\D*)(\d*)$', s)
    
    if match:
        non_numeric = match.group(1)
        numeric = match.group(2)
        
        # Convert the numeric part to an integer if it is not empty, otherwise return nan
        if numeric:
            numeric = int(numeric)
        else:
            numeric = math.nan
    
    return (non_numeric, numeric)


def dict_to_csv(data, with_trace=False):
    """flatten list of requirement dict contain list of instances hop
    to list of dict with requirement, model_name, one_hop, two_hop, three_hop
    example:
    {
        "requirement": "CR1.7",
        "model_name": "chatGPT4o",
        "generation": "1",
        "docname": "iec62443_4.2",
        "num_nodes": 19,
        "num_edges": 19,
        "parent_child": "{'MainClaim-0': ['SubClaim1-3', 'SubClaim2-35'], 
                          'SubClaim1-3': ['ArgumentClaim-6'], 
                          'ArgumentClaim-6': ['ArgumentSubClaim1-9', 'ArgumentSubClaim2-22'], 
                          'ArgumentSubClaim1-9': ['Evidence1-12', 'Evidence2-17'], 
                          'Evidence1-12': [], 'Evidence2-17': [], 
                          'ArgumentSubClaim2-22': ['Evidence1-25', 'Evidence2-30'], 
                          'Evidence1-25': [], 'Evidence2-30': [], 
                          'SubClaim2-35': ['ArgumentClaim-38'], 
                          'ArgumentClaim-38': ['ArgumentSubClaim1-41', 'ArgumentSubClaim2-64'], 
                          'ArgumentSubClaim1-41': ['Evidence1-44', 'Evidence2-49', 'Evidence3-54', 'Evidence4-59'], 
                          'Evidence1-44': [], 'Evidence2-49': [], 'Evidence3-54': [], 'Evidence4-59': [], 
                          'ArgumentSubClaim2-64': ['Evidence1-67', 'Evidence2-72'], 'Evidence1-67': [], 'Evidence2-72': []
                        }",
        "1_hop": [
            {
                "premise": "The system enforces configurable password strength in compliance with internationally recognized guidelines.",
                "hypothesis": "The system provides the ability to enforce password strength through configurable policies.",
                "meta": {
                    "current_node": "SubClaim1-3",
                    "parent_node": "MainClaim-0",
                    "hop": 1
                },
            }, ...
            ...
        ],
    }
    """
    new_data = []
    for instance in data:
        # print(instance)
        req = instance['requirement']
        model_name = instance['model_name']
        docname = instance['docname']
        # parent_child = instance['parent_child']
        # get the hop inferences by iterating the hops that contains list as value
        for key, value in instance.items():
            if isinstance(value, list) and 'hop' in key:
                for hop_instance in value:
                    new_instance = {}
                    new_instance['requirement'] = req
                    new_instance['model_name'] = model_name
                    new_instance['docname'] = docname
                    # new_instance['parent_child'] = parent_child
                    new_instance['premise'] = str(hop_instance['premise'])
                    new_instance['hypothesis'] = str(hop_instance['hypothesis'])
                    # new_instance['meta'] = hop_instance['meta']
                    # unroll meta
                    for meta_key, meta_value in hop_instance['meta'].items():
                        new_instance[meta_key] = meta_value

                    # print(new_instance.keys())

                    if 'trace' in hop_instance and with_trace:
                        for i, trace_dict in enumerate(hop_instance['trace']):
                            for trace_key, trace_value in trace_dict.items():
                                # check if there is a . in the new_instance premise last character
                                # if new_instance['premise'][-1] == '.':
                                #     new_instance['premise'] += f" {trace_value}"
                                # else:
                                #     new_instance['premise'] += f". {trace_value}"
                                new_instance['premise'] += f"|| {trace_value}"

                    new_instance['label'] = 'entailment'
                    new_data.append(new_instance)
        
        # get the negative samples
        # for negative_instance in instance['negative_samples?']:
        #     new_instance = {}
        #     new_instance['requirement'] = req
        #     new_instance['model_name'] = model_name
        #     new_instance['docname'] = docname
        #     # new_instance['parent_child'] = parent_child
        #     new_instance['premise'] = str(negative_instance['premise'])
        #     new_instance['hypothesis'] = str(negative_instance['hypothesis'])
        #     # new_instance['meta'] = negative_instance['meta']
        #     # unroll meta
        #     for meta_key, meta_value in negative_instance['meta'].items():
        #         new_instance[meta_key] = meta_value

        #     # new_instance['label'] = 'not_entailment'
        #     new_instance['label'] = 'neutral'
        #     new_data.append(new_instance)

    # negative samples from different requirement 
    negative_data = [] 
    for instance in new_data:
        instance_requirement = instance['requirement']
        # get the basename without number
        instance_type = instance['current_node'].split('-')[0]
        instance_type = extract_numeric_suffix(instance_type)[0]
        # sample new_data, select 10% of the data
        new_data_sample = random.sample(new_data, math.ceil(len(new_data)*0.1))
        for negative_instance in new_data_sample:
            # pick negative samples from different docname with the same level
            negative_instance_type = negative_instance['current_node'].split('-')[0]
            negative_instance_type = extract_numeric_suffix(negative_instance_type)[0]
            if random.random() < 0.001 and (negative_instance['requirement'] != instance_requirement) and negative_instance_type == instance_type:
                # print(instance_type, negative_instance_type)
                new_instance = {}
                new_instance['requirement'] = instance['requirement']
                new_instance['model_name'] = instance['model_name']
                new_instance['docname'] = instance['docname']
                new_instance['premise'] = str(negative_instance['premise'])
                new_instance['hypothesis'] = str(instance['hypothesis'])
                new_instance['current_node'] = instance['current_node']
                new_instance['parent_node'] = negative_instance['current_node']
                new_instance['hop'] = -1
                new_instance['target'] = 'negative'
                # new_instance['target'] = 'negative'
                new_instance['label'] = 'not_entailment'
                negative_data.append(new_instance)
    
    new_data.extend(negative_data)

    
    print(len(new_data))
    # new_data to df csv
    new_data = pd.DataFrame(new_data)
    # fix premise and hypothesis to string format in pandas column
    new_data['premise'] = new_data['premise'].astype(str)
    new_data['hypothesis'] = new_data['hypothesis'].astype(str)
    return new_data


def group_by_model(data, outname=''):
    # group by model name
    modelname_dict = {}
    # for item in new_data:
    for item in data:
        modelname = item['model_name']
        if modelname not in modelname_dict:
            modelname_dict[modelname] = []
        modelname_dict[modelname].append(item)

    # model_name pair into train and test
    # eg: train ChatGPT-4o, test: Qwen2.5
    # make a pair of model_name
    # model_names = list(modelname_dict.keys())

    # save the modelname_dict to json for each model name
    for model_name, items in modelname_dict.items():
        with open(args.output_dir + f"/{outname}{model_name}.json", 'w') as f:
            json.dump(items, f, indent=4)
        
        # convert the items into csv
        model_data = dict_to_csv(items, with_trace=args.with_trace)
        # save the data to tsv with tab separator
        model_data.to_csv(args.output_dir + f"/{outname}{model_name}.csv", index=False)
        # model_data.to_csv(args.output_dir + f"/{model_name}.csv", index=False, sep='\t')


def main(args):
    data = read_json(args.json_path)
    print(len(data))
    # generate one-hop, two-hop and three-hop inferences for each claim

    new_data = []
    # hop_stats = {"one_hop": 0, "two_hop": 0, "three_hop": 0}
    hop_stats = {f"{i}_hop": 0 for i in range(1, 6)}
    for instance in data:
        # filter claim only that has dict value in elem
        claims = {}
        meta = {}
        for key, value in instance.items():
            if isinstance(value, dict):
                claims[key] = value
            else:
                meta[key] = value
        # print(claims)
        # inferences, hop_stat = generate_threehop_inferences(claims)

        inferences, hop_stat = generate_n_hop_inferences(claims)
        
        # assert the inferences structure
        assert_inference_structure(inferences, claims)

        negative_samples = create_negative_samples(claims)
        # negative_samples = create_negative_samples_non_shared_root(claims)
        
        print(hop_stat)

        # hop_stats["one_hop"] += hop_stat["one_hop"]
        # hop_stats["two_hop"] += hop_stat["two_hop"]
        # hop_stats["three_hop"] += hop_stat["three_hop"]
        # update the hop_stats
        for key, value in hop_stat.items():
            hop_stats[key] += value

        new_instances = {}
        new_instances.update(meta)
        new_instances.update(inferences)
        new_instances['negative_samples?'] = negative_samples
        new_data.append(new_instances)

    print(hop_stats)


    # group by docname
    docname_dict = {}
    for item in new_data:
        docname = item['docname']
        if docname not in docname_dict:
            docname_dict[docname] = []
        docname_dict[docname].append(item)

    print(docname_dict.keys())

    # for each docname, group the data into requirement
    for docname, items in docname_dict.items():
        req_dict = {}
        for item in items:
            req = item['requirement']
            if req not in req_dict:
                req_dict[req] = []
            req_dict[req].append(item)
        docname_dict[docname] = req_dict

    
    train_data_docname = []
    test_data_docname = []
    # split into train and test for each docname 
    for docname, items in docname_dict.items():
        # get the requirement
        reqs = list(items.keys())
        # split the requirement into train and test
        train_reqs = reqs[:int(len(reqs)*0.8)]
        test_reqs = reqs[int(len(reqs)*0.8):]
        # shuffle the requirement
        random.shuffle(train_reqs)
        # get the items for train and test
        train_items = []
        test_items = []
        for req in train_reqs:
            train_items.extend(items[req])
        for req in test_reqs:
            test_items.extend(items[req])
        train_data_docname.extend(train_items)
        test_data_docname.extend(test_items)

    # train_hop_stats = {"one_hop": 0, "two_hop": 0, "three_hop": 0}
    train_hop_stats = {f"{i}_hop": 0 for i in range(1, 6)}
    # test_hop_stats = {"one_hop": 0, "two_hop": 0, "three_hop": 0}
    test_hop_stats = {f"{i}_hop": 0 for i in range(1, 6)}
    # print the requirement and model name for train and test data_docname
    print("Train data_docname", len(train_data_docname))
    for item in train_data_docname:
        print(item['requirement'], item['model_name'])
        # train_hop_stats["one_hop"] += len(item['one_hop'])
        # train_hop_stats["two_hop"] += len(item['two_hop'])
        # train_hop_stats["three_hop"] += len(item['three_hop'])
        # update the hop_stats
        for key, value in item.items():
            if 'hop' in key:
                train_hop_stats[key] += len(value)
            # if 'negative_samples?' in key:
            #     train_hop_stats['negative_samples?'] += len(value)

    print(train_hop_stats)
    print("Test data_docname", len(test_data_docname))
    for item in test_data_docname:
        print(item['requirement'], item['model_name'])
        # test_hop_stats["one_hop"] += len(item['one_hop'])
        # test_hop_stats["two_hop"] += len(item['two_hop'])
        # test_hop_stats["three_hop"] += len(item['three_hop'])
        # update the hop_stats
        for key, value in item.items():
            if 'hop' in key:
                test_hop_stats[key] += len(value)
            # if 'negative_samples?' in key:
            #     test_hop_stats['negative_samples?'] += len(value)

    print(test_hop_stats)
    
    if args.output_dir:
        with open(args.output_dir + "/train_data_docname.json", 'w') as f:
            json.dump(train_data_docname, f, indent=4)
        with open(args.output_dir + "/test_data_docname.json", 'w') as f:
            json.dump(test_data_docname, f, indent=4)

    # convert the train and test data_docname into csv
    train_data = dict_to_csv(train_data_docname, with_trace=args.with_trace)
    # print(train_data)
    test_data = dict_to_csv(test_data_docname, with_trace=args.with_trace)

    # save the data to csv
    if args.output_dir:
        train_data.to_csv(args.output_dir + "/train_data_docname.csv", index=False)
        test_data.to_csv(args.output_dir + "/test_data_docname.csv", index=False)

    # group the data by model name
    group_by_model(train_data_docname, 'train-')
    group_by_model(test_data_docname, 'test-')

if __name__ == "__main__":
    args = argparse.ArgumentParser("create a multi-hop inference from a claim list of a Assurance Case CAE")
    args.add_argument("json_path", type=str, help="Path to the json file")
    # add output dir
    args.add_argument("--output_dir", type=str, help="Path to the output directory")

    args.add_argument("--with_trace", action='store_true', help="Include trace in the premise")
    args.set_defaults(with_trace=False)

    args = args.parse_args()

    # check if the output dir is exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
