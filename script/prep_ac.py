import re
import argparse
import glob
from pathlib import Path
import math
import json

import itertools

from typing import List, Union, Counter
from collections import OrderedDict

# import numpy as np
import networkx as nx

# import pandas as pd


import logging

log = logging.getLogger()
logging.basicConfig(level='DEBUG',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s')


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
            numeric = 0
    
    return (non_numeric, numeric)
    



def flatten_json(json_data, id=1, parent='root'):

    entities = []
    parent_dict = OrderedDict()
    parent_dict['root'] = None

    for key, val in json_data.items():
        # log.info(f'key: {key}')
        
        if isinstance(val, dict):
            if 'description' in val:
                if key.startswith("Evidence"):
                    evidence_type = val["type"]
                else:
                    evidence_type = math.nan
                key_extracted = extract_numeric_suffix(key)
                entities.append([id, key_extracted[0], key_extracted[1], val["description"], evidence_type, parent])
                parent_dict[f'{key}-{id}'] = parent
                id += 1

            if isinstance(val, dict):
                for kkey, vval in val.items():
                    if kkey == "description":
                        continue

                    if isinstance(vval, list):
                        for vvval in vval:
                            # log.info('recursive')
                            sub_entities, sub_parent_dict = flatten_json(vvval, id=id, parent=f'{key}-{id-1}')
                            entities.extend(sub_entities)
                            parent_dict.update(sub_parent_dict)
                            id += 1

    return parent_dict, entities


def traverse_json_entities(json_data):
    # Dictionary to store parent of each key
    parent_dict = OrderedDict()
    parent_dict['root'] = None
    entities = OrderedDict()

    # Counter to generate unique IDs
    id_counter = itertools.count(0)

    def traverse_json(obj, parent='root'):
        # if isinstance(obj, Union[str, int, float, bool]):
        #     # base case
        #     return
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_id = next(id_counter)

                if isinstance(value, dict):
                    if 'description' in value:
                        if key.startswith("Evidence"):
                            # evidence_type = value["type"]
                            evidence_type = value.get("type", None)
                        else:
                            evidence_type = None
                        key_extracted = extract_numeric_suffix(key)
                        entities[f'{key}-{current_id}'] = {
                            'type': key_extracted[0],
                            'number': key_extracted[1],
                            'description': value["description"],
                            'evidence_type': evidence_type,
                            'parent': parent
                        }

                # check if parent exists
                if parent not in parent_dict:
                    raise ValueError(f'Parent {parent} does not exist in parent_dict, {parent_dict}')
                
                if f'{key}-{current_id}' in parent_dict:
                    raise ValueError(f'Key {key}-{current_id} already exists in parent_dict, {parent_dict}')
                
                if key not in ['description','type', 'source', 'date']:
                    parent_dict[f'{key}-{current_id}'] = parent

                traverse_json(value, f'{key}-{current_id}')

        elif isinstance(obj, list):
            for item in obj:
                if parent not in parent_dict:
                    raise ValueError(f'Parent {parent} does not exist in parent_dict, {parent_dict}')
                else:
                    traverse_json(item, parent_dict[parent])


    # Traverse the JSON data
    traverse_json(json_data)

    # remove root from parent_dict
    # parent_dict.pop('root')
    list_key = set(parent_dict.keys() - entities.keys())
    # print(list_key)
    for key in list_key:
        parent_dict.pop(key)

    # print parent dict
    # for key, val in parent_dict.items():
        # print(key, val)

    # print(len(parent_dict), len(entities), len(list_key))

    return parent_dict, entities


def custom_graph_edit_distance(g1, g2, root1, root2, timeout=10):
    # Ensure the roots are in the graphs
    if root1 not in g1 or root2 not in g2:
        raise ValueError("Roots must be in the respective graphs")

    # Create subgraphs rooted at the specified nodes
    subgraph1 = nx.dfs_tree(g1, root1)
    subgraph2 = nx.dfs_tree(g2, root2)

    # Compute the graph edit distance between the subgraphs
    ged = nx.graph_edit_distance(subgraph1, subgraph2, timeout=timeout)
    return ged

def main(args):
    # json_files = glob.glob('./*/iec62443*/*.json')
    # json_files = glob.glob('./data/sac_iec/iec62443*/*.json')
    json_files = glob.glob(f'{args.input_dir}/**/*.json', recursive=True)

    # file = json_files[0]
    # pd.read_json(file)

    # print(json_files)

    with open(json_files[0], 'r') as file:
        json_data = file.read()

    data = json.loads(json_data)

    # print(data)


    claims_list = []
    claims_pd = []
    for file in json_files:

        path = Path(file)
        
        log.info(f'Processing {path.stem}')
        file_features = path.stem.split('_')
        
        docname = path.parent.name.split('_')[0]
        file_features_dict = {
            'requirement': file_features[1],
            'model_name': file_features[0],
            'generation': file_features[2],
            'docname': path.parent.name
        }



        with open(file, 'r') as file:
            json_data = file.read()
        data = json.loads(json_data)

        # claims_pd.append(pd.DataFrame(flatten_json(data), columns=['id','type','number','description','evidence_type']).assign(**file_features_dict))
        # result = flatten_json(data, id=0)
        # parent_dict, entities = result
        # print(len(parent_dict), len(entities))
        # claims_pd.append(result)
        # check if data['MainClaim'] is template The primary claim that is being argued or discussed
        # if template, then the file is invalid
        if 'MainClaim' not in data:
            log.warning(f'Invalid file: {file}, no MainClaim')
            file_features_dict['invalid'] = True
            continue
        if data['MainClaim']['description'] == "The primary claim that is being argued or discussed.":
            log.warning(f'Invalid file: {file}')
            file_features_dict['invalid'] = True
            continue
        
        parent_dict, entities = traverse_json_entities(data)
        claims_list.append(entities)

        # iterate the entities to check if template in the description
        break_and_continue = False
        for key, val in entities.items():
            # log.info(f'key: {key}, val: {val}')
            # if "Description of Evidence" in val['description']:
            # check if match template regex Description of Evidence NUM supporting * NUM
            if re.match(r'^Description of Evidence \d+', val['description']):
                log.warning(f'Invalid file: {file}, use template')
                file_features_dict['invalid'] = True
                break_and_continue = True
                break
                
        if break_and_continue:
            # remove claims_list[-1]
            claims_list.pop()
            continue
                

        # from the parent dict infer how many direct child entities each parent has
        child_count = Counter(parent_dict.values())
        # print(child_count)
        # make list of parent with child type without number
        parent_child = {key: [key_ent for key_ent, val in entities.items() if val['parent'] == key] for key in parent_dict.keys()}
        # count the number of nodes and edges where edges are the number of child nodes
        file_features_dict['num_nodes'] = len(entities)
        file_features_dict['num_edges'] = sum(child_count.values())
        file_features_dict['parent_child'] = str(parent_child)

        # check if invalid when there is no evidence node
        # iterate parent child, check if EvidenceNum-Num is in the parent_child if not, then it is invalid
        if 'Evidence' not in str(parent_child) or 'ArgumentSubClaim' not in str(parent_child) or 'ArgumentClaim' not in str(parent_child) or 'SubClaim' not in str(parent_child):
            log.warning(f'Invalid file: {file}')
            file_features_dict['invalid'] = True
            # remove claims_list[-1]
            claims_list.pop()
        # check if main claim just copy template: "The primary claim that is being argued or discussed."
        else:
            # add file features_dict to the last element of the list
            # print(type(claims_list[-1]))
            claims_list[-1].update(file_features_dict)

    # print(claims_list)
    claims_list_to_file = []
    num_of_instances = 0
    for ents in claims_list:
        claims_list_to_file.append(ents.copy())

        # num_of_instances += len(ents)

        type_counts = Counter()

        requirement = ents['requirement']
        model_name = ents['model_name']
        generation = ents['generation']
        # doc = ents['iec62443']

        for key, val in ents.items():
            if isinstance(val, dict):
                # print(key, val, type(val))
                # add parent description to the entity
                if val['parent'] == 'root':
                    pass
                elif val['parent'] is not None:
                    parent_key = val['parent']
                    if parent_key in ents:
                        val['parent_description'] = ents[parent_key]['description']
                else:
                    raise ValueError(f'Parent of {key} is None')
                
                # update type counts
                type_counts[val['type']] += 1

                num_of_instances += 1

                # save_val = val.copy()
                # save_val.update({'requirement': requirement, 'model_name': model_name, 'generation': generation, 'iec62443': doc})
                # claims_list_to_file.append(save_val)

        # add counts to the entity
        ents['type_counts'] = type_counts

    # print(num_of_instances)
    log.info(f'Number of instances: {num_of_instances}')

    # save claims_list to a json file
    # with open('data/sac_iec/claims_list.json', 'w') as file:
    with open(args.output_file, 'w') as file:
        # print(len(claims_list))
        json.dump(claims_list_to_file, file, indent=4)

    # group the claims by requirement
    claims_list_grouped = {}
    for ents in claims_list:
        requirement = ents['requirement']
        if requirement not in claims_list_grouped:
            claims_list_grouped[requirement] = []
        claims_list_grouped[requirement].append(ents)

    all_type_count = []

    intra_model = []
    intra_model_dict = {}
    inter_model = []
    inter_model_dict = {}

    intra_model_struct = []
    inter_model_struct = []

    # calculate the average counts of subclaims, argument claims, argument subclaims, and evidences
    for requirement, ents_list in claims_list_grouped.items():
        num_ents = len(ents_list)
        print(requirement, num_ents)
        type_counts = Counter()
        for ents in ents_list:
            type_counts.update(ents['type_counts'])

        # count the type counts average
        for key, val in type_counts.items():
            type_counts[key] = val / len(ents_list)
            # print(key, val, len(ents_list), type_counts[key])
        

        # within requirements, analyze the consistency of the subclaims, argument claims, argument subclaims, and evidences within the models
        print(requirement)
        # group by the models
        claims_list_grouped_models = {}
        for ents in ents_list:
            model_name = ents['model_name']
            if model_name not in claims_list_grouped_models:
                claims_list_grouped_models[model_name] = []
            claims_list_grouped_models[model_name].append(ents)
        
        print("intra-model consistency")
        type_grouped = {}
        for model_name, ents_list_model in claims_list_grouped_models.items():
            print(model_name)
            type_counts_models = Counter()
            for ents in ents_list_model:
                # create graph of the entities
                # g_ents = nx.DiGraph()
                g_ents = nx.Graph()
                # print(ents['parent_child'])
                parent_child = ents['parent_child']
                try:
                    parent_child = eval(parent_child)
                except Exception as e:
                    print(parent_child)
                
                # make parent_child numeric by using ordered int
                nodes_numeric = {key: i for i, key in enumerate(parent_child.keys())}
                parent_child_numeric = {nodes_numeric[key]: [nodes_numeric[key_ent] 
                                        for key_ent in val] for key, val in parent_child.items()}
                # print(parent_child_numeric)

                # print(parent_child)
                for node, child_nodes in parent_child_numeric.items():
                    # print(key, val)
                    for chiild_node in child_nodes:
                        # print(key_ent)
                        g_ents.add_edge(node, chiild_node)
                
                # print(g_ents)
                # add the graph to the entity
                ents['graph'] = g_ents
                type_counts_models.update(ents['type_counts'])
            
            # calculate the model type counts average
            for key, val in type_counts_models.items():
                type_counts_models[key] = val / len(ents_list_model)
                print(key, val, len(ents_list_model), type_counts[key])
            # claims_list_grouped_models[model_name] = type_counts_models
            type_grouped[model_name] = type_counts_models
        
            intra_diff = []
            struct_diff = []
            for ents1 in ents_list_model:
                for ents2 in ents_list_model:
                    if ents1['generation'] == ents2['generation']:
                        continue
                    # print(ents1['model_name'], ents2['model_name'])
                    diff_ = {key: abs(ents1['type_counts'][key] - ents2['type_counts'][key]) for key in ents1['type_counts'].keys()}
                    intra_diff.append(diff_)

                    g_ents1 = ents1['graph']
                    g_ents2 = ents2['graph']

                    # print(g_ents1.nodes)

                    intra_ged = 0
                    # check if the graph is empty
                    if len(g_ents1.nodes) > 0 and len(g_ents2.nodes) > 0:
                        # intra_ged = nx.graph_edit_distance(g_ents1, g_ents2, roots=(0, 0), timeout=10)
                        intra_ged = next(nx.optimize_graph_edit_distance(g_ents1, g_ents2))
                    # intra_ged = custom_graph_edit_distance(g_ents1, g_ents2, 
                    #                                        'MainClaim-0', 'MainClaim-0', 
                    #                                        timeout=10)
                    # print("graph edit distance: ", intra_ged)
                    struct_diff.append(intra_ged)
            
            # average diff between the models
            diff_avg = {}
            if len(intra_diff) > 0:
                for key in intra_diff[0].keys():
                    # print(intra_diff)
                    diff_avg[key] = sum(d[key] for d in intra_diff) / len(intra_diff)
            # print(f"intra-model consistency {model_name}")
            print(diff_avg)
            # intra_model[f'{requirement}-{model_name}'] = diff_avg
            intra_model.append(diff_avg)
            if model_name in intra_model_dict:
                intra_model_dict[model_name].append(diff_avg)
            else:
                intra_model_dict[model_name] = [diff_avg]

            if len(struct_diff) > 0:
                print(f"average graph edit distance intra-model {model_name} : ", sum(struct_diff) / len(struct_diff))
                intra_model_struct.append({model_name: sum(struct_diff) / len(struct_diff)})
        
        # print(claims_list_grouped_models)

        # inter-model consistency within requirements
        inter_diff = {}
        inter_struct_diff = {}
        for model_name, type_counts in type_grouped.items():
            # print(len(claims_list_grouped_models[model_name]))
            for model_name2, type_counts2 in type_grouped.items():
                if model_name == model_name2:
                    continue
                # elif (model_name2, model_name) in inter_diff:
                #     print("already calculated, bi-directional", model_name, model_name2)
                #     continue
                else:
                    inter_diff[(model_name, model_name2)] = {key: abs(type_counts[key] - type_counts2[key]) for key in type_counts.keys()}
                    print(f"inter-model consistency {model_name} and {model_name2}", inter_diff[(model_name, model_name2)])

                    gents1 = [ents['graph'] for ents in claims_list_grouped_models[model_name]]
                    gents2 = [ents['graph'] for ents in claims_list_grouped_models[model_name2]]
                    # calculate the graph edit distance between the two models output
                    inter_ged = []
                    for g1 in gents1:
                        for g2 in gents2:
                            if len(g1.nodes) > 0 and len(g2.nodes) > 0:
                                # inter_ged.append(nx.graph_edit_distance(g1, g2, roots=(0, 0), timeout=10))
                                inter_ged.append(next(nx.optimize_graph_edit_distance(g1, g2)))
                            # inter_ged.append(custom_graph_edit_distance(g1, g2, 
                            #                 'MainClaim-0', 'MainClaim-0', 
                            #                 timeout=10))
                            # inter_ged.append(0)
                    print(f"average graph edit distance between {model_name} and {model_name2}: ", sum(inter_ged) / len(inter_ged))
                    inter_struct_diff[(model_name, model_name2)] = sum(inter_ged) / len(inter_ged)
                
        # print('inter-model consistency')
        # pretty print the diff
        for key, val in inter_diff.items():
            # print(key, val)
            # inter_model[f'{requirement}-{key}'] = val
            inter_model.append(val)
            # inter_model.append({f'{requirement}-{key}': val})
            if key in inter_model_dict:
                inter_model_dict[key].append(val)
            else:
                inter_model_dict[key] = [val]
        
        for key, val in inter_struct_diff.items():
            # print(key, val)
            inter_model_struct.append({key: val})

    # print(intra_model[0], len(intra_model))
    # calculate intra for the whole dataset
    print("average intra-model consistency for each type")
    intra_type_cnt = {key: 0 for key in intra_model[0].keys()}
    # print(inter_model)
    for intra in intra_model:
        for key, val in intra.items():
            intra_type_cnt[key] += val

    for key, val in intra_type_cnt.items():
        intra_type_cnt[key] = val / len(intra_model)
        print(key, val, len(intra_model), intra_type_cnt[key])

    # intra_model_type_cnt = {key: 0 for key in intra_model_dict.keys()}
    # for key, val in intra_model_dict.items():
    #     # print(key, val)
    #     for intra in val:
    #         for key_, val_ in intra.items():
    #             intra_model_type_cnt[f'{key}-{key_}'] += val_
    
    # for key, val in intra_model_type_cnt.items():
    #     intra_model_type_cnt[key] = val / len(intra_model_dict[key])
    #     print(key, val, len(intra_model_dict[key]), intra_model_type_cnt[key])


    # print(inter_model[0], len(inter_model))
    # calculate inter for the whole dataset
    print("average inter-model consistency for each type")
    inter_type_cnt = {key: 0 for key in inter_model[0].keys()}
    for inter in inter_model:
        # print(inter)
        for key, val in inter.items():
            inter_type_cnt[key] += val

    for key, val in inter_type_cnt.items():
        inter_type_cnt[key] = val / len(inter_model)
        print(key, val, len(inter_model), inter_type_cnt[key])

    # inter_model_type_cnt = {key: 0 for key in inter_model_dict.keys()}
    # for key, val in inter_model_dict.items():
    #     # print(key, val)
    #     for inter in val:
    #         for key_, val_ in inter.items():
    #             inter_model_type_cnt[f'{key}-{key_}'] += val_
    
    # for key, val in inter_model_type_cnt.items():
    #     inter_model_type_cnt[key] = val / len(inter_model_dict[key])
    #     print(key, val, len(inter_model_dict[key]), inter_model_type_cnt[key])


    avg_intra_ged = [list(d.values())[0] for d in intra_model_struct]
    intra_model_struct_per_model = {}
    for intra in intra_model_struct:
        for key, val in intra.items():
            if key not in intra_model_struct_per_model:
                intra_model_struct_per_model[key] = [val]
            intra_model_struct_per_model[key].append(val)

    # print("graph edit distance intra-model", intra_model_struct)
    print("average graph edit distance intra-model per model", intra_model_struct_per_model)

    print("average graph edit distance intra-model", sum(avg_intra_ged) / len(avg_intra_ged))

    # print(inter_model_struct)
    avg_inter_ged = [list(d.values())[0] for d in inter_model_struct]
    inter_model_struct_per_pairs = {}
    for inter in inter_model_struct:
        for key, val in inter.items():
            if key not in inter_model_struct_per_pairs:
                inter_model_struct_per_pairs[key] = [val]
            inter_model_struct_per_pairs[key].append(val)

    print("average graph edit distance inter-model per pairs")
    for key, val in inter_model_struct_per_pairs.items():
        print(key, len(val), sum(val) / len(val))
    print("average graph edit distance inter-model", sum(avg_inter_ged) / len(avg_inter_ged))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
