import random
import json
import argparse

from typing import Iterable

import pandas as pd

from prep_dpa import load_dataset

def gather_all_description(claim_arg_evidence: dict):
    all_desc = ''
    for k, v in claim_arg_evidence.items():
        # print(k, v)
        if isinstance(v, dict) and 'description' in v:
            all_desc += '. ' +v['description']
    return all_desc

def gather_evidence(claim_arg_evidence: dict):
    all_evidence = ''
    for k, v in claim_arg_evidence.items():
        if 'evidence' in v.lower():
            all_evidence += '. ' + v['description']
    return all_evidence

def gather_main_claim(claim_arg_evidence: dict):
    main_claim = claim_arg_evidence['MainClaim-0']['description']
    return main_claim


def prep_nli_data(data: pd.DataFrame, class_label: Iterable, negative_sampling_rate: float =1.0, use_other: bool =True, ):
    pairs_data = []

    print(data)
    # map the data to the class labels description
    for idx, datum in data.iterrows():
        # get the class label description
        label = datum['target']
        dataset_type = datum['dataset_type']
        sentence = datum['Sentence']
        for ac in class_label:
            if str(ac['requirement']) == str(label):
                # print(f"Processing requirement {idx}, label {label}", class_label_row['ID'])
                # iterate ac dict which have description
                # print(ac)
                # desc = gather_all_description(ac)
                desc = gather_main_claim(ac)
                pairs_data.append({
                    'ID': datum['DPA'],
                    'premise': desc, 'hypothesis': sentence,
                    # 'premise': sentence, 'hypothesis': desc,
                    'label': 'entailment', 'class': ac['requirement'],
                    # unroll all the datum columns
                    'target': label, 'dataset_type': dataset_type,
                })
            elif label == 'other' and use_other:
                if random.random() <= negative_sampling_rate:
                # iterate ac dict which have description
                    desc = gather_main_claim(ac)
                    pairs_data.append({
                        'ID': datum['DPA'],
                        'premise': desc, 'hypothesis': sentence,
                        # 'premise': sentence, 'hypothesis': desc,
                        'label': 'not_entailment', 'class': ac['requirement'],
                        # unroll all the datum columns
                        'target': label, 'dataset_type': dataset_type,
                    })
                    # break from the loop to avoid adding multiple negative samples for the same datum
                    break
            else:
                if random.random() <= negative_sampling_rate:
                    # iterate ac dict which have description
                    desc = gather_main_claim(ac)
                    pairs_data.append({
                        'ID': datum['DPA'],
                        'premise': desc, 'hypothesis': sentence,
                        # 'premise': sentence, 'hypothesis': desc,
                        'label': 'not_entailment', 'class': ac['requirement'],
                        # unroll all the datum columns
                        'target': label, 'dataset_type': dataset_type,
                    })
    return pairs_data 

def main(args):

    print('Loading the dataset')

    df_train = pd.read_csv('data/dpa-multi/train_set.csv')
    df_train = pd.read_csv(args.train_path)
    train, val = load_dataset(df_train)

    df_test = pd.read_csv('data/dpa-multi/test_set.csv')
    df_test = pd.read_csv(args.test_path)
    _, test = load_dataset(df_test)

    print('Finished loading the dataset')

    # count the label distribution except the 'other' label
    # Assuming df is your DataFrame and 'label' is the column containing the labels

    # Count the label distribution except the 'other' label
    # label_distribution = train[train['target'] != 'other']['target'].value_counts()

    # Print the label distribution
    # print(label_distribution, label_distribution.sum())

    # load the gdpr cleaned label description
    acs = {}
    with open('data/sac_gdpr/merged_gdpr.json', 'r') as f:
        acs = json.load(f)
        print(len(acs))

    negative_sampling_rate = 0.1

    train_data = prep_nli_data(train, acs, negative_sampling_rate=negative_sampling_rate)

    # make new pandas dataframe with premise, hypothesis and label
    train_data = pd.DataFrame(train_data)

    # print the datasets label distribution count
    # print(train_data['label'].value_counts())
    # print(train_data['target'].value_counts())

    # save the dataset to csv
    train_data.to_csv('data/dpa-multi/train-sac-nli.csv', index=False)


    test_data = prep_nli_data(test, acs, negative_sampling_rate=0.1, use_other=True)
    test_data = pd.DataFrame(test_data)
    test_data.to_csv('data/dpa-multi/test-sac-nli.csv', index=False)

    print('Finished preparing the test dataset')
    print(test_data['target'].value_counts())


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Prepare the DPA dataset for Multi-hop NLI from SAC.")
    args.add_argument('--train_path', type=str, default='data/dpa-multi/train_set.csv', help='Path to the train set')
    args.add_argument('--test_path', type=str, default='data/dpa-multi/test_set.csv', help='Path to the test set')
    args.add_argument('--sac_path', type=str, default='data/sac_gdpr/merged_gdpr.json', help='Path to the sac gdpr json file')
    args.add_argument('--output_path', type=str, default='data/dpa-multi/train-sac-nli.csv', help='Path to the output set')
    args = args.parse_args()
    main(args)
