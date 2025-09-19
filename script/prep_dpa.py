import random

import pandas as pd

# partially taken from: https://zenodo.org/records/11047441
'''
@article{Azeem2023AMS,
  title={A Multi-solution Study on GDPR AI-enabled Completeness Checking of DPAs},
  author={Muhammad Ilyas Azeem and Sallam Abualhaija},
  journal={Empir. Softw. Eng.},
  year={2023},
  volume={29},
  pages={96},
  url={https://api.semanticscholar.org/CorpusID:265445936}
}
'''


optional_reqs = ['R30', 'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'R39', 'R40', 'R41', 'R42', 'R43', 'R44', 'R45', 'R46']
excluded_reqs = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R1XX', 'Full', 'Partial']
excluded_more_reqs = ['R24', 'R22', 'R29', 'R11', 'R13', 'R20', 'R18', 'R19', 'R21', 'R16']
reqs_list = ['R10', 'R11', 'R12', 'R13', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29']
label_names_list = ['R10', 'R11', 'R12', 'R13', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'other']
labels_list = [19, 2, 4, 16, 17, 12, 6, 15, 0, 13, 3, 1, 10, 7, 14, 5, 11, 18, 9, 8]

def load_dataset(df):
    # # Dataset without the metadata requirements
    # df.loc[df['target'].isin(excluded_reqs), 'target'] = 'other'
    # # Dataset without the optional requirements
    # df.loc[df['target'].isin(optional_reqs), 'target'] = 'other'
    df['target'] = pd.Categorical(df['target'])
    df['label'] = df['target'].cat.codes

    # filter data with sentence length less than 3
    print(f'Total dataset size: {df.shape[0]}')
    df = df[df['Sentence'].str.split().str.len() > 3]
    print(f'Total dataset size after filtering: {df.shape[0]}')

    train = df.loc[df['dataset_type'] == 'train']
    test = df.loc[df['dataset_type'] == 'test']
    print(f'Train dataset size: {train.shape[0]}, Test dataset size: {test.shape[0]}')
    print(f'Total labels in train dataset: {len(train.target.unique())}, {train.target.unique()}')
    print(f'Total labels in test dataset {len(test.target.unique())}, {test.target.unique()}')
    # train = under_sample_other_label(train)
    return train, test


def prep_nli_data(data, class_label, negative_sampling_rate=1.0, use_other=True, ):
    pairs_data = []

    print(data)
    # map the data to the class labels description
    for idx, datum in data.iterrows():
        # get the class label description
        label = datum['target']
        if label != 'other':
            for idx_cls, class_label_row in class_label.iterrows():
                if str(class_label_row['ID']) == str(label):
                    # print(f"Processing requirement {idx}, label {label}", class_label_row['ID'])
                    pairs_data.append({
                        'ID': datum['DPA'],
                        # 'premise': datum['Sentence'], 'hypothesis': class_label_row['Requirement'], 
                        'premise': class_label_row['Requirement'], 'hypothesis': datum['Sentence'],
                        'label': 'entailment', 'class': class_label_row['ID'],
                        # unroll all the datum columns
                        'target': datum['target'], 'dataset_type': datum['dataset_type'],
                        })
                # elif label == 'other' and use_other:
                    # pairs_data.append({
                    #     'ID': datum['DPA'],
                    #     # 'premise': datum['Sentence'], 'hypothesis': class_label_row['Requirement'],
                    #     'premise': class_label_row['Requirement'], 'hypothesis': datum['Sentence'],
                    #     'label': 'not_entailment', 'class': class_label_row['ID'],
                    #     # unroll all the datum columns
                    #     'target': datum['target'], 'dataset_type': datum['dataset_type'],
                    #     })
                else:
                    if random.random() <= negative_sampling_rate:
                        pairs_data.append({
                            'ID': datum['DPA'],
                            # 'premise': datum['Sentence'], 'hypothesis': class_label_row['Requirement'], 
                            'premise': class_label_row['Requirement'], 'hypothesis': datum['Sentence'],
                            'label': 'not_entailment', 'class': class_label_row['ID'],
                            # unroll all the datum columns
                            'target': datum['target'], 'dataset_type': datum['dataset_type'],
                            })
        # iterate over data where label is others and take one requirement as sampling
        if label == 'other' and use_other:
            # iterate over data where label is others and take one requirement as sampling
            sample = class_label.sample(1)
            pairs_data.append({
                'ID': datum['DPA'],
                # 'premise': datum['Sentence'], 'hypothesis': class_label_row['Requirement'], 
                'premise': sample['Requirement'].values[0], 'hypothesis': datum['Sentence'],
                'label': 'not_entailment', 'class': sample['ID'].values[0],
                # unroll all the datum columns
                'target': datum['target'], 'dataset_type': datum['dataset_type'],
            })

    # print(f'Pairs data size: {len(pairs_data)}')
    return pairs_data 

def main():
    print('Loading the dataset')

    df_train = pd.read_csv('data/dpa-multi/train_set.csv')
    train, val = load_dataset(df_train)
    df_test = pd.read_csv('data/dpa-multi/test_set.csv')
    _, test = load_dataset(df_test)

    print('Finished loading the dataset')

    # count the label distribution except the 'other' label
    # Assuming df is your DataFrame and 'label' is the column containing the labels

    # Count the label distribution except the 'other' label
    label_distribution = train[train['target'] != 'other']['target'].value_counts()

    # Print the label distribution
    print(label_distribution, label_distribution.sum())

    # load the gdpr cleaned label description
    class_label = pd.read_csv('data/dpa-multi/gdpr_cleaned.csv')

    negative_sampling_rate = 0.1

    train_data = prep_nli_data(train, class_label, negative_sampling_rate=negative_sampling_rate)

    # make a few-shot dataset from the train dataframe and prep the NLI data for sampling
    few_shot = {}
    for i in range(1, 10):
        fs_negative_sampling_rate = 0.1
        # few_shot_positive = train[train['target'] != 'other'].sample(i, random_state=42)
        # get maxsample per class
        max_per_class = train[train['target'] != 'other']['target'].value_counts()
        # convert to dictionary
        max_per_class = max_per_class.to_dict()
        print(f'Max per class: {max_per_class}')
        # sample few-shot positive per class
        # few_shot_positive = train[train['target'] != 'other'].groupby('target').apply(lambda x: x.sample(i, random_state=42))

        all_few_shot_positive = []
        for k, v in max_per_class.items():
            few_shot_positive = None
            if v > 0:
                max_sample = min(i, v)
                few_shot_positive = train[train['target'] == k].sample(max_sample, random_state=42)
                all_few_shot_positive.append(few_shot_positive)
        
        few_shot_positive = pd.concat(all_few_shot_positive)
            
        few_shot[i] = prep_nli_data(few_shot_positive, class_label, negative_sampling_rate=fs_negative_sampling_rate)
        print(f'Few shot dataset size: {i}, {len(few_shot[i])}')
        print(few_shot_positive['target'].value_counts())

    # save the few-shot dataset to jsonl for each few-shot size
    for i in range(1, 10):
        pd.DataFrame(few_shot[i]).to_csv(f'data/dpa-multi/few-shot-{i}-nli.csv', index=False)

    # make new pandas dataframe with premise, hypothesis and label
    train_data = pd.DataFrame(train_data)

    # print the datasets label distribution count
    print(train_data['label'].value_counts())
    print(train_data['target'].value_counts())

    # save the dataset to csv
    # train_data.to_csv('data/dpa-multi/train-nli.csv', index=False)

    # val_data = prep_nli_data(val, negative_sampling_rate=0.0)
    # val_data = pd.DataFrame(val_data)
    # val_data.to_csv('data/dpa-multi/val-nli.csv', index=False)

    print('Finished preparing the validation dataset')
    # print(val_data['target'].value_counts())

    test_data = prep_nli_data(test, class_label, negative_sampling_rate=0.0, use_other=False)
    test_data = pd.DataFrame(test_data)
    test_data.to_csv('data/dpa-multi/test-nli.csv', index=False)

    test_data_all = prep_nli_data(test, class_label, negative_sampling_rate=0.0, use_other=True)
    test_data_all = pd.DataFrame(test_data_all)
    test_data_all.to_csv('data/dpa-multi/test-nli-others.csv', index=False)

    print('Finished preparing the test dataset')
    print(test_data['target'].value_counts())

    print('Finished preparing the test dataset')
    print(test_data_all['target'].value_counts())

if __name__ == "__main__":
    main()


        