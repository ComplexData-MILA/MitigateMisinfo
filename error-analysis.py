'''
Runs error analysis found in section 4.5 of the paper.
Data for this script can be found here: https://drive.google.com/drive/folders/1Ur-JcWBfQ4HfB0Fr8Sm-TnRqpqATrsXQ?usp=drive_link
'''


import json
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import pairwise_distances
import pandas as pd

import random
random.seed(1234)
EXCLUDE_NON_NUMERIC = True # whether to exclude cases where GPT-4 did not return a score, which were checked manually to be Impossible

def load_data(json_file):
    total_failed_cases = 0
    failed_cases = []
    data = []
    with open(json_file, 'r') as f:
        for line in f:
            row = json.loads(line)
            try:
                gpt4_answer = int(row['gpt4-answer'])
                data.append((row['label'], gpt4_answer))
            except ValueError:
                if not EXCLUDE_NON_NUMERIC:
                  data.append((row['label'], random.randint(0,100)))
                failed_cases.append(row)
                total_failed_cases += 1
                pass  # Ignore rows with non-integer 'gpt4-answer' values
    print('cases with non-int output:', total_failed_cases)
    return data, failed_cases

def to_binary(label, threshold):
    return 1 if label >= threshold else 0

def optimize_threshold(y_true, y_pred):
    best_threshold = 0
    best_score = 0
    for t in range(0, 101):
        binary_preds = [to_binary(p, t) for p in y_pred]
        score = f1_score(y_true, binary_preds, average='weighted')
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold, best_score

def evaluate_threshold(y_true, y_pred, threshold):
    binary_preds = [to_binary(p, threshold) for p in y_pred]
    f1 = f1_score(y_true, binary_preds, average='weighted')
    acc = accuracy_score(y_true, binary_preds)
    return f1, acc

# File paths
validation_json_file = "LIAR_val_gpt4preds.jsonl"
test_json_file = "LIAR_test_gpt4preds.jsonl"

# Load validation and test data
validation_data, failed_cases = load_data(validation_json_file)
test_data, failed_cases_test = load_data(test_json_file)

# Convert to binary labels
y_val_true_binary = [to_binary(label, 3) for label, _ in validation_data]
y_val_pred_binary = [gpt4 for _, gpt4 in validation_data]

y_test_true_binary = [to_binary(label, 3) for label, _ in test_data]
y_test_pred_binary = [gpt4 for _, gpt4 in test_data]

# Check the distribution of binary labels
val_binary_distribution = Counter(y_val_true_binary)
test_binary_distribution = Counter(y_test_true_binary)

print('GPT-4 RESULTS')

print("Validation binary label distribution:", val_binary_distribution)
print("Test binary label distribution:", test_binary_distribution)

# Optimize threshold using validation set
best_threshold, best_score = optimize_threshold(y_val_true_binary, y_val_pred_binary)

# Evaluate performance on the test dataset
test_f1, test_acc = evaluate_threshold(y_test_true_binary, y_test_pred_binary, best_threshold)
test_f1_50, test_acc_50 = evaluate_threshold(y_test_true_binary, y_test_pred_binary, 50)

print("Optimal Threshold:", best_threshold)
print("Validation Score:", best_score)
print("Binary", "Test F1 Score:", test_f1, "Test Accuracy:", test_acc)
print("Binary", "Test F1 Score with threshold=50:", test_f1_50, "Test Accuracy with threshold=50:", test_acc_50)


'''
cases with non-int output: 40
cases with non-int output: 69
GPT-4 RESULTS
Validation binary label distribution: Counter({1: 651, 0: 593})
Test binary label distribution: Counter({1: 673, 0: 525})
Optimal Threshold: 71
Validation Score: 0.6651424305924835
Binary Test F1 Score: 0.6913163497145366 Test Accuracy: 0.6936560934891486
Binary Test F1 Score with threshold=50: 0.6068242750326421 Test Accuracy with threshold=50: 0.6527545909849749
'''




# ROBERTA

def get_prob_binary(inp):
    return inp['probs'][0]
def get_label(inp):
    return inp['label'][0]

df_test_preds = pd.read_json('test_binary_preds.jsonl', lines=True, orient='records')
df_test = pd.read_json('LIAR_test_binary.jsonl', lines=True, orient='records')

#df_test['prob'] = df_test_preds[0].apply(get_prob_binary)
df_test['pred_label'] = df_test_preds[0].apply(get_label)

impossible_cases = pd.read_csv('gpt4-failedcases-test-labeling - Sheet1.tsv', sep='\t')
impossible_cases_ids = impossible_cases.id.tolist()

print('ROBERTA RESULTS')
print('unfiltered')

print(round(100*accuracy_score(df_test.label, df_test.pred_label.astype(int)),1))
print(round(100*f1_score(df_test.label, df_test.pred_label.astype(int), average='weighted'),1))

print('filtered to only include "possible" cases')

df_test_filtered = df_test[~df_test.id.isin(impossible_cases_ids)]

print(round(100*accuracy_score(df_test_filtered.label, df_test_filtered.pred_label.astype(int)),1))
print(round(100*f1_score(df_test_filtered.label, df_test_filtered.pred_label.astype(int), average='weighted'),1))

print('filtered to investigate only the "impossible" cases')

df_test_reverse_filtered = df_test[df_test.id.isin(impossible_cases_ids)]

print(round(100*accuracy_score(df_test_reverse_filtered.label, df_test_reverse_filtered.pred_label.astype(int)),1))
print(round(100*f1_score(df_test_reverse_filtered.label, df_test_reverse_filtered.pred_label.astype(int), average='weighted'),1))


'''
ROBERTA RESULTS
unfiltered
63.5
62.1
filtered to only include "possible" cases
63.8
62.3
filtered to investigate only the "impossible" cases
59.4
58.4
'''


binarized_gpt = [to_binary(x, best_threshold) for x in y_test_pred_binary]

df_test_filtered['gpt4_pred'] = binarized_gpt
df_test_filtered['gpt4_raw'] = y_test_pred_binary

ada_preds_train = pd.read_json('train_preds_openai-ada-002.jsonl', lines=True)
ada_preds_test = pd.read_json('test_preds_openai-ada-002.jsonl', lines=True)

investigation_target = df_test_filtered[(df_test_filtered.gpt4_pred.astype(int) != df_test_filtered.label.astype(int)) & (df_test_filtered.pred_label.astype(int) == df_test_filtered.label.astype(int))]
investigation_target_ids = investigation_target.id.tolist()
ada_preds_test_filtered = ada_preds_test[ada_preds_test.id.isin(investigation_target_ids)]


ada_preds_train = ada_preds_train.reset_index()
train_embeddings = ada_preds_train.embedding.tolist()

test_embeddings = ada_preds_test_filtered.embedding.tolist()

distances_mtx = pairwise_distances(train_embeddings,test_embeddings, metric='cosine')

min_distances = np.amin(distances_mtx, axis=0)
ada_preds_test_filtered['min_distance'] = min_distances


distances_argmin = np.argmin(distances_mtx, axis=0)
def get_argmin_train_text(inp):
    return ada_preds_train.iloc[inp].text
ada_preds_test_filtered['argmin_distance'] = distances_argmin
ada_preds_test_filtered['argmin_train_example'] = ada_preds_test_filtered.argmin_distance.apply(get_argmin_train_text)


#ada_preds_test.to_json('gpt-right_roberta-wrong.jsonl', lines=True, orient='records')
ada_preds_test_filtered.to_json('gpt-wrong_roberta-right.jsonl', lines=True, orient='records')

print('average min distance', np.mean(min_distances), 'standard deviation', np.std(min_distances), 'sample size', len(min_distances) )


'''average min distance 0.11674094329845495 standard deviation 0.031702800655761594 sample size 174'''



investigation_target = df_test_filtered[(df_test_filtered.gpt4_pred.astype(int) == df_test_filtered.label.astype(int)) & (df_test_filtered.pred_label.astype(int) != df_test_filtered.label.astype(int))]
investigation_target_ids = investigation_target.id.tolist()
ada_preds_test_filtered = ada_preds_test[ada_preds_test.id.isin(investigation_target_ids)]


ada_preds_train = ada_preds_train.reset_index()
train_embeddings = ada_preds_train.embedding.tolist()

test_embeddings = ada_preds_test_filtered.embedding.tolist()


distances_mtx = pairwise_distances(train_embeddings,test_embeddings, metric='cosine')

min_distances = np.amin(distances_mtx, axis=0)
ada_preds_test_filtered['min_distance'] = min_distances


distances_argmin = np.argmin(distances_mtx, axis=0)
def get_argmin_train_text(inp):
    return ada_preds_train.iloc[inp].text
ada_preds_test_filtered['argmin_distance'] = distances_argmin
ada_preds_test_filtered['argmin_train_example'] = ada_preds_test_filtered.argmin_distance.apply(get_argmin_train_text)


ada_preds_test_filtered.to_json('gpt-right_roberta-wrong.jsonl', lines=True, orient='records')

print('average min distance', np.mean(min_distances), 'standard deviation', np.std(min_distances), 'sample size', len(min_distances) )


'''average min distance 0.12740541884984716 standard deviation 0.02943612268240887 sample size 241'''