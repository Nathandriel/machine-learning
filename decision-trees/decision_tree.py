from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import random
from pprint import pprint


def main():
    df = pd.read_csv("Iris.csv")
    df = df.drop("Id", axis=1)
    df = df.rename(columns={"Species": "output"})

    train_df, test_df = train_test_split(df, test_size=0.5)

    tree = decision_tree_algorithm(train_df)
    pprint(tree)

    accuracy = measure_accuracy(test_df, tree)
    print "Accuracy: " + str(accuracy)


def train_test_split(df, test_size):
    test_size = int(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):
    output = data[:, -1]
    ouput_classes = np.unique(output)

    if len(ouput_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    output = data[:, -1]
    output_classes, counts = np.unique(output, return_counts=True)

    index = counts.argmax()
    classification = output_classes[index]
    return classification


def get_possible_splits(data):
    possible_splits = {}
    _, columns = data.shape
    for col in range(columns-1):
        values = data[:, col]
        unique_values = np.unique(values)

        feature_type = FEATURE_TYPES[col]

        if feature_type == "continuous":
            possible_splits[col] = []
            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index-1]
                    possible_split = (current_value + previous_value)/2

                    # possible_splits[col].append(possible_split)
                    possible_splits[col] = np.append(
                        possible_splits[col], possible_split)

                else:  # categorical
                    possible_splits[col] = unique_values

    return possible_splits


def split_data(data, column_to_split, split_value):
    split_column_values = data[:, column_to_split]

    feature_type = FEATURE_TYPES[column_to_split]
    if feature_type == "continuous":
        data_left = data[split_column_values <= split_value]
        data_right = data[split_column_values > split_value]
    else:  # categorical
        data_left = data[split_column_values == split_value]
        data_right = data[split_column_values != split_value]

    return data_left, data_right


def get_entropy(data):
    output = data[:, -1]
    _, counts = np.unique(output, return_counts=True)

    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def get_overall_entropy(data_left, data_right):

    size = len(data_left) + len(data_right)
    pl = len(data_left) / size
    pr = len(data_right) / size

    overall_entropy = (pl * get_entropy(data_left) +
                       pr * get_entropy(data_right))

    return overall_entropy


def decision_tree_algorithm(df, counter=0, min_samples=2):

    if counter == 0:
        global COL_HEADERS, FEATURE_TYPES
        COL_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # base case
    if (check_purity(data)) or (len(data) < min_samples):
        classification = classify_data(data)
        return classification

    else:
        counter += 1

        possible_splits = get_possible_splits(data)
        column_to_split, split_value = get_best_split(data, possible_splits)
        data_left, data_right = split_data(data, column_to_split, split_value)

        # instantiate a sub-tree
        feature_name = COL_HEADERS[column_to_split]
        question = "{} <= {}" .format(feature_name, split_value)
        sub_tree = {question: []}

        # find answers
        yes_answer = decision_tree_algorithm(data_left, counter, min_samples)
        no_answer = decision_tree_algorithm(data_right, counter, min_samples)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

    return sub_tree


def get_best_split(data, possible_splits):

    overall_entropy = 999
    for col in possible_splits:
        for value in possible_splits[col]:
            data_left, data_right = split_data(
                data, column_to_split=col, split_value=value)
            current_overall_entropy = get_overall_entropy(
                data_left, data_right)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_col = col
                best_split_value = value

    return best_split_col, best_split_value


def classify_sample(sample, tree):
    question = list(tree.keys())[0]
    feature_name, operator, value = question.split(" ")

    # ask question
    if operator == "<=":
        if sample[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(sample[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive
    else:
        residual_tree = answer
        return classify_sample(sample, residual_tree)


def measure_accuracy(df, tree):

    df["classification"] = df.apply(classify_sample, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["output"]

    accuracy = df["classification_correct"].mean()

    return accuracy


def determine_type_of_feature(df):

    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "output":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types


main()
