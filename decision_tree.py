# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu).
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package.
import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

'''
    Student Name: Kautil Reddy D
    NetID: krd180000
'''
# A simple utility class to load data sets for this assignment
class DataSet:
    def __init__(self, data_set, column_count=6, delimiter=' ', label_column=0):
        """
        Initialize a data set and load both training and test data
        DO NOT MODIFY THIS FUNCTION
        """
        self.name = data_set
        self.column_count = column_count
        self.delimiter = delimiter

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]  # training example count
        self.num_test = self.examples['test'].shape[0]  # test example count
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        """
        Load a training set of the specified type (train/test). Returns None if either the training or test files were
        not found. NOTE: This is hard-coded to use only the first seven columns, and will not work with all data sets.
        DO NOT MODIFY THIS FUNCTION
        """
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0,
                                          usecols=range(self.column_count), dtype=int, delimiter=self.delimiter)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


def weighted_length(w):
    return sum(w)


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    value_to_index_dict = dict()
    for (idx, v) in enumerate(x):
        if v not in value_to_index_dict:
            value_to_index_dict[v] = list()
        value_to_index_dict[v].append(idx)
    return value_to_index_dict


def entropy(y, weights):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    partitioned_y = partition(y)
    entropy_y = 0
    total_values = weighted_length(weights)
    for values in partitioned_y.values():
        weights_for_values = [weights[i] for i in values]
        prob_of_value = (weighted_length(weights_for_values))/total_values
        entropy_y -= prob_of_value*np.log2(prob_of_value)
    return entropy_y


def mutual_information(x, y, weights, entropy_y=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    if entropy_y is None:
        entropy_y = entropy(y, weights)
    partitioned_x = partition(x)
    total_values = weighted_length(weights)
    entropy_x = 0
    for val in partitioned_x.keys():
        labels_for_val = [y[i] for i in partitioned_x[val]]
        weights_for_val = [weights[i] for i in partitioned_x[val]]
        entropy_x += (weighted_length(weights_for_val)/total_values)*entropy(labels_for_val, weights_for_val)
    return entropy_y - entropy_x


def majority_label(label_count_map):
    max_count = -1
    major_label = None
    for label in label_count_map:
        elements_with_label_count = weighted_length(label_count_map[label])
        if elements_with_label_count > max_count:
            max_count = elements_with_label_count
            major_label = label
    return major_label


def get_attribute_value_pairs(attributes, x):
    attribute_values = dict()
    for attribute in attributes:
        values = set(x[:, attribute])
        attribute_values[attribute] = values
    return attribute_values


def id3(x, y, attributes, max_depth, weights: list =None, attribute_values: dict=None, depth: int=0) -> dict:
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of attributes
    to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attributes is empty (there is nothing to split on), then return the most common value of y
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y
    Otherwise the algorithm selects the next best attribute using INFORMATION GAIN as the splitting criterion and
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    See https://gkunapuli.github.io/files/cs6375/04-DecisionTrees.pdf (Slide 18) for more details.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current level.
    * The subtree itself can be nested dictionary, or a single label (leaf node).

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1): 1,
     (4, 2): {(3, 1): 0,
              (3, 2): 0,
              (3, 3): 0},
     (4, 3): {(5, 1): 0,
              (5, 2): 0},
     (4, 4): {(0, 1): 0,
              (0, 2): 0,
              (0, 3): 1}}
    """
    if len(y) == 0:
        raise Exception("No data passed")
    if weights is None:
        weights = [1/len(y)]*len(y)
    if attribute_values is None:
        attribute_values = get_attribute_value_pairs(attributes, x)
    label_map = partition(y)
    if len(label_map.keys()) == 1:  # congrats all the examples have same label!
        return (label_map.popitem())[0]

    if len(attributes) == 0 or depth == max_depth:  # majority voted label is selected.
        return majority_label(label_map)

    root = dict()
    max_info_gain = -1
    max_gain_pair = None
    entropy_y = entropy(y, weights)
    for attribute in attributes:
        for attr_value in attribute_values[attribute]:
            attr_column = convert_column_to_dual_values(x[:, attribute], attr_value)
            current_gain = mutual_information(attr_column, y, weights, entropy_y)
            if current_gain > max_info_gain:
                max_info_gain = current_gain
                max_gain_pair = (attribute, attr_value)
    attribute_values[max_gain_pair[0]].remove(max_gain_pair[1])
    subset_x = {True: [], False: []}
    subset_y = {True: [], False: []}
    subset_weights = {True: [], False: []}
    attr_subset_true = list(attributes)
    attr_subset_true.remove(max_gain_pair[0])
    subset_attr = {True: attr_subset_true, False: list(attributes)}
    root_attr = max_gain_pair[0]
    root_val = max_gain_pair[1]
    for i in range(len(y)):
        key = x[i, root_attr] == root_val
        subset_x[key].append(x[i])
        subset_weights[key].append(weights[i])
        subset_y[key].append(y[i])
    for key in [True, False]:
        root[(root_attr, root_val, key)] = id3(np.asarray(subset_x[key]), np.asarray(subset_y[key]), subset_attr[key],
                                               max_depth, subset_weights[key], copy.deepcopy(attribute_values), depth+1)
    root[(max_gain_pair, 'default')] = majority_label(label_map)
    return root


def convert_column_to_dual_values(column, key_value):
    result = []
    for i in range(len(column)):
        result.append(1 if column[i] == key_value else 0)
    return result


def predict_example_tree(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0][0]
    attribute_val = list(tree.keys())[0][1]
    key = (attribute, attribute_val, x[attribute] == attribute_val)
    if key in tree:
        return predict_example_tree(x, tree[key])
    else:
        if (attribute, 'default') not in tree:
            return -1
        else:
            return tree[attribute, 'default']


def predict_example(x, h_ens):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    results = []
    weights = set()
    weight_sum = 0
    for i in range(len(h_ens)):
        tree = h_ens[i][1]
        tree_weight = h_ens[i][0]
        weight_sum += tree_weight
        weights.add(tree_weight)
        results.append(tree_weight*predict_example_tree(x, tree))
    if len(weights) == 1:
        return majority_label(partition(results))
    pred = sum(results)
    pred /= weight_sum
    return 0 if pred < 0.5 else 1


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    length = len(y_true)
    error = 0
    for i in range(len(y_true)):
        if y_pred is None or y_true[i] != y_pred[i]:
            error += 1
    return (error/length)*100


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        if len(split_criterion) > 2:
            print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))
        else:
            print('+-- [SPLIT: x{0} = default]'.format(split_criterion[0]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def compute_confusion(y_true, y_predicted):
    confusion_matrix = [[0, 0], [0, 0]]
    for i, y in enumerate(y_true):
        if y_true[i] == y_predicted[i]:
            if y_true[i] == 1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if y_true[i] == 1:
                confusion_matrix[0][1] += 1
            else:
                confusion_matrix[1][0] += 1
    return confusion_matrix


def bagging(x, y, max_depth, num_trees):
    h_ens = []
    attribute_idx = np.array(range(np.shape(x)[1]))
    for i in range(num_trees):
        bootstrap_sample = sample_with_replacement(x, y)
        tree = id3(bootstrap_sample["data"], bootstrap_sample["labels"], attribute_idx, max_depth)
        h_ens.append((1, tree))
    return h_ens


def sample_with_replacement(x, y):
    sample_idx = np.random.choice(len(y), size=len(y), replace=True)
    sample_set = {"data": [], "labels": []}
    for i in sample_idx:
        sample_set["data"].append(x[i])
        sample_set["labels"].append(y[i])
    sample_set["data"] = np.array(sample_set["data"])
    sample_set["labels"] = np.array(sample_set["labels"])
    return sample_set


def compute_error_rate(y, prediction, w):
    total_weight = sum(w)
    return sum([w[i] for i in range(len(y)) if y[i] != prediction[i]])/total_weight


def reweigh_examples(y, predictions, w, alpha_m):
    norm_factor = 0
    w_new = [None]*len(w)
    for i in range(len(w)):
        exponent = 0
        if y[i] != predictions[i]:
            exponent = alpha_m
        w_i = w[i]*np.exp(exponent)
        norm_factor += w_i
        w_new[i] = w_i
    return list(map(lambda x: x/norm_factor, w_new))


def boosting(x, y, max_depth, num_stumps):
    n = len(y)
    w = [1 / n] * n  # weights
    h_ens = []
    attributes = range(np.shape(x)[1])
    total_classes = len(np.unique(y))
    for i in range(num_stumps):
        tree = id3(x, y, attributes, max_depth, w)
        predictions = [predict_example_tree(x[i], tree) for i in range(n)]
        error_rate = compute_error_rate(y, predictions, w)
        alpha_i = np.log((1-error_rate)/error_rate)/2 + np.log(total_classes - 1)
        w = reweigh_examples(y, predictions, w, alpha_i)
        h_ens.append((alpha_i, tree))
    return h_ens


# code for different experiments on bagging and boosting

def bag_them_models(data):
    print("\nBagging: ")
    for d in [3, 5]:
        for k in [10, 20]:
            h_ens = bagging(data.examples['train'], data.labels['train'], d, k)
            trn_pred, tst_pred = get_predictions(data, h_ens)
            print_errors(data, trn_pred, tst_pred, d, k)
    print("\n\n")


def boost_them_models(data):
    print("\nBoosting: ")
    for k in [20, 40]:
        for d in [1, 2]:
            h_ens = boosting(data.examples["train"], data.labels["train"], d, k)
            trn_pred, tst_pred = get_predictions(data, h_ens)
            print_errors(data, trn_pred, tst_pred, d, k)
    print("\n\n")


def bag_and_boost_scikit(data):
    print("\nBagging SciKit Learn")
    for d in [3, 5]:
        tree = DecisionTreeClassifier(max_depth=d)
        for k in [10, 20]:
            model = BaggingClassifier(tree, n_estimators=k).fit(data.examples['train'], data.labels['train'])
            trn_pred, tst_pred = get_predictions_scikit(data, model)
            print_errors(data, trn_pred, tst_pred, d, k)
    print("\n\n")

    print("\nBoosting SciKit Learn")
    for d in [1, 2]:
        tree = DecisionTreeClassifier(max_depth=d)
        for k in [20, 40]:
            model = AdaBoostClassifier(tree, n_estimators=k, algorithm='SAMME').fit(data.examples['train'], data.labels['train'])
            trn_pred, tst_pred = get_predictions_scikit(data, model)
            print_errors(data, trn_pred, tst_pred, d, k)
    print("\n\n")


def get_predictions(data, h_ens):
    # Compute the training error
    trn_pred = [predict_example(data.examples['train'][i, :], h_ens) for i in range(data.num_train)]
    # Compute the test error
    tst_pred = [predict_example(data.examples['test'][i, :], h_ens) for i in range(data.num_test)]
    return trn_pred, tst_pred


def get_predictions_scikit(data, model):
    # Compute the training error
    trn_pred = model.predict(data.examples['train'])
    # Compute the test error
    tst_pred = model.predict(data.examples['test'])
    return trn_pred, tst_pred


def test_custom_bagging(data, num_stumps, max_depth):
    h_ens = bagging(data.examples['train'], data.labels['train'], max_depth, num_stumps)
    return get_predictions(data, h_ens)


def test_custom_boost(data, num_stumps, max_depth):
    h_ens = boosting(data.examples["train"], data.labels["train"], max_depth, num_stumps)
    return get_predictions(data, h_ens)


def print_errors(data, trn_pred, tst_pred, max_depth, num_stumps):
    trn_err = compute_error(data.labels['train'], trn_pred)
    tst_err = compute_error(data.labels['test'], tst_pred)

    # print the information
    print('k={3} d={0} trn={1}, tst={2}'.format(max_depth, trn_err, tst_err, num_stumps))
    print('-----------------------------------------------------------')
    confusion = compute_confusion(data.labels['train'], trn_pred)
    print("train confusion ")
    print(confusion[0])
    print(confusion[1])
    confusion = compute_confusion(data.labels['test'], tst_pred)
    print("test confusion ")
    print(confusion[0])
    print(confusion[1])


if __name__ == '__main__':
    dataset = DataSet('mushroom', 22, delimiter=',')
    bag_them_models(dataset)
    boost_them_models(dataset)
    bag_and_boost_scikit(dataset)
