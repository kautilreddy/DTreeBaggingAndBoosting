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
import graphviz as graphviz
import numpy as np
from sklearn import tree

'''
    Student Name: Kautil Reddy D
    NetID: krd180000
'''
# A simple utility class to load data sets for this assignment
class DataSet:
    def __init__(self, data_set, column_count=6):
        """
        Initialize a data set and load both training and test data
        DO NOT MODIFY THIS FUNCTION
        """
        self.name = data_set
        self.column_count = column_count

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
                                          usecols=range(self.column_count), dtype=int)

            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


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


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    partitioned_y = partition(y)
    entropy_y = 0
    total_values = sum([len(partitioned_y[i]) for i in partitioned_y])
    for value in partitioned_y.values():
        prob_of_value = (len(value))/total_values
        entropy_y -= prob_of_value*np.log2(prob_of_value)
    return entropy_y


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    entropy_y = entropy(y)
    partitioned_x = partition(x)
    total_values = len(y)
    entropy_x = 0
    for val in partitioned_x.keys():
        labels_for_val = [y[i] for i in partitioned_x[val]]
        entropy_x += (len(labels_for_val)/total_values)*entropy(labels_for_val)
    return entropy_y - entropy_x


def majority_label(label_count_map):
    max_count = -1
    major_label = None
    for label in label_count_map:
        elements_with_label_count = len(label_count_map[label])
        if elements_with_label_count > max_count:
            max_count = elements_with_label_count
            major_label = label
    return major_label


def id3(x, y, attributes, max_depth, depth=0):
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
    label_map = partition(y)
    if len(label_map.keys()) == 1:  # congrats all the examples have same label!
        return (label_map.popitem())[0]

    if len(attributes) == 0 or depth == max_depth:  # majority voted label is selected.
        return majority_label(label_map)

    root = dict()
    max_info_gain = -1
    max_gain_attr = attributes[0]
    max_gain_attr_data = []
    for value in attributes:
        attr_column = x[:, value]
        current_gain = mutual_information(attr_column, y)
        if current_gain > max_info_gain:
            max_info_gain = current_gain
            max_gain_attr = value
            max_gain_attr_data = attr_column

    partition_on_attr = partition(max_gain_attr_data)
    subset_attributes = np.asarray([x for x in attributes if x != max_gain_attr])
    for value in partition_on_attr:
        subset_x = []
        subset_y = []
        for row_index in partition_on_attr[value]:
            subset_x.append(x[row_index])
            subset_y.append(y[row_index])
        root[(max_gain_attr, value)] = id3(np.asarray(subset_x), np.asarray(subset_y), subset_attributes, max_depth, depth+1)
    root[(max_gain_attr, 'default')] = majority_label(label_map)
    return root


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    if not isinstance(tree, dict):
        return tree
    attribute = list(tree.keys())[0][0]
    key = (attribute, x[attribute])
    if key in tree:
        return predict_example(x, tree[key])
    else:
        if (attribute, 'default') not in tree:
            return -1
        else:
            return tree[attribute, 'default']


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
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

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


def train_and_test_monks():
    # Load a data set
    data = DataSet('monks-1')

    # Get a list of all the attribute indices
    attribute_idx = np.array(range(data.dim))

    decision_tree = id3(data.examples['train'], data.labels['train'], attribute_idx, 1)
    visualize(decision_tree)
    decision_tree = id3(data.examples['train'], data.labels['train'], attribute_idx, 2)
    visualize(decision_tree)

    # Learn a decision tree of depth 3
    for d in range(1, 7):
        print("\nd =" + str(d))
        decision_tree = id3(data.examples['train'], data.labels['train'], attribute_idx, d)
        # Compute the training error
        trn_pred = [predict_example(data.examples['train'][i, :], decision_tree) for i in range(data.num_train)]
        trn_err = compute_error(data.labels['train'], trn_pred)
        confusion = compute_confusion(data.labels['train'], trn_pred)
        print("test confusion ")
        print(confusion)

        # Compute the test error
        tst_pred = [predict_example(data.examples['test'][i, :], decision_tree) for i in range(data.num_test)]
        tst_err = compute_error(data.labels['test'], tst_pred)
        confusion = compute_confusion(data.labels['test'], tst_pred)
        print('d={0} trn={1}, tst={2}'.format(d, trn_err, tst_err))
        print(confusion)

    data_monks1 = DataSet('monks-1')
    clf = tree.DecisionTreeClassifier()
    fitted_tree = clf.fit(data_monks1.examples['train'], data_monks1.labels['train'])
    dot_data = tree.export_graphviz(fitted_tree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("scikit-tree")
    predicted = fitted_tree.predict(data_monks1.examples['test'])
    confusion = compute_confusion(data.labels['test'], predicted)
    print("SciKit Learn confusion")
    print(confusion)


def train_and_test_breast_cancer_recurrence():
    data = DataSet('breast-cancer', 10)
    attribute_idx = np.array(range(data.dim))
    for d in (1, 2, 8):
        decision_tree = id3(data.examples['train'], data.labels['train'], attribute_idx, d)
        visualize(decision_tree)

        # Compute the training error
        trn_pred = [predict_example(data.examples['train'][i, :], decision_tree) for i in range(data.num_train)]
        trn_err = compute_error(data.labels['train'], trn_pred)
        confusion = compute_confusion(data.labels['train'], trn_pred)
        print("test confusion ")
        print(confusion)

        # Compute the test error
        tst_pred = [predict_example(data.examples['test'][i, :], decision_tree) for i in range(data.num_test)]
        tst_err = compute_error(data.labels['test'], tst_pred)
        confusion = compute_confusion(data.labels['test'], tst_pred)
        print('d={0} trn={1}, tst={2}'.format(d, trn_err, tst_err))
        print(confusion)

    clf = tree.DecisionTreeClassifier()
    fitted_tree = clf.fit(data.examples['train'], data.labels['train'])
    predicted = fitted_tree.predict(data.examples['test'])
    trn_pred = fitted_tree.predict(data.examples['train'])
    trn_err = compute_error(data.labels['train'], trn_pred)
    tst_err = compute_error(data.labels['test'], predicted)
    confusion = compute_confusion(data.labels['test'], predicted)
    dot_data = tree.export_graphviz(fitted_tree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("scikit-breast-cancer-tree")
    print('scikit trn={1}, tst={2}'.format(10, trn_err, tst_err))
    print(confusion)


if __name__ == '__main__':
    pass
    train_and_test_monks()
    train_and_test_breast_cancer_recurrence()

