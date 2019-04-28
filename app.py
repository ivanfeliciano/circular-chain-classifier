import math
import json
import itertools

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from circular_chain_classifier import CircularChainClassifier
from binary_relevance import BinaryRelevance	
from chain_classifier import ChainClassifier
from skmultilearn.problem_transform import BinaryRelevance as BRSklearn
from sklearn.metrics import accuracy_score
from evaluation_measures import *


# data_flags = arff.loadarff('./flags/flags-train.arff')
# train_df_flags = pd.DataFrame(data_flags[0])
# data_flags = arff.loadarff('./flags/flags-test.arff')
# test_df_flags = pd.DataFrame(data_flags[0])

data_flags = pd.read_csv('flags.csv').drop('No', axis=1)
labels = ['red','green','blue','yellow','white','black','orange']
# labels = ['green', 'blue', 'black', 'yellow', 'orange', 'white', 'red']
# 2-3-6-4-7-5-1
# labels = ['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']
le = LabelEncoder()
# train_df_flags = train_df_flags[train_df_flags.columns[:]].apply(le.fit_transform)
# test_df_flags = test_df_flags[test_df_flags.columns[:]].apply(le.fit_transform)
# train_df_flags = pd.concat([train_df_flags, test_df_flags])
# print(train_df_flags.shape)

CCC = CircularChainClassifier(MultinomialNB())
BR = BinaryRelevance(MultinomialNB())
CC = ChainClassifier(MultinomialNB())


print("CCC")
CCC.train(data_flags, labels, number_of_iterations=5, k=10)
# CCC.train(data_flags, labels)
# CCC.classify(test_df_flags)
print("BR")
BR.train(data_flags, labels, k=10)
# BR.classify(test_df_flags)
print("CC")
CC.train(data_flags, labels, k=10)
# CC.classify(test_df_flags)

# print("BR sklearn")
# classifier = BRSklearn(MultinomialNB())
# x_train = train_df_flags.drop(labels, axis=1)
# y_train = train_df_flags[labels]
# x_test = test_df_flags.drop(labels, axis=1)
# y_test = test_df_flags[labels]
# classifier.fit(x_train, y_train)
# pred = classifier.predict(x_test)
# pred = pd.DataFrame(pred.todense())
# pred.columns = labels
# print("Accuracy = ", accuracy_score(y_test, pred))
# print("GAcc = {}".format(global_accuracy(y_test, pred)))
# print("MAcc = {}".format(mean_accuracy(y_test, pred)))
# print("MLAcc = {}".format(multilabel_accuracy(y_test, pred)))
# print("FMeasure = {}".format(f_measure(y_test, pred)))


# for permutation_labels in list(itertools.permutations(labels)):
# 	print(permutation_labels)
	# CCC.train(train_df_flags, list(permutation_labels))
	# CCC.classify(test_df_flags)
	# BR.train(train_df_flags, list(permutation_labels))
	# BR.classify(test_df_flags)
	# CC.train(train_df_flags, list(permutation_labels))
	# CC.classify(test_df_flags)

