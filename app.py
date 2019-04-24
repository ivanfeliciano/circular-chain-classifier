import math
import json
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from circular_chain_classifier_sucar import CircularChainClassifier
from binary_relevance import BinaryRelevance	
from chain_classifier import ChainClassifier
import itertools

data_flag = arff.loadarff('./flags/flags-train.arff')
train_df_flags = pd.DataFrame(data_flag[0])
data_flag = arff.loadarff('./flags/flags-test.arff')
test_df_flags = pd.DataFrame(data_flag[0])
labels = ['green', 'red','blue','yellow','white','black','orange']


le = LabelEncoder()
train_df_flags = train_df_flags[train_df_flags.columns[:]].apply(le.fit_transform)
test_df_flags = test_df_flags[test_df_flags.columns[:]].apply(le.fit_transform)

CCC = CircularChainClassifier(LinearSVC(max_iter=5000))
BR = BinaryRelevance(LinearSVC(max_iter=5000))
CC = ChainClassifier(LinearSVC(max_iter=5000))

print("CCC")
# CCC.train(train_df_flags, labels, number_of_iterations=5, k=5)
CCC.train(train_df_flags, labels)
CCC.classify(test_df_flags)
print("BR")
BR.train(train_df_flags, labels)
BR.classify(test_df_flags)
print("CC")
CC.train(train_df_flags, labels)
CC.classify(test_df_flags)

# for permutation_labels in list(itertools.permutations(labels)):
# 	print(permutation_labels)
	# CCC.train(train_df_flags, list(permutation_labels))
	# CCC.classify(test_df_flags)
	# BR.train(train_df_flags, list(permutation_labels))
	# BR.classify(test_df_flags)
	# CC.train(train_df_flags, list(permutation_labels))
	# CC.classify(test_df_flags)

