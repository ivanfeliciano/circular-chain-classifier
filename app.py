import math
import json
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from circular_chain_classifier import CircularChainClassifier
from sklearn.neighbors import KNeighborsClassifier
from binary_relevance import BinaryRelevance

data_flag = arff.loadarff('./flags/flags-train.arff')
train_df_flags = pd.DataFrame(data_flag[0])
data_flag = arff.loadarff('./flags/flags-test.arff')
test_df_flags = pd.DataFrame(data_flag[0])
labels = ['red','green','blue','yellow','white','black','orange']


le = LabelEncoder()
train_df_flags = train_df_flags[train_df_flags.columns[:]].apply(le.fit_transform)
test_df_flags = test_df_flags[test_df_flags.columns[:]].apply(le.fit_transform)
ccc = CircularChainClassifier(MultinomialNB())
ccc.train(train_df_flags, labels)
ccc.run(test_df_flags)
BR = BinaryRelevance(MultinomialNB())
BR.train(train_df_flags, labels)
BR.classify(test_df_flags)

