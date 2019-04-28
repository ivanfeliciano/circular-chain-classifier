# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.base import clone
from evaluation_measures import *
from sklearn.model_selection import cross_val_score
from multilabel_classifier import MultilabelClassifier

class BinaryRelevance(MultilabelClassifier):
	"""docstring for BinaryRelevance"""
	def train(self, X, labels, k=10):
		self.k = k
		self.classifiers = OrderedDict()
		self.labels = labels
		k_folds = StratifiedKFold(n_splits=k)
		for label in self.labels:
			self.classifiers[label] = clone(self.classifier)
		training_set = X.drop(labels, axis=1)
		predictions = X.copy()
		for label in self.labels:
			y_true = X[label]
			# scores = cross_val_score(self.classifiers[label], dataset_only_attr, y, cv=5)
			# print(scores)
			classifier_cv_outputs = np.array([])
			k_folds = StratifiedKFold(n_splits=k)
			for train_index, test_index in k_folds.split(training_set, y_true):
				x_train, x_test = training_set.iloc[train_index, :], training_set.iloc[test_index, :]
				y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
				self.classifiers[label].partial_fit(x_train, y_train, classes=np.unique(y_train))
				y_pred = self.classifiers[label].predict(x_test)
				classifier_cv_outputs = np.append(classifier_cv_outputs, y_pred)
			predictions[label] = classifier_cv_outputs.astype(int)
		self.update_eval_measures(X, predictions)
		self.print_mean_and_std_measures()
	def classify(self, X):
		X_hat = X.drop(self.labels, axis=1)
		dataset_only_attr = X.drop(self.labels, axis=1)
		y_pred_df = None
		for label in self.classifiers:
			y = X[label] 
			y_pred = self.classifiers[label].predict(dataset_only_attr)
			y_pred_dict = OrderedDict([(label, y_pred)])
			y_pred_df = pd.DataFrame.from_dict(y_pred_dict)
			X_hat = pd.concat([X_hat, y_pred_df], axis=1)
		print("GAcc = {}".format(global_accuracy(X[self.labels], X_hat[self.labels])))
		print("MAcc = {}".format(mean_accuracy(X[self.labels], X_hat[self.labels])))
		print("MLAcc = {}".format(multilabel_accuracy(X[self.labels], X_hat[self.labels])))
		print("FMeasure = {}".format(f_measure(X[self.labels], X_hat[self.labels])))
	def save(self):
		pass
	def load(self):
		pass
