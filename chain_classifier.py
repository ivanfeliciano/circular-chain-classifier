# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from evaluation_measures import global_accuracy


class ChainClassifier(object):
	"""docstring for ChainClassifier"""
	def __init__(self, classifier):
		self.classifier = classifier
		self.cc_classifiers = None
	def train(self, X, labels):
		self.cc_classifiers = OrderedDict()
		self.labels = labels
		for label in self.labels:
			self.cc_classifiers[label] = clone(self.classifier)
		self.train_set = X
		for label in self.labels:
			label_index = self.labels.index(label)
			labels_to_drop = self.labels[label_index:]
			X_train = X.drop(labels_to_drop, axis=1)
			y = self.train_set[label]
			self.cc_classifiers[label].fit(X_train, y)
	def classify(self, X):
		X_hat = X.drop(self.labels, axis=1)
		for label in self.cc_classifiers:
			y = X[label]
			y_pred = self.cc_classifiers[label].predict(X_hat)
			y_pred_dict = OrderedDict([(label, y_pred)])
			y_pred_df = pd.DataFrame.from_dict(y_pred_dict)
			X_hat = pd.concat([X_hat, y_pred_df], axis=1)
		print("GAcc = {}".format(global_accuracy(X[self.labels], X_hat[self.labels])))
	def save(self):
		pass
	def load(self):
		pass
