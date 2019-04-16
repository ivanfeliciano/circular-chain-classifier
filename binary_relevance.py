# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from evaluation_measures import global_accuracy


class BinaryRelevance(object):
	"""docstring for BinaryRelevance"""
	def __init__(self, classifier):
		self.classifier = classifier
		self.br_classifiers = OrderedDict()
	def train(self, X, labels):
		self.labels = labels
		for label in self.labels:
			self.br_classifiers[label] = clone(self.classifier)
		self.train_set = X
		dataset_only_attr = X.drop(labels, axis=1)
		for label in self.labels:
			y = self.train_set[label]
			self.br_classifiers[label].fit(dataset_only_attr, y)
	def classify(self, X):
		X_hat = X.drop(self.labels, axis=1)
		dataset_only_attr = X.drop(self.labels, axis=1)
		y_pred_df = None
		for label in self.br_classifiers:
			y = X[label] 
			y_pred = self.br_classifiers[label].predict(dataset_only_attr)
			y_pred_dict = OrderedDict([(label, y_pred)])
			y_pred_df = pd.DataFrame.from_dict(y_pred_dict)
			X_hat = pd.concat([X_hat, y_pred_df], axis=1)
		print("GAcc = {}".format(global_accuracy(X[self.labels], X_hat[self.labels])))
	def save(self):
		pass
	def load(self):
		pass
