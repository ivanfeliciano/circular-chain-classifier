# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from evaluation_measures import *

class CircularChainClassifier(object):
	"""
	Súper clase de clasificador en cadena circular, recibe
	como parámetros un dataframe de pandas que contiene al 
	conjunto de datos y la lista de atributos que son las posibles
	etiquetas de cada ejemplo.
	"""
	def __init__(self, classifier):
		self.classifier = classifier
		self.list_of_classifiers_first_iteration = None
		self.list_of_classifiers_general_case = None
	def train(self, X, labels):
		self.list_of_classifiers_first_iteration = OrderedDict()
		self.list_of_classifiers_general_case = OrderedDict()
		for label in labels:
			self.list_of_classifiers_first_iteration[label] = clone(self.classifier)
			self.list_of_classifiers_general_case[label] = clone(self.classifier)
		self.train_set_x = X
		self.labels = labels
		for label in self.labels:
			self.train_one_link_first_iteration(label)
			self.train_one_link_general_case(label)
	def train_one_link_general_case(self, label):
		X = self.train_set_x.drop(label, axis=1)
		y = self.train_set_x[label]
		# print("label general {} with shape {}".format(label, X.shape))
		self.list_of_classifiers_general_case[label].fit(X, y)

	def train_one_link_first_iteration(self, label):
		X = self.drop_not_depend_on_columns(self.train_set_x, label)
		y = self.train_set_x[label]
		# print("label first {} with shape {}".format(label, X.shape))
		self.list_of_classifiers_first_iteration[label].fit(X, y)

	def drop_not_depend_on_columns(self, X, label):
		label_index = self.labels.index(label)
		if label_index + 1 == len(self.labels):
			return X.drop(label, axis=1)
		labels_to_drop = self.labels[label_index:]
		X = X.drop(labels_to_drop, axis=1)
		return X

	def classify(self, X, steps=10):
		X_hat = X.drop(self.labels, axis=1)
		y_pred_df = None
		i = 1
		print("Iteration {}".format(i))
		for classifier_f in self.list_of_classifiers_first_iteration:
			y_pred = self.list_of_classifiers_first_iteration[classifier_f].predict(X_hat)
			y_pred_dict = OrderedDict([(classifier_f, y_pred)])
			y_pred_df = pd.DataFrame.from_dict(y_pred_dict)
			X_hat = pd.concat([X_hat, y_pred_df], axis=1)
			y = X[classifier_f]
		print("GAcc = {}".format(global_accuracy(X[self.labels], X_hat[self.labels])))
		print("MAcc = {}".format(mean_accuracy(X[self.labels], X_hat[self.labels])))
		print("MLAcc = {}".format(multilabel_accuracy(X[self.labels], X_hat[self.labels])))
		print("FMeasure = {}".format(f_measure(X[self.labels], X_hat[self.labels])))
		while i < steps:
			i += 1
			print("Iteration {}".format(i))
			for classifier_label in self.list_of_classifiers_general_case:
				X_hat = X_hat.drop(classifier_label, axis=1)
				y = X[classifier_label] 
				y_pred = self.list_of_classifiers_general_case[classifier_label].predict(X_hat)
				y_pred_dict = OrderedDict([(classifier_label, y_pred)])
				y_pred_df = pd.DataFrame.from_dict(y_pred_dict)
				X_hat = pd.concat([X_hat, y_pred_df], axis=1)
			print("GAcc = {}".format(global_accuracy(X[self.labels], X_hat[self.labels])))
			print("MAcc = {}".format(mean_accuracy(X[self.labels], X_hat[self.labels])))
			print("MLAcc = {}".format(multilabel_accuracy(X[self.labels], X_hat[self.labels])))
			print("FMeasure = {}".format(f_measure(X[self.labels], X_hat[self.labels])))
if __name__ == '__main__':
	main()