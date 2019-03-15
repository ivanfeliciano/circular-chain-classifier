# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
class CircularChainClassifier(object):
	"""
	Súper clase de clasificador en cadena circular, recibe
	como parámetros un dataframe de pandas que contiene al 
	conjunto de datos y la lista de atributos que son las posibles
	etiquetas de cada ejemplo.
	"""
	def __init__(self, classifier):
		self.classifier = classifier

	def train(self, X, labels):
		self.list_of_classifiers = { label : self.classifier for label in labels }
		self.train_set_x = X
		self.labels = labels
		self.visited = { label : False for label in labels }
		label_first_classifier = self.labels[0]
		self.list_of_classifiers[label_first_classifier + "_first_iteration"] = self.classifier
		self.train_one_link(label_first_classifier, True)
		for label in self.labels:
			self.train_one_link(label)
	def train_one_link(self, label, weirdo=False):
		print(label)
		if not weirdo:
			X = self.drop_not_depend_on_columns(self.train_set_x, label)
		else:
			X = self.train_set_x.drop(label, axis=1)
		y = self.train_set_x[label]
		self.list_of_classifiers[label].fit(X, y)

	def drop_not_depend_on_columns(self, X, label):
		label_index = self.labels.index(label)
		if label_index + 1 == len(self.labels):
			return X
		labels_to_drop = self.labels[label_index:]
		X = X.drop(labels_to_drop, axis=1)
		return X
	def run(self, X):
		for classifier_label in self.list_of_classifiers:
			label = classifier_label
			print(classifier_label)
			if not "_first_iteration" in classifier_label:
				print("classifier_label")
				X = self.drop_not_depend_on_columns(X, classifier_label)
			else:
				X = X.drop(classifier_label, axis=1)
				label = label.replace('_first_iteration', '')
			y = X[label]
			print(label)
			# y_pred = self.list_of_classifiers[label].predict(X)
			# print("Accuracy score for classifier {} = {}".format(label, accuracy_score(y, y_pred)))
	def classify(self):
		pass
	def evaluation(self):
		pass
