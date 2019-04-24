# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import KFold
from evaluation_measures import global_accuracy
from sklearn.utils import shuffle
from evaluation_measures import *

class CircularChainClassifier(object):
	"""
	Clase de clasificador en cadena circular, recibe
	como parámetros un dataframe de pandas que contiene al 
	conjunto de datos y la lista de atributos que son las posibles
	etiquetas de cada ejemplo.
	"""
	def __init__(self, classifier):
		self.classifier = classifier
		self.ccc_classifiers = None
	def train(self, X, labels, number_of_iterations=3, k=5):
		self.ccc_classifiers = OrderedDict()
		self.labels = labels
		training_set = X.copy()
		k_folds = KFold(n_splits=k)
		for label in self.labels:
			self.ccc_classifiers[label] = clone(self.classifier)
		for label in self.labels[1:]:
			training_set[label] = 1
		counter = 0
		while counter < number_of_iterations:
			counter += 1
			for label in self.labels:
				y_true = X[label]
				training_set.sort_index(inplace=True)
				training_set = training_set.drop(label, axis=1)
				cv_scores = np.array([])
				classifier_cv_outputs = np.array([])
				training_set, y_true = shuffle(training_set, y_true)
				# print(training_set.head())
				# print(y_true.head())
				for train_index, test_index in k_folds.split(training_set):
					x_train, x_test = training_set.iloc[train_index,:], training_set.iloc[test_index,:]
					y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
					self.ccc_classifiers[label].partial_fit(x_train, y_train, classes=np.unique(y_train))
					y_pred = self.ccc_classifiers[label].predict(x_test)
					cv_scores = np.append(cv_scores, accuracy_score(y_test, y_pred))
					classifier_cv_outputs = np.append(classifier_cv_outputs, y_pred)
				# print(cv_scores)
				training_set[label] = classifier_cv_outputs.astype(int)
			print("Iteration {}".format(counter))
			print("GAcc = {}".format(global_accuracy(X[self.labels], training_set[self.labels])))
			print("MAcc = {}".format(mean_accuracy(X[self.labels], training_set[self.labels])))
			print("MLAcc = {}".format(multilabel_accuracy(X[self.labels], training_set[self.labels])))
			print("FMeasure = {}".format(f_measure(X[self.labels], training_set[self.labels])))

	def classify(self, X):
		test_set = X.copy()
		for label in self.labels[1:]:
			test_set[label] = 1
		for label in self.labels:
			y_true = X[label]
			test_set = test_set.drop(label, axis=1)
			y_pred = self.ccc_classifiers[label].predict(test_set)
			test_set[label] = y_pred.astype(int)
		print("GAcc = {}".format(global_accuracy(X[self.labels], test_set[self.labels])))
		print("MAcc = {}".format(mean_accuracy(X[self.labels], test_set[self.labels])))
		print("MLAcc = {}".format(multilabel_accuracy(X[self.labels], test_set[self.labels])))
		print("FMeasure = {}".format(f_measure(X[self.labels], test_set[self.labels])))

def main():
	pass

if __name__ == '__main__':
	main()