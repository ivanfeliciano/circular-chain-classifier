# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
from evaluation_measures import *
from multilabel_classifier import MultilabelClassifier
from scipy.sparse import hstack, csr, issparse


class CircularChainClassifier(MultilabelClassifier):
	"""
	Clase de clasificador en cadena circular, recibe
	como parámetros un dataframe de pandas que contiene al 
	conjunto de datos y la lista de atributos que son las posibles
	etiquetas de cada ejemplo.
	"""
	def train(self, X, labels, number_of_iterations=3, k=10, labels_df=None):
		self.classifiers = OrderedDict()
		self.labels = labels
		self.k = k
		for label in self.labels:
			self.classifiers[label] = clone(self.classifier)
		if issparse(X):
			self.train_text(X, number_of_iterations, labels_df)
			return 0

		training_set = X.copy()
		outputs_for_eval = X.copy()

		for label in self.labels[1:]:
			training_set[label] = 1
		counter = 0
		
		while counter < number_of_iterations:
			counter += 1
			for label in self.labels:
				y_true = X[label]
				# training_set.sort_index(inplace=True)
				training_set = training_set.drop(label, axis=1)
				# cv_scores = np.array([])
				# classifier_cv_outputs = np.array([])
				# training_set, y_true = shuffle(training_set, y_true)
				# print(training_set.head())
				# print(y_true.head())
				k_folds = StratifiedKFold(n_splits=k)
				# print(k_folds.split(training_set, y_true))
				for train_index, test_index in k_folds.split(training_set, y_true):
					x_train, x_test = training_set.iloc[train_index,:], training_set.iloc[test_index,:]
					y_train, y_test = y_true.iloc[train_index], y_true.iloc[test_index]
					# print(y_train.value_counts())
					# print(y_test.value_counts())
					self.classifiers[label].partial_fit(x_train, y_train, classes=np.unique(y_train))
					y_pred = self.classifiers[label].predict(x_test)
					# cv_scores = np.append(cv_scores, accuracy_score(y_test, y_pred))
					# classifier_cv_outputs = np.append(classifier_cv_outputs, y_pred)
					outputs_for_eval.loc[test_index, label] = y_pred
				# print(cv_scores)
				# training_set[label] = classifier_cv_outputs.astype(int)
				training_set[label] = outputs_for_eval[label]
			# print("Iteration {}".format(counter))
			self.update_eval_measures(X, outputs_for_eval)
			# self.print_mean_and_std_measures()
		self.print_mean_and_std_measures()
	def train_text(self, X, number_of_iterations, labels_df):
		training_set = X.copy()
		predictions = labels_df.copy()

		for label in labels_df:
			ones = np.ones_like(labels_df[label])
			training_set = hstack((training_set, np.array(ones)[:, None]))
			print(training_set.shape)

	def classify(self, X):
		test_set = X.copy()
		for label in self.labels[1:]:
			test_set[label] = 1
		for label in self.labels:
			y_true = X[label]
			test_set = test_set.drop(label, axis=1)
			y_pred = self.classifiers[label].predict(test_set)
			test_set[label] = y_pred.astype(int)
		print("GAcc = {}".format(global_accuracy(X[self.labels], test_set[self.labels])))
		print("MAcc = {}".format(mean_accuracy(X[self.labels], test_set[self.labels])))
		print("MLAcc = {}".format(multilabel_accuracy(X[self.labels], test_set[self.labels])))
		print("FMeasure = {}".format(f_measure(X[self.labels], test_set[self.labels])))

def main():
	pass

if __name__ == '__main__':
	main()