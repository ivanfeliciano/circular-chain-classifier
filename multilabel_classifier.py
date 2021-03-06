# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from evaluation_measures import *

class MultilabelClassifier(object):
	"""docstring for MultilabelClassifier"""
	def __init__(self, classifier):
		self.classifier = classifier
		self.classifiers = None
		self.gacc = np.array([])
		self.macc = np.array([])
		self.mlacc = np.array([])
		self.f_measure = np.array([])
		self.labels = None
		self.k = 10
		self.table_measures = '<div style="overflow-x:auto;"> <table> <tr> <th>GAccMean</th><th>GAcc_std</th><th>MAccMean</th><th>MAcc_std</th><th>MLAccMean</th><th>MLAcc_std</th><th>F-measureMean</th><th>F-measure_std</th> </tr>'
		self.last_results = None
	def print_mean_and_std_measures(self):
		# print(self.gacc)
		# print(self.macc)
		# print(self.mlacc)
		# print(self.f_measure)
		# gacc_mean, gacc_std = np.mean(self.gacc), np.std(self.gacc)
		# print("GAcc: mean = {}, std = {}".format(np.mean(self.gacc), np.std(self.gacc)))
		# macc_mean, macc_std = np.mean(self.macc), np.std(self.macc)
		# print("MAcc: mean = {}, std = {}".format(np.mean(self.macc), np.std(self.macc)))
		# mlacc_mean, mlacc_std = np.mean(self.mlacc), np.std(self.mlacc)
		# print("MLAcc: mean = {}, std = {}".format(np.mean(self.mlacc), np.std(self.mlacc)))
		# f_measure_mean, f_measure_std = np.mean(self.f_measure), np.std(self.f_measure)
		# print("FMeasure: mean = {}, std = {}".format(np.mean(self.f_measure), np.std(self.f_measure)))

		self.table_measures += '</table></div>'
		print(self.table_measures)

	def update_eval_measures(self, true, predictions, print_current_vals=False):
		true_outputs_chunks = []
		predictions_chunks = []
		for g, df in predictions.groupby(np.arange(len(predictions)) // self.k):
			predictions_chunks.append(df)
		for g, df in true.groupby(np.arange(len(true)) // self.k):
			true_outputs_chunks.append(df)
		for t, p in zip(true_outputs_chunks, predictions_chunks):
			self.gacc = np.append(self.gacc, global_accuracy(t[self.labels], p[self.labels]))
			self.macc = np.append(self.macc, mean_accuracy(t[self.labels], p[self.labels]))
			self.mlacc = np.append(self.mlacc, multilabel_accuracy(t[self.labels], p[self.labels]))
			self.f_measure = np.append(self.f_measure, f_measure(t[self.labels], p[self.labels]))
		gacc_mean, gacc_std = np.mean(self.gacc), np.std(self.gacc)
		macc_mean, macc_std = np.mean(self.macc), np.std(self.macc)
		mlacc_mean, mlacc_std = np.mean(self.mlacc), np.std(self.mlacc)
		f_measure_mean, f_measure_std = np.mean(self.f_measure), np.std(self.f_measure)
		n_digits = 4
		self.last_results = [round(gacc_mean, n_digits), round(gacc_std, n_digits), round(macc_mean, n_digits), round(macc_std, n_digits), round(mlacc_mean, n_digits), round(mlacc_std, 4), round(f_measure_mean, 4), round(f_measure_std, 4)]
		self.table_measures += '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(self.last_results[0], self.last_results[1], self.last_results[2], self.last_results[3], self.last_results[4], self.last_results[5], self.last_results[6], self.last_results[7])
		# if print_current_vals:

			# print("GAcc = {}".format(self.gacc[-1]))
			# print("MAcc = {}".format(self.macc[-1]))
			# print("MLAcc = {}".format(self.mlacc[-1]))
			# print("FMeasure = {}".format(self.f_measure[-1]))