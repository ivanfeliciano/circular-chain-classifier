# -*- coding: utf-8 -*-
def check_shape():
	pass

def global_accuracy(true_classes, predicted_classes):
	p = true_classes.shape[0]
	if true_classes.shape != predicted_classes.shape:
		print("Outputs with different sizes.")
		return 0
	c_and_c_hat = 0
	for index in range(true_classes.shape[0]):
		if true_classes.iloc[index].equals(predicted_classes.iloc[index]):
			c_and_c_hat += 1
	return c_and_c_hat / p

def mean_accuracy(true_classes, predicted_classes):
	p = true_classes.shape[0]
	labels = true_classes.columns
	q = len(labels)
	acc = [0 for i in range(q)]
	if true_classes.shape != predicted_classes.shape:
		print("Outputs with different sizes.")
		return 0
	for u in range(true_classes.shape[0]):
		j = 0
		for label in labels:
			if true_classes.iloc[u][label] == predicted_classes.iloc[u][label]:
				acc[j] += 1
			j += 1
	acc = [d / p for d in acc]
	macc = sum(acc) / q
	return macc

def multilabel_accuracy(true_classes, predicted_classes):
	p = true_classes.shape[0]
	labels = true_classes.columns
	q = len(labels)
	ans = 0
	if true_classes.shape != predicted_classes.shape:
		print("Outputs with different sizes.")
		return 0
	for u in range(true_classes.shape[0]):
		predicted_correct_labels = 0
		total_number_of_labels = 0
		for label in labels:
			if true_classes.iloc[u][label] and predicted_classes.iloc[u][label]:
				predicted_correct_labels += 1
			if true_classes.iloc[u][label] or predicted_classes.iloc[u][label]:
				total_number_of_labels += 1
		ans += (predicted_correct_labels / total_number_of_labels)
	ans /= p
	return ans
def f_measure(true_classes, predicted_classes):
	p = true_classes.shape[0]
	labels = true_classes.columns
	q = len(labels)
	ans = 0
	if true_classes.shape != predicted_classes.shape:
		print("Outputs with different sizes.")
		return 0
	ans = 0
	for u in range(true_classes.shape[0]):
		for label in labels:
			if true_classes.iloc[u][label] and predicted_classes.iloc[u][label]:
				ans += 1
	ans /= (p * q)
	return ans
