# -*- coding: utf-8 -*-
import pandas as pd

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