import math
import json
import itertools

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from circular_chain_classifier import CircularChainClassifier
from binary_relevance import BinaryRelevance	
from chain_classifier import ChainClassifier
from skmultilearn.problem_transform import BinaryRelevance as BRSklearn
from sklearn.metrics import accuracy_score
from evaluation_measures import *

BASE_DIR = './datasets/'
data_paths = ['flags.csv', 'discrete_scene.csv', 'discrete_CAL500.csv']

all_labels = [
	['green', 'blue', 'black', 'yellow', 'orange', 'white', 'red'],
	["Beach","Sunset","FallFoliage","Field","Mountain","Urban"],
	["Angry-Agressive","NOT-Emotion-Angry-Agressive","Emotion-Arousing-Awakening","NOT-Emotion-Arousing-Awakening","Emotion-Bizarre-Weird","NOT-Emotion-Bizarre-Weird","Emotion-Calming-Soothing","NOT-Emotion-Calming-Soothing","Emotion-Carefree-Lighthearted","NOT-Emotion-Carefree-Lighthearted","Emotion-Cheerful-Festive","NOT-Emotion-Cheerful-Festive","Emotion-Emotional-Passionate","NOT-Emotion-Emotional-Passionate","Emotion-Exciting-Thrilling","NOT-Emotion-Exciting-Thrilling","Emotion-Happy","NOT-Emotion-Happy","Emotion-Laid-back-Mellow","NOT-Emotion-Laid-back-Mellow","Emotion-Light-Playful","NOT-Emotion-Light-Playful","Emotion-Loving-Romantic","NOT-Emotion-Loving-Romantic","Emotion-Pleasant-Comfortable","NOT-Emotion-Pleasant-Comfortable","Emotion-Positive-Optimistic","NOT-Emotion-Positive-Optimistic","Emotion-Powerful-Strong","NOT-Emotion-Powerful-Strong","Emotion-Sad","NOT-Emotion-Sad","Emotion-Tender-Soft","NOT-Emotion-Tender-Soft","Emotion-Touching-Loving","NOT-Emotion-Touching-Loving","Genre--_Alternative","Genre--_Alternative_Folk","Genre--_Bebop","Genre--_Brit_Pop","Genre--_Classic_Rock","Genre--_Contemporary_Blues","Genre--_Contemporary_RandB","Genre--_Cool_Jazz","Genre--_Country_Blues","Genre--_Dance_Pop","Genre--_Electric_Blues","Genre--_Funk","Genre--_Gospel","Genre--_Metal-Hard_Rock","Genre--_Punk","Genre--_Roots_Rock","Genre--_Singer-Songwriter","Genre--_Soft_Rock","Genre--_Soul","Genre--_Swing","Genre-Bluegrass","Genre-Blues","Genre-Country","Genre-Electronica","Genre-Folk","Genre-Hip_Hop-Rap","Genre-Jazz","Genre-Pop","Genre-RandB","Genre-Rock","Genre-World","Instrument_-_Acoustic_Guitar","Instrument_-_Ambient_Sounds","Instrument_-_Backing_vocals","Instrument_-_Bass","Instrument_-_Drum_Machine","Instrument_-_Drum_Set","Instrument_-_Electric_Guitar_(clean)","Instrument_-_Electric_Guitar_(distorted)","Instrument_-_Female_Lead_Vocals","Instrument_-_Hand_Drums","Instrument_-_Harmonica","Instrument_-_Horn_Section","Instrument_-_Male_Lead_Vocals","Instrument_-_Organ","Instrument_-_Piano","Instrument_-_Samples","Instrument_-_Saxophone","Instrument_-_Sequencer","Instrument_-_String_Ensemble","Instrument_-_Synthesizer","Instrument_-_Tambourine","Instrument_-_Trombone","Instrument_-_Trumpet","Instrument_-_Violin-Fiddle","Song-Catchy-Memorable","NOT-Song-Catchy-Memorable","Song-Changing_Energy_Level","NOT-Song-Changing_Energy_Level","Song-Fast_Tempo","NOT-Song-Fast_Tempo","Song-Heavy_Beat","NOT-Song-Heavy_Beat","Song-High_Energy","NOT-Song-High_Energy","Song-Like","NOT-Song-Like","Song-Positive_Feelings","NOT-Song-Positive_Feelings","Song-Quality","NOT-Song-Quality","Song-Recommend","NOT-Song-Recommend","Song-Recorded","NOT-Song-Recorded","Song-Texture_Acoustic","Song-Texture_Electric","Song-Texture_Synthesized","Song-Tonality","NOT-Song-Tonality","Song-Very_Danceable","NOT-Song-Very_Danceable","Usage-At_a_party","Usage-At_work","Usage-Cleaning_the_house","Usage-Driving","Usage-Exercising","Usage-Getting_ready_to_go_out","Usage-Going_to_sleep","Usage-Hanging_with_friends","Usage-Intensely_Listening","Usage-Reading","Usage-Romancing","Usage-Sleeping","Usage-Studying","Usage-Waking_up","Usage-With_the_family","Vocals-Aggressive","Vocals-Altered_with_Effects","Vocals-Breathy","Vocals-Call_and_Response","Vocals-Duet","Vocals-Emotional","Vocals-Falsetto","Vocals-Gravelly","Vocals-High-pitched","Vocals-Low-pitched","Vocals-Monotone","Vocals-Rapping","Vocals-Screaming","Vocals-Spoken","Vocals-Strong","Vocals-Vocal_Harmonies","Genre-Best--_Alternative","Genre-Best--_Classic_Rock","Genre-Best--_Metal-Hard_Rock","Genre-Best--_Punk","Genre-Best--_Soft_Rock","Genre-Best--_Soul","Genre-Best-Blues","Genre-Best-Country","Genre-Best-Electronica","Genre-Best-Folk","Genre-Best-Hip_Hop-Rap","Genre-Best-Jazz","Genre-Best-Pop","Genre-Best-RandB","Genre-Best-Rock","Genre-Best-World","Instrument_-_Acoustic_Guitar-Solo","Instrument_-_Electric_Guitar_(clean)-Solo","Instrument_-_Electric_Guitar_(distorted)-Solo","Instrument_-_Female_Lead_Vocals-Solo","Instrument_-_Harmonica-Solo","Instrument_-_Male_Lead_Vocals-Solo","Instrument_-_Piano-Solo","Instrument_-_Saxophone-Solo","Instrument_-_Trumpet-Solo"]
]

# data_flags = arff.loadarff('./flags/flags-train.arff')
# train_df_flags = pd.DataFrame(data_flags[0])
# data_flags = arff.loadarff('./flags/flags-test.arff')
# test_df_flags = pd.DataFrame(data_flags[0])

# labels = ['red','green','blue','yellow','white','black','orange']
# labels = ['green', 'blue', 'black', 'yellow', 'orange', 'white', 'red']
# 2-3-6-4-7-5-1
# labels = ['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']
# le = LabelEncoder()
# train_df_flags = train_df_flags[train_df_flags.columns[:]].apply(le.fit_transform)
# test_df_flags = test_df_flags[test_df_flags.columns[:]].apply(le.fit_transform)
# train_df_flags = pd.concat([train_df_flags, test_df_flags])
# print(train_df_flags.shape)


for i in range(len(data_paths)):
	CCC = CircularChainClassifier(MultinomialNB())
	BR = BinaryRelevance(MultinomialNB())
	CC = ChainClassifier(MultinomialNB())

	data_path = BASE_DIR + data_paths[i]
	labels = all_labels[i]
	data = pd.read_csv(data_path)
	table_color='#4CAF50'
	print('<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>table{{border-collapse: collapse; font-family: "Trebuchet MS", Arial, Helvetica, sans-serif; width: 50%; margin: 0px auto;}}h2{{font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;}}th, td{{text-align: left; padding: 8px; border: 1px solid #ddd;}}tr:nth-child(even){{background-color: #f2f2f2;}}tr:hover{{background-color: #ddd;}}th{{padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: {}; color: white;}}</style></head><body>'.format(table_color))
	print("<h2>{}</h2>".format(data_path))
	print("<h2>{}</h2>".format(labels))
	print("<h2>CCC</h2>")
	CCC.train(data, labels, number_of_iterations=25, k=10)
	print("<h2>BR</h2>")
	BR.train(data, labels, k=10)
	print("<h2>CC</h2>")
	CC.train(data, labels, k=10)
	print('</body></html>')
