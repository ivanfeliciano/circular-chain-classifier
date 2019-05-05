import logging
from time import time
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
# data_paths = ['flags.csv', 'discrete_scene.csv', 'discrete_emotions.csv']
data_paths = ['flags.csv',]
# data_paths = ['discrete_emotions.csv']

all_labels = [
	list(itertools.permutations(['red', 'green', 'blue', 'yellow', 'white', 'black', 'orange'])),
	# list(itertools.permutations(['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban'])),
	# list(itertools.permutations(["amazed-suprised","happy-pleased", "relaxing-calm", "quiet-still", "sad-lonely", "angry-aggresive"]))
]
# label_order_name = ["Original", "ScoreBR", "Rand1", "Rand2"]
# all_labels = [[
# 	['red', 'green', 'blue', 'yellow', 'white', 'black', 'orange'], #Original
# 	["red", "orange", "white", "yellow", "black", "blue", "green"],	#Ordenados score BR
# 	["yellow", "red", "green", "blue", "white", "black", "orange"], # Rand 1
# 	["white", "green", "black", "orange", "red", "blue", "yellow"] # Rand 2
# 	],
# 	[
# 	['Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban'],
# 	["Sunset", "Field", "Beach", "Urban", "FallFoliage", "Mountain"],
# 	["Beach", "Sunset", "Mountain", "FallFoliage", "Urban", "Field"],
# 	["Field", "Beach", "Sunset", "Mountain", "Urban", "FallFoliage"]
# 	],
# 	[
# 	['Angry-Agressive', 'NOT-Emotion-Angry-Agressive', 'Emotion-Arousing-Awakening', 'NOT-Emotion-Arousing-Awakening', 'Emotion-Bizarre-Weird', 'NOT-Emotion-Bizarre-Weird', 'Emotion-Calming-Soothing', 'NOT-Emotion-Calming-Soothing', 'Emotion-Carefree-Lighthearted', 'NOT-Emotion-Carefree-Lighthearted', 'Emotion-Cheerful-Festive', 'NOT-Emotion-Cheerful-Festive', 'Emotion-Emotional-Passionate', 'NOT-Emotion-Emotional-Passionate', 'Emotion-Exciting-Thrilling', 'NOT-Emotion-Exciting-Thrilling', 'Emotion-Happy', 'NOT-Emotion-Happy', 'Emotion-Laid-back-Mellow', 'NOT-Emotion-Laid-back-Mellow', 'Emotion-Light-Playful', 'NOT-Emotion-Light-Playful', 'Emotion-Loving-Romantic', 'NOT-Emotion-Loving-Romantic', 'Emotion-Pleasant-Comfortable', 'NOT-Emotion-Pleasant-Comfortable', 'Emotion-Positive-Optimistic', 'NOT-Emotion-Positive-Optimistic', 'Emotion-Powerful-Strong', 'NOT-Emotion-Powerful-Strong', 'Emotion-Sad', 'NOT-Emotion-Sad', 'Emotion-Tender-Soft', 'NOT-Emotion-Tender-Soft', 'Emotion-Touching-Loving', 'NOT-Emotion-Touching-Loving', 'Genre--_Alternative', 'Genre--_Alternative_Folk', 'Genre--_Bebop', 'Genre--_Brit_Pop', 'Genre--_Classic_Rock', 'Genre--_Contemporary_Blues', 'Genre--_Contemporary_RandB', 'Genre--_Cool_Jazz', 'Genre--_Country_Blues', 'Genre--_Dance_Pop', 'Genre--_Electric_Blues', 'Genre--_Funk', 'Genre--_Gospel', 'Genre--_Metal-Hard_Rock', 'Genre--_Punk', 'Genre--_Roots_Rock', 'Genre--_Singer-Songwriter', 'Genre--_Soft_Rock', 'Genre--_Soul', 'Genre--_Swing', 'Genre-Bluegrass', 'Genre-Blues', 'Genre-Country', 'Genre-Electronica', 'Genre-Folk', 'Genre-Hip_Hop-Rap', 'Genre-Jazz', 'Genre-Pop', 'Genre-RandB', 'Genre-Rock', 'Genre-World', 'Instrument_-_Acoustic_Guitar', 'Instrument_-_Ambient_Sounds', 'Instrument_-_Backing_vocals', 'Instrument_-_Bass', 'Instrument_-_Drum_Machine', 'Instrument_-_Drum_Set', 'Instrument_-_Electric_Guitar_(clean)', 'Instrument_-_Electric_Guitar_(distorted)', 'Instrument_-_Female_Lead_Vocals', 'Instrument_-_Hand_Drums', 'Instrument_-_Harmonica', 'Instrument_-_Horn_Section', 'Instrument_-_Male_Lead_Vocals', 'Instrument_-_Organ', 'Instrument_-_Piano', 'Instrument_-_Samples', 'Instrument_-_Saxophone', 'Instrument_-_Sequencer', 'Instrument_-_String_Ensemble', 'Instrument_-_Synthesizer', 'Instrument_-_Tambourine', 'Instrument_-_Trombone', 'Instrument_-_Trumpet', 'Instrument_-_Violin-Fiddle', 'Song-Catchy-Memorable', 'NOT-Song-Catchy-Memorable', 'Song-Changing_Energy_Level', 'NOT-Song-Changing_Energy_Level', 'Song-Fast_Tempo', 'NOT-Song-Fast_Tempo', 'Song-Heavy_Beat', 'NOT-Song-Heavy_Beat', 'Song-High_Energy', 'NOT-Song-High_Energy', 'Song-Like', 'NOT-Song-Like', 'Song-Positive_Feelings', 'NOT-Song-Positive_Feelings', 'Song-Quality', 'NOT-Song-Quality', 'Song-Recommend', 'NOT-Song-Recommend', 'Song-Recorded', 'NOT-Song-Recorded', 'Song-Texture_Acoustic', 'Song-Texture_Electric', 'Song-Texture_Synthesized', 'Song-Tonality', 'NOT-Song-Tonality', 'Song-Very_Danceable', 'NOT-Song-Very_Danceable', 'Usage-At_a_party', 'Usage-At_work', 'Usage-Cleaning_the_house', 'Usage-Driving', 'Usage-Exercising', 'Usage-Getting_ready_to_go_out', 'Usage-Going_to_sleep', 'Usage-Hanging_with_friends', 'Usage-Intensely_Listening', 'Usage-Reading', 'Usage-Romancing', 'Usage-Sleeping', 'Usage-Studying', 'Usage-Waking_up', 'Usage-With_the_family', 'Vocals-Aggressive', 'Vocals-Altered_with_Effects', 'Vocals-Breathy', 'Vocals-Call_and_Response', 'Vocals-Duet', 'Vocals-Emotional', 'Vocals-Falsetto', 'Vocals-Gravelly', 'Vocals-High-pitched', 'Vocals-Low-pitched', 'Vocals-Monotone', 'Vocals-Rapping', 'Vocals-Screaming', 'Vocals-Spoken', 'Vocals-Strong', 'Vocals-Vocal_Harmonies', 'Genre-Best--_Alternative', 'Genre-Best--_Classic_Rock', 'Genre-Best--_Metal-Hard_Rock', 'Genre-Best--_Punk', 'Genre-Best--_Soft_Rock', 'Genre-Best--_Soul', 'Genre-Best-Blues', 'Genre-Best-Country', 'Genre-Best-Electronica', 'Genre-Best-Folk', 'Genre-Best-Hip_Hop-Rap', 'Genre-Best-Jazz', 'Genre-Best-Pop', 'Genre-Best-RandB', 'Genre-Best-Rock', 'Genre-Best-World', 'Instrument_-_Acoustic_Guitar-Solo', 'Instrument_-_Electric_Guitar_(clean)-Solo', 'Instrument_-_Electric_Guitar_(distorted)-Solo', 'Instrument_-_Female_Lead_Vocals-Solo', 'Instrument_-_Harmonica-Solo', 'Instrument_-_Male_Lead_Vocals-Solo', 'Instrument_-_Piano-Solo', 'Instrument_-_Saxophone-Solo', 'Instrument_-_Trumpet-Solo'],
# 	["Vocals-Monotone", "Genre--_Electric_Blues", "Instrument_-_Acoustic_Guitar-Solo", "Genre-Best--_Soul", "Genre--_Cool_Jazz", "Instrument_-_Harmonica-Solo", "Instrument_-_Hand_Drums", "Genre--_Brit_Pop", "Genre-Best-RandB", "Vocals-Falsetto", "Genre-Best-Jazz", "Instrument_-_Saxophone-Solo", "Genre-Best--_Punk", "Genre--_Contemporary_Blues", "Genre-Best-Blues", "Emotion-Bizarre-Weird", "Genre-Best-Country", "Usage-Waking_up", "Instrument_-_Female_Lead_Vocals-Solo", "Usage-Exercising", "Instrument_-_Trumpet-Solo", "Genre-Best-Folk", "Usage-Studying", "Usage-At_work", "Instrument_-_Ambient_Sounds", "Genre--_Swing", "Genre--_Soul", "Genre--_Bebop", "Instrument_-_Trombone", "Genre-Hip_Hop-Rap", "Genre-RandB", "Instrument_-_Violin-Fiddle", "Usage-Sleeping", "Vocals-Call_and_Response", "Genre-Best-Hip_Hop-Rap", "Usage-With_the_family", "Vocals-Rapping", "Instrument_-_Male_Lead_Vocals-Solo", "NOT-Song-Recorded", "Vocals-Duet", "Instrument_-_Organ", "Usage-Getting_ready_to_go_out", "Genre--_Country_Blues", "Genre-Bluegrass", "Genre-Best-World", "Usage-Intensely_Listening", "Genre--_Roots_Rock", "Instrument_-_Acoustic_Guitar", "Genre--_Contemporary_RandB", "Instrument_-_Harmonica", "Genre-Best--_Soft_Rock", "Vocals-Screaming", "Vocals-Gravelly", "Instrument_-_String_Ensemble", "Genre-Best--_Metal-Hard_Rock", "Vocals-Spoken", "Song-Changing_Energy_Level", "Instrument_-_Piano-Solo", "Vocals-Breathy", "Genre-Best--_Classic_Rock", "Genre--_Alternative_Folk", "Genre--_Gospel", "Genre-Jazz", "Usage-Romancing", "Instrument_-_Samples", "Usage-Reading", "Instrument_-_Electric_Guitar_(clean)-Solo", "Genre--_Funk", "NOT-Emotion-Arousing-Awakening", "Emotion-Laid-back-Mellow", "Genre-World", "NOT-Song-Quality", "Instrument_-_Tambourine", "Emotion-Touching-Loving", "Genre--_Metal-Hard_Rock", "Angry-Agressive", "NOT-Song-High_Energy", "Genre-Country", "Vocals-Low-pitched", "Genre--_Dance_Pop", "Usage-Cleaning_the_house", "Instrument_-_Sequencer", "Genre-Folk", "Vocals-Altered_with_Effects", "Emotion-Calming-Soothing", "Emotion-Tender-Soft", "Genre--_Singer-Songwriter", "Emotion-Light-Playful", "Vocals-Aggressive", "NOT-Song-Fast_Tempo", "NOT-Song-Tonality", "Genre-Best--_Alternative", "Song-Very_Danceable", "NOT-Song-Heavy_Beat", "Instrument_-_Saxophone", "Genre-Blues", "NOT-Emotion-Positive-Optimistic", "NOT-Emotion-Exciting-Thrilling", "Vocals-Strong", "NOT-Emotion-Pleasant-Comfortable", "Vocals-High-pitched", "Genre-Best-Electronica", "NOT-Song-Positive_Feelings", "Genre-Electronica", "Instrument_-_Synthesizer", "Instrument_-_Horn_Section", "NOT-Emotion-Powerful-Strong", "Instrument_-_Trumpet", "Emotion-Loving-Romantic", "Genre--_Soft_Rock", "Usage-At_a_party", "Genre--_Punk", "NOT-Emotion-Tender-Soft", "Usage-Hanging_with_friends", "NOT-Emotion-Calming-Soothing", "Usage-Going_to_sleep", "Genre-Best-Rock", "Instrument_-_Piano", "Genre-Best-Pop", "Instrument_-_Electric_Guitar_(distorted)", "Instrument_-_Electric_Guitar_(distorted)-Solo", "NOT-Song-Like", "Emotion-Exciting-Thrilling", "Emotion-Carefree-Lighthearted", "Emotion-Sad", "Song-Tonality", "Emotion-Cheerful-Festive", "Genre-Pop", "Instrument_-_Drum_Machine", "Song-Recorded", "Instrument_-_Electric_Guitar_(clean)", "Song-Texture_Electric", "NOT-Emotion-Laid-back-Mellow", "Emotion-Pleasant-Comfortable", "Song-Recommend", "NOT-Emotion-Happy", "Genre--_Classic_Rock", "NOT-Emotion-Carefree-Lighthearted", "Song-Texture_Synthesized", "Vocals-Vocal_Harmonies", "NOT-Emotion-Emotional-Passionate", "Song-Heavy_Beat", "Emotion-Happy", "NOT-Emotion-Angry-Agressive", "Song-Fast_Tempo", "Instrument_-_Male_Lead_Vocals", "Emotion-Positive-Optimistic", "NOT-Song-Catchy-Memorable", "NOT-Emotion-Light-Playful", "Instrument_-_Female_Lead_Vocals", "Genre--_Alternative", "Song-High_Energy", "Vocals-Emotional", "Song-Like", "Emotion-Powerful-Strong", "Emotion-Arousing-Awakening", "NOT-Song-Changing_Energy_Level", "NOT-Emotion-Cheerful-Festive", "Usage-Driving", "Instrument_-_Backing_vocals", "Instrument_-_Drum_Set", "Genre-Rock", "Song-Catchy-Memorable", "Instrument_-_Bass", "NOT-Emotion-Sad", "NOT-Emotion-Touching-Loving", "Song-Positive_Feelings", "Emotion-Emotional-Passionate", "NOT-Emotion-Loving-Romantic", "Song-Quality", "Song-Texture_Acoustic", "NOT-Song-Recommend", "NOT-Song-Very_Danceable", "NOT-Emotion-Bizarre-Weird"],
# 	["Genre-Best-Blues", "NOT-Song-Quality", "Vocals-Emotional", "Instrument_-_Organ", "Genre--_Bebop", "Song-Changing_Energy_Level", "NOT-Song-Recorded", "Song-High_Energy", "Usage-Driving", "Genre-Best-RandB", "Genre-Best-Jazz", "NOT-Song-High_Energy", "Usage-Sleeping", "Genre--_Funk", "Genre-Electronica", "Song-Very_Danceable", "Genre-Folk", "Instrument_-_Bass", "Instrument_-_Electric_Guitar_(clean)-Solo", "Instrument_-_Electric_Guitar_(distorted)", "Instrument_-_Saxophone-Solo", "Usage-Getting_ready_to_go_out", "NOT-Emotion-Pleasant-Comfortable", "Vocals-Falsetto", "Song-Quality", "Emotion-Cheerful-Festive", "NOT-Emotion-Touching-Loving", "Genre--_Cool_Jazz", "NOT-Emotion-Exciting-Thrilling", "NOT-Emotion-Carefree-Lighthearted", "Genre--_Alternative", "Genre-Jazz", "NOT-Emotion-Powerful-Strong", "Song-Positive_Feelings", "Genre-Bluegrass", "Instrument_-_Samples", "Usage-Waking_up", "NOT-Emotion-Sad", "Song-Tonality", "NOT-Emotion-Arousing-Awakening", "Song-Fast_Tempo", "Instrument_-_Female_Lead_Vocals-Solo", "Instrument_-_Saxophone", "Genre-Rock", "Vocals-Monotone", "Emotion-Arousing-Awakening", "Vocals-Spoken", "Song-Catchy-Memorable", "Usage-At_a_party", "Instrument_-_Harmonica", "Instrument_-_Ambient_Sounds", "Genre-RandB", "Genre-Best-Rock", "Genre-Best--_Punk", "Usage-Intensely_Listening", "Genre-Best--_Soft_Rock", "Song-Texture_Synthesized", "NOT-Song-Positive_Feelings", "Vocals-Low-pitched", "Genre-Best--_Alternative", "Genre-Best-Hip_Hop-Rap", "Emotion-Emotional-Passionate", "Instrument_-_Sequencer", "Usage-Hanging_with_friends", "Usage-Romancing", "Song-Texture_Acoustic", "NOT-Emotion-Angry-Agressive", "Genre--_Swing", "NOT-Song-Tonality", "NOT-Song-Catchy-Memorable", "Instrument_-_Male_Lead_Vocals-Solo", "Instrument_-_Horn_Section", "Instrument_-_String_Ensemble", "Genre-Best-Electronica", "NOT-Emotion-Light-Playful", "Instrument_-_Synthesizer", "Genre-Hip_Hop-Rap", "NOT-Emotion-Cheerful-Festive", "Instrument_-_Piano-Solo", "Genre--_Brit_Pop", "Song-Texture_Electric", "Genre--_Contemporary_RandB", "Instrument_-_Trumpet-Solo", "Instrument_-_Tambourine", "Vocals-Breathy", "Usage-Cleaning_the_house", "Emotion-Carefree-Lighthearted", "Emotion-Tender-Soft", "Emotion-Touching-Loving", "Instrument_-_Backing_vocals", "Usage-Studying", "Genre-Best-Folk", "Genre--_Gospel", "Emotion-Loving-Romantic", "NOT-Emotion-Loving-Romantic", "Genre-Best-Pop", "Instrument_-_Drum_Machine", "Genre--_Roots_Rock", "Genre-World", "Genre-Best--_Classic_Rock", "Vocals-Duet", "NOT-Emotion-Positive-Optimistic", "Emotion-Exciting-Thrilling", "Instrument_-_Female_Lead_Vocals", "NOT-Emotion-Laid-back-Mellow", "NOT-Song-Like", "Instrument_-_Trumpet", "Instrument_-_Acoustic_Guitar", "Emotion-Positive-Optimistic", "NOT-Emotion-Tender-Soft", "Angry-Agressive", "Emotion-Bizarre-Weird", "Instrument_-_Electric_Guitar_(clean)", "Emotion-Sad", "Vocals-Altered_with_Effects", "Usage-Exercising", "Genre--_Metal-Hard_Rock", "Instrument_-_Male_Lead_Vocals", "Emotion-Laid-back-Mellow", "Instrument_-_Harmonica-Solo", "NOT-Song-Changing_Energy_Level", "Vocals-Aggressive", "NOT-Emotion-Calming-Soothing", "Instrument_-_Hand_Drums", "Genre--_Punk", "Vocals-Call_and_Response", "Instrument_-_Violin-Fiddle", "Genre--_Country_Blues", "Instrument_-_Acoustic_Guitar-Solo", "Genre--_Electric_Blues", "NOT-Song-Fast_Tempo", "Vocals-High-pitched", "Emotion-Calming-Soothing", "Vocals-Screaming", "NOT-Song-Very_Danceable", "Genre-Best-World", "Genre-Pop", "Song-Recorded", "NOT-Emotion-Emotional-Passionate", "Genre-Best--_Soul", "Genre--_Dance_Pop", "Genre--_Singer-Songwriter", "Instrument_-_Trombone", "Emotion-Pleasant-Comfortable", "Genre--_Contemporary_Blues", "Vocals-Rapping", "NOT-Song-Heavy_Beat", "Genre--_Alternative_Folk", "Usage-At_work", "Genre-Best-Country", "Emotion-Powerful-Strong", "Instrument_-_Electric_Guitar_(distorted)-Solo", "Emotion-Happy", "Genre--_Soul", "Vocals-Vocal_Harmonies", "Usage-Going_to_sleep", "NOT-Emotion-Happy", "Song-Like", "Genre-Country", "Genre-Blues", "Usage-Reading", "Genre--_Classic_Rock", "Song-Recommend", "Vocals-Gravelly", "Instrument_-_Piano", "NOT-Emotion-Bizarre-Weird", "Vocals-Strong", "NOT-Song-Recommend", "Genre--_Soft_Rock", "Genre-Best--_Metal-Hard_Rock", "Usage-With_the_family", "Song-Heavy_Beat", "Instrument_-_Drum_Set", "Emotion-Light-Playful"],
# 	["Song-Like", "Usage-Hanging_with_friends", "Genre-Best--_Alternative", "Genre--_Classic_Rock", "NOT-Emotion-Happy", "Genre--_Alternative", "NOT-Emotion-Arousing-Awakening", "Usage-Driving", "Instrument_-_Saxophone-Solo", "Instrument_-_String_Ensemble", "Genre-World", "Usage-Exercising", "Genre-Best-RandB", "Emotion-Bizarre-Weird", "NOT-Emotion-Cheerful-Festive", "Song-High_Energy", "Instrument_-_Ambient_Sounds", "NOT-Song-Recommend", "Genre--_Metal-Hard_Rock", "Genre--_Contemporary_Blues", "Song-Positive_Feelings", "NOT-Song-Recorded", "Genre--_Soft_Rock", "Genre-Best--_Metal-Hard_Rock", "NOT-Song-Like", "Song-Recommend", "Genre-Best-Hip_Hop-Rap", "Genre-Best-Pop", "Genre--_Swing", "Vocals-Vocal_Harmonies", "Genre--_Punk", "Vocals-High-pitched", "NOT-Emotion-Exciting-Thrilling", "Usage-Sleeping", "NOT-Song-Quality", "Emotion-Sad", "Vocals-Screaming", "Genre--_Soul", "Instrument_-_Samples", "NOT-Song-Positive_Feelings", "Song-Tonality", "Instrument_-_Trumpet-Solo", "Song-Quality", "Emotion-Touching-Loving", "Instrument_-_Electric_Guitar_(clean)", "Song-Texture_Synthesized", "Emotion-Happy", "Usage-With_the_family", "Vocals-Aggressive", "Emotion-Arousing-Awakening", "Genre-Blues", "NOT-Emotion-Sad", "Usage-At_a_party", "Instrument_-_Organ", "Instrument_-_Male_Lead_Vocals", "Vocals-Call_and_Response", "Instrument_-_Piano-Solo", "Instrument_-_Backing_vocals", "NOT-Emotion-Loving-Romantic", "Emotion-Powerful-Strong", "Song-Heavy_Beat", "Instrument_-_Electric_Guitar_(distorted)-Solo", "Usage-Waking_up", "NOT-Emotion-Angry-Agressive", "NOT-Emotion-Calming-Soothing", "Usage-Studying", "Vocals-Altered_with_Effects", "Instrument_-_Drum_Set", "Genre-Best--_Soul", "Emotion-Pleasant-Comfortable", "Instrument_-_Female_Lead_Vocals", "Genre--_Electric_Blues", "Genre-Bluegrass", "Genre-Best-Blues", "Instrument_-_Harmonica", "Vocals-Strong", "Vocals-Duet", "Genre-Best--_Classic_Rock", "Genre-Best-Country", "NOT-Emotion-Tender-Soft", "Vocals-Falsetto", "Genre-Best-Electronica", "Angry-Agressive", "NOT-Song-Very_Danceable", "NOT-Song-Changing_Energy_Level", "Vocals-Gravelly", "Emotion-Exciting-Thrilling", "Song-Recorded", "NOT-Song-Fast_Tempo", "Usage-Intensely_Listening", "Emotion-Tender-Soft", "Genre-Best--_Soft_Rock", "Instrument_-_Trombone", "Instrument_-_Female_Lead_Vocals-Solo", "Instrument_-_Harmonica-Solo", "NOT-Song-Heavy_Beat", "Instrument_-_Tambourine", "NOT-Emotion-Laid-back-Mellow", "Emotion-Cheerful-Festive", "Instrument_-_Violin-Fiddle", "Instrument_-_Electric_Guitar_(clean)-Solo", "Instrument_-_Electric_Guitar_(distorted)", "Instrument_-_Acoustic_Guitar-Solo", "NOT-Emotion-Pleasant-Comfortable", "Song-Very_Danceable", "Song-Texture_Electric", "Genre-Best-Rock", "Instrument_-_Male_Lead_Vocals-Solo", "Instrument_-_Bass", "Genre--_Gospel", "Genre--_Country_Blues", "Genre-RandB", "Song-Changing_Energy_Level", "Genre-Best-Jazz", "Genre-Best-World", "Genre-Hip_Hop-Rap", "Instrument_-_Acoustic_Guitar", "Usage-Going_to_sleep", "Vocals-Low-pitched", "Usage-Getting_ready_to_go_out", "Instrument_-_Horn_Section", "Usage-Reading", "Vocals-Breathy", "Genre--_Roots_Rock", "Emotion-Calming-Soothing", "Vocals-Monotone", "Genre-Pop", "Genre-Country", "NOT-Emotion-Positive-Optimistic", "Genre--_Brit_Pop", "Vocals-Rapping", "Instrument_-_Synthesizer", "Genre--_Singer-Songwriter", "Genre--_Funk", "Instrument_-_Sequencer", "Emotion-Loving-Romantic", "Emotion-Laid-back-Mellow", "Genre--_Bebop", "Vocals-Emotional", "Usage-Cleaning_the_house", "Genre-Jazz", "Emotion-Emotional-Passionate", "Genre--_Alternative_Folk", "Usage-Romancing", "NOT-Song-Catchy-Memorable", "NOT-Emotion-Powerful-Strong", "Song-Texture_Acoustic", "Emotion-Positive-Optimistic", "Instrument_-_Piano", "NOT-Emotion-Bizarre-Weird", "Instrument_-_Trumpet", "NOT-Emotion-Emotional-Passionate", "Genre--_Contemporary_RandB", "Emotion-Light-Playful", "Genre-Electronica", "Instrument_-_Drum_Machine", "Genre--_Dance_Pop", "Song-Fast_Tempo", "NOT-Emotion-Light-Playful", "Instrument_-_Hand_Drums", "Genre--_Cool_Jazz", "Genre-Folk", "NOT-Song-High_Energy", "Instrument_-_Saxophone", "NOT-Emotion-Carefree-Lighthearted", "Usage-At_work", "NOT-Song-Tonality", "Emotion-Carefree-Lighthearted", "Vocals-Spoken", "Song-Catchy-Memorable", "NOT-Emotion-Touching-Loving", "Genre-Best--_Punk", "Genre-Best-Folk", "Genre-Rock"]
# 	]
# ]

# data_paths = ['medical.csv']
# all_labels = [[
# 	["Class-0-593_70", "Class-1-079_99", "Class-2-786_09", "Class-3-759_89", "Class-4-753_0", "Class-5-786_2", "Class-6-V72_5", "Class-7-511_9", "Class-8-596_8", "Class-9-599_0", "Class-10-518_0", "Class-11-593_5", "Class-12-V13_09", "Class-13-791_0", "Class-14-789_00", "Class-15-593_1", "Class-16-462", "Class-17-592_0", "Class-18-786_59", "Class-19-785_6", "Class-20-V67_09", "Class-21-795_5", "Class-22-789_09", "Class-23-786_50", "Class-24-596_54", "Class-25-787_03", "Class-26-V42_0", "Class-27-786_05", "Class-28-753_21", "Class-29-783_0", "Class-30-277_00", "Class-31-780_6", "Class-32-486", "Class-33-788_41", "Class-34-V13_02", "Class-35-493_90", "Class-36-788_30", "Class-37-753_3", "Class-38-593_89", "Class-39-758_6", "Class-40-741_90", "Class-41-591", "Class-42-599_7", "Class-43-279_12", "Class-44-786_07"],
# # 	["Class-5-786_2", "Class-20-V67_09", "Class-40-741_90", "Class-42-599_7", "Class-18-786_59", "Class-6-V72_5", "Class-26-V42_0", "Class-33-788_41", "Class-8-596_8", "Class-29-783_0", "Class-7-511_9", "Class-15-593_1", "Class-25-787_03", "Class-3-759_89", "Class-13-791_0", "Class-2-786_09", "Class-16-462", "Class-27-786_05", "Class-28-753_21", "Class-19-785_6", "Class-22-789_09", "Class-12-V13_09", "Class-14-789_00", "Class-17-592_0", "Class-30-277_00", "Class-1-079_99", "Class-21-795_5", "Class-39-758_6", "Class-11-593_5", "Class-37-753_3", "Class-10-518_0", "Class-34-V13_02", "Class-23-786_50", "Class-35-493_90", "Class-43-279_12", "Class-38-593_89", "Class-44-786_07", "Class-24-596_54", "Class-31-780_6", "Class-9-599_0", "Class-36-788_30", "Class-41-591", "Class-0-593_70", "Class-4-753_0", "Class-32-486"],
# # 	["Class-19-785_6", "Class-30-277_00", "Class-26-V42_0", "Class-38-593_89", "Class-20-V67_09", "Class-8-596_8", "Class-35-493_90", "Class-18-786_59", "Class-36-788_30", "Class-42-599_7", "Class-22-789_09", "Class-0-593_70", "Class-34-V13_02", "Class-3-759_89", "Class-10-518_0", "Class-1-079_99", "Class-32-486", "Class-9-599_0", "Class-39-758_6", "Class-43-279_12", "Class-24-596_54", "Class-21-795_5", "Class-2-786_09", "Class-37-753_3", "Class-15-593_1", "Class-6-V72_5", "Class-28-753_21", "Class-27-786_05", "Class-13-791_0", "Class-40-741_90", "Class-29-783_0", "Class-33-788_41", "Class-12-V13_09", "Class-7-511_9", "Class-14-789_00", "Class-23-786_50", "Class-16-462", "Class-44-786_07", "Class-5-786_2", "Class-31-780_6", "Class-17-592_0", "Class-4-753_0", "Class-41-591", "Class-11-593_5", "Class-25-787_03"],
# # 	["Class-19-785_6", "Class-32-486", "Class-35-493_90", "Class-41-591", "Class-22-789_09", "Class-26-V42_0", "Class-13-791_0", "Class-29-783_0", "Class-0-593_70", "Class-6-V72_5", "Class-31-780_6", "Class-5-786_2", "Class-11-593_5", "Class-8-596_8", "Class-18-786_59", "Class-17-592_0", "Class-40-741_90", "Class-30-277_00", "Class-10-518_0", "Class-25-787_03", "Class-3-759_89", "Class-28-753_21", "Class-43-279_12", "Class-34-V13_02", "Class-39-758_6", "Class-1-079_99", "Class-7-511_9", "Class-36-788_30", "Class-16-462", "Class-38-593_89", "Class-27-786_05", "Class-12-V13_09", "Class-9-599_0", "Class-44-786_07", "Class-20-V67_09", "Class-14-789_00", "Class-21-795_5", "Class-4-753_0", "Class-2-786_09", "Class-15-593_1", "Class-24-596_54", "Class-37-753_3", "Class-42-599_7", "Class-33-788_41", "Class-23-786_50"]

# ]]
# all_labels = [[
# 	["amazed-suprised","happy-pleased", "relaxing-calm", "quiet-still", "sad-lonely", "angry-aggresive"],
# 	["relaxing-calm", "angry-aggresive", "amazed-suprised", "quiet-still", "sad-lonely", "happy-pleased"],
# 	['sad-lonely', 'quiet-still', 'amazed-suprised', 'angry-aggresive', 'relaxing-calm', 'happy-pleased'],
# 	['relaxing-calm', 'quiet-still', 'angry-aggresive', 'sad-lonely', 'amazed-suprised', 'happy-pleased']
# ]]
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




logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='multilabel_classifiers_permutations_flags_cc.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


table_color='#4CAF50'
print('<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>table{{border-collapse: collapse; font-family: "Trebuchet MS", Arial, Helvetica, sans-serif; width: 50%; margin: 0px auto;}}h2{{font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;}}th, td{{text-align: left; padding: 8px; border: 1px solid #ddd;}}tr:nth-child(even){{background-color: #f2f2f2;}}tr:hover{{background-color: #ddd;}}th{{padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: {}; color: white;}}</style></head><body>'.format(table_color))
for i in range(len(data_paths)):
	logging.info("Empezando entrenamiento del conjunto {}".format(data_paths[i]))
	data_path = BASE_DIR + data_paths[i]
	labels = all_labels[i]
	data = pd.read_csv(data_path, sep='\s*,\s*', encoding='ascii', engine='python')
	print("<h2>{}</h2>".format(data_path))
	if "id"  in data.columns:
		data.drop("id", axis=1)
	
	permutation_results = []

	for permutation_idx in range(len(all_labels[i])):
		# CCC = CircularChainClassifier(MultinomialNB())
		# BR = BinaryRelevance(MultinomialNB())
		CC = ChainClassifier(MultinomialNB())
		logging.info("Usando orden {}".format(permutation_idx))
		# print("Order: {}".format(label_order_name[permutation_idx]))
		
		# print("<h2>CCC</h2>")
		# print("<h2>{}</h2>".format(all_labels[i][permutation_idx]))
		# logging.info("Comenzando entrenamiento con CCC")
		t_start = time()
		logging.info(all_labels[i][permutation_idx])
		CC.train(data, list(all_labels[i][permutation_idx]), k=10)
		# CCC.train(data, list(all_labels[i][permutation_idx]), number_of_iterations=10, k=10)
		permutation_results.append(CC.last_results)
		logging.info("Tiempo de entrenamiento CC = {}".format(time() - t_start))
	table_measures = '<div style="overflow-x:auto;"> <table> <tr><th>Permutación</th><th>GAccMean</th><th>GAcc_std</th><th>MAccMean</th><th>MAcc_std</th><th>MLAccMean</th><th>MLAcc_std</th><th>F-measureMean</th><th>F-measure_std</th> </tr>'
	for last_results in permutation_results:
		table_measures += '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(all_labels[i][permutation_idx], last_results[0], last_results[1], last_results[2], last_results[3], last_results[4], last_results[5], last_results[6], last_results[7])
	table_measures += '</table></div>'
	print(table_measures)
	# print(permutation_results)
		# print("<h2>BR</h2>")
		# logging.info("Comenzando entrenamiento con BR")
		# t_start = time()
		# BR.train(data, all_labels[i][permutation_idx], k=10)
		# logging.info("Tiempo de entrenamiento BR = {}".format(time() - t_start))

		# print("<h2>CC</h2>")
		# logging.info("Comenzando entrenamiento con CC")
		# t_start = time()
		# CC.train(data, all_labels[i][permutation_idx], k=10)
		# logging.info("Tiempo de entrenamiento CC = {}".format(time() - t_start))

print('</body></html>')
