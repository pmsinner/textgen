#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from termcolor import colored
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from gensim.models.word2vec import Word2Vec, Text8Corpus
import numpy as np
import random
import os
import math
import argparse
import sys

def lToStr(lst):
	result = ""
	for st in lst:
		result = result + st + " "
	return result

def createParser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument ('-h', '--help', action='store_const', const=True, default=False)
	parser.add_argument ('-g', '--generate', action='store_const', const=True, default=False)
	parser.add_argument ('--seed', default="")
	parser.add_argument ('--hdf5', default='input.hdf5')
	parser.add_argument ('-s', '--source', default='wonderland.txt')
	parser.add_argument ('--erasepunctuation', action='store_const', const=True, default=False)
	parser.add_argument ('--wordsonly', action='store_const', const=True, default=False)
	parser.add_argument ('--epoch', type=int, default=100)
	parser.add_argument ('-b', '--batchsize', type=int, default=128)
	parser.add_argument ('-o', '--output', default='output.txt')
	#parser.add_argument ('-l', '--layers', type=int, default=1)
	parser.add_argument ('--genlength', type=int, default=1000)
	return parser

def clearConsole():
	print("\033[1J")
	print("\033[;H")

def getDistance(p1,p2):
	sum_sqr = 0
	for i, j in zip(p1,p2):
		sum_sqr += (i-j)**2
	distance = math.sqrt(sum_sqr)
	return distance

print(colored("Загружается парсер аргументов", "green"))
parser = createParser()
namespace = parser.parse_args()

if namespace.help:
	print("\n textgen.py - programm, that generates special hdf5 bases, that can be used for making new text, based on old ones. \n\n -h, --h for help \n -s, --source [source.txt] path to source .txt file \n -g, --generate if you use this flag, programm will generate text using existing hdf5 file, else it will generate hdf5 file. If you use this flag, you need to set the --seed, the --hdf5 file, the number of --layers, the --genlength and the --source file \n --seed [42] is used, when the -g flag is on. Use [42] and program will get a random seed from the text, it's an integer. \n --hdf5 [input.hdf5] the path to the hdf5 file, if you use the -g flag \n --erasepunctuation (maybe, it will impprove the result) \n --wordsonly will fill you output only with words. It must be used wiith --generate flag \n --epoch [100] the bigger, the better (and slower) \n -b, --batchsize [128] less is better (and slower) \n -o, --output [weights-improvement-{epoch:02d}-{loss:.4f}.hdf5] the path to the output file, don't change if you don't understand, how does it work \n -l --layers [1] number of layers, that will neural network have. 1-3 is optimal \n --genlength length of generated text if -generate flag is set. \n\n by Peter Sinner \n")
	sys.exit()

path = namespace.source
commas = [",", ".", "/", "\\", ",",  "\'", "\"", "(", ")", "-", "!", "?", "{", "}", '*', ':', ';', '[', ']', '_', "”", "‘", "’", "“"]
print(colored("Идёт обработка текста", "green"))

text = open(path).read().lower()
for c in commas:
	text = text.replace(c,"")
f = open(path, "w")
f.write(text)
f.close()
if os.path.exists(namespace.source + ".w2vm"):
	print(colored("Загружается словарь", "green"))
	tw2v = Word2Vec.load(namespace.source + ".w2vm") 
else:
	print(colored("Создаётся словарь. Это может занять некоторое время", "green"))
	tw2v = Word2Vec(Text8Corpus(path), size=100, min_count=1)
	#tw2v.build_vocab(wText)
	#print(tw2v.vocab)
	#tw2v.train(wText)
	#print(tw2v.vocab)
	tw2v.save(namespace.source + ".w2vm")
	print(colored("Словарь сохранён", "green"))

gWords = tw2v.vocab.keys()
words = list(set(text.split()))


def vecToWord(vec):
	key = ""
	minD = None
	for word in tw2v.vocab.keys():
		if minD == None:
			#print(getDistance(vec,tw2v[key]))
			key = word
			minD = getDistance(vec.tolist(),tw2v[key].tolist())
		else:
			distance = getDistance(vec.tolist(),tw2v[word].tolist())
			if distance <= minD:
				key = word
				minD = distance
	return key

for word in words:
	if not word in gWords:
		text = text.replace(word,"")
print(colored("Идёт векторизация текста", "green"))
vecText = []
wText = text.split()
for word in text.split():
	vecText.append(tw2v[word])

print(colored("Векторизация текста успешно завершена", "green"))
#----------------------ТЕСТ------------------------------

maxlen = 20
step = 3
sentences = []
next_chars = []

index_dict = {}
dict_index = {}
for i, key in enumerate(tw2v.vocab.keys()):
	index_dict[key] = i
	dict_index[i] = key
dimensions = len(vecText[0])
n_symbols = len(index_dict) + 1

for i in range(0, len(wText) - maxlen, step):
	sentences.append(wText[i: i + maxlen])
	next_chars.append(wText[i + maxlen])
print(colored("Получено шаблонов: " + str(len(sentences)), "green"))

embedding_weights = np.zeros((n_symbols+1,dimensions))
for word,index in index_dict.items():
    embedding_weights[index,:] = tw2v[word]
print(colored("Dimensions: " + str(dimensions), "green"))
X = np.zeros((len(sentences), maxlen), dtype=np.int)
y = np.zeros((len(sentences)), dtype=np.int)
for i, sentence in enumerate(sentences):
	for t, word in enumerate(sentence):
		X[i,t] = index_dict[word]
	y[i] = index_dict[next_chars[i]]

print(colored("Строим модель", "green"))

model = Sequential()
model.add(Embedding(output_dim=dimensions, input_dim=n_symbols + 1, input_length=maxlen, weights=[embedding_weights]))
model.add(LSTM(dimensions, return_sequences=False))
model.add(Dropout(0.25))
model.add(Dense(n_symbols + 1))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)


if namespace.generate:
	filename = namespace.hdf5
	model.load_weights(filename)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
	if namespace.seed == '':
		start_index = random.randint(0, len(vecText) - maxlen - 1)
		sentence = wText[start_index: start_index + maxlen]
	else:
		if len(seed.split()) < maxlen:
			print(colored("Длина начального текста должна равняться " + str(maxlen) + " словам. Для случайного начала оставьте параметр seed пустым","red"))
			sys.exit()
		sentence = seed.split()[:maxlen:]
	generated = lToStr(sentence)
	#print(score)
	print("-"*15, " НАЧАЛО ", "-"*15)
	sys.stdout.write(generated)

	while len(generated) < namespace.genlength:
		x = np.zeros((1, maxlen), dtype=np.int)
		for i, word in enumerate(sentence):
			x[0, i] = index_dict[word]
		preds = model.predict(x, verbose=0)[0]
		#print(model.predict(x, verbose=0))
		next_index = preds
		#print(next_index)
		next_word =  dict_index[preds.tolist().index(max(preds))]
		generated = generated + " " +next_word
		del(sentence[0])
		sentence.append(next_word)
		#print(lToStr(sentence))
		sys.stdout.write(" " + next_word)
		sys.stdout.flush()
	print()
	print("-"*15, " КОНЕЦ ", "-"*15)
	sys.exit()

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)


for iteration in range(1, namespace.epoch + 1):
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(X, y, batch_size=namespace.batchsize, nb_epoch=1)
	model.save_weights(namespace.source + "-" + str(iteration) + ".hdf5")
	start_index = random.randint(0, len(vecText) - maxlen - 1)
	generated = ''
	sentence = wText[start_index: start_index + maxlen]
	generated = generated + lToStr(sentence)
	#print(score)
	print('----- Example with seed: "' + lToStr(sentence) + '"')
	sys.stdout.write(generated)

	while len(generated) < 300:
		x = np.zeros((1, maxlen), dtype=np.int)
		for i, word in enumerate(sentence):
			x[0, i] = index_dict[word]
		preds = model.predict(x, verbose=0)[0]
		#print(model.predict(x, verbose=0))
		next_index = preds
		#print(next_index)
		next_word =  dict_index[preds.tolist().index(max(preds))]
		generated = generated + " " +next_word
		del(sentence[0])
		sentence.append(next_word)
		#print(lToStr(sentence))
		sys.stdout.write(" " + next_word)
		sys.stdout.flush()
	print()
print(colored("Обучение завершено!", "green"))



