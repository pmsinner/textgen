#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from math import sqrt
import argparse
import sys

def createParser():	#Парсер аргументов
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument ('-h', '--help', action='store_const', const=True, default=False)
	parser.add_argument ('-g', '--generate', action='store_const', const=True, default=False)
	parser.add_argument ('--seed', default="")
	parser.add_argument ('--hdf5', default='input.hdf5')
	parser.add_argument ('-s', '--source', default='wonderland.txt')
	parser.add_argument ('--wordsonly', action='store_const', const=True, default=False)
	parser.add_argument ('--epoch', type=int, default=100)
	parser.add_argument ('-d', '--dimensions', type=int, default=100)
	parser.add_argument ('--sentencelen', type=int, default=20)
	parser.add_argument ('-b', '--batchsize', type=int, default=128)
	parser.add_argument ('-l', '--layers', type=int, default=1)
	parser.add_argument ('--genlength', type=int, default=1000)
	return parser

def clearConsole(): 
	print("\033[1J")
	print("\033[;H")

def getDistance(p1,p2): #Расстояние между двумя точка в пространстве (p1, p2 -- списки равной размерности)
	sum_sqr = 0
	for i, j in zip(p1,p2):
		sum_sqr += (i-j)**2
	distance = sqrt(sum_sqr)
	return distance

print(colored("Загружается парсер аргументов", "green"))
parser = createParser()
namespace = parser.parse_args()

if namespace.help:
	print("\n textgen.py - программа для генерации новых текстов на основе готовых произведений \n\n -h, --help для вызова справки \n -s, --source [source.txt] путь к исходному текстовому файлу \n -g, --generate если флаг активен, программа будет генерировать текст вместо создания hdf5 \n --seed [42] семя для генерации случайного начала текста \n --hdf5 [input.hdf5] путь к файлу с нейросетью. Нужен, если включен флаг -g \n --epoch [100] количество эпох обучения нейросети \n -d --dimensions [100] размерность вектора слова \n --sentencelen [20] длина обрабатываемого предложения\n-b, --batchsize [128] размер батча нейросети  \n -l --layers [1] количество рекуррентных слоёв нейросети \n --genlength длина текста в режиме генерации \n\n by Peter Sinner \n")
	sys.exit()

path = namespace.source
commas = [",", ".", "/", "\\", ",",  "\'", "\"", "(", ")", "-", "!", "?", "{", "}", '*', ':', ';', '[', ']', '_', "”", "‘", "’", "“"]
print(colored("Идёт обработка текста", "green"))

text = open(path).read().lower()
for c in commas:
	text = text.replace(c,"")		#Удаление знаков препинания
f = open(path, "w")
f.write(text) 					#Перезаписываем очищенный текст
f.close()
if os.path.exists(namespace.source + ".w2vm"): 
	print(colored("Загружается словарь", "green"))
	tw2v = Word2Vec.load(namespace.source + ".w2vm") 
else: 						#Генерация стандартного словаря для word2vec
	print(colored("Создаётся словарь. Это может занять некоторое время", "green"))
	tw2v = Word2Vec(Text8Corpus(path), size=namespace.dimensions, min_count=1)
	tw2v.save(namespace.source + ".w2vm")
	print(colored("Словарь сохранён", "green"))

	
def vecToWord(vec):
	key = ""
	minD = None
	for word in tw2v.vocab.keys():
		if minD == None:
			key = word
			minD = getDistance(vec.tolist(),tw2v[key].tolist())
		else:
			distance = getDistance(vec.tolist(),tw2v[word].tolist())
			if distance <= minD:
				key = word
				minD = distance
	return key
gWords = tw2v.vocab.keys()
wText = text.split()
for word in list(set(wText)):		#Удаление не пойманых word2vec слов из текста 
	if not word in gWords:
		text = text.replace(word,"")
print(colored("Идёт векторизация текста", "green")) #Заполняем vecText векторами вместо слов
vecText = []
for word in wText:
	vecText.append(tw2v[word])

print(colored("Векторизация текста успешно завершена", "green"))

maxlen = namespace.sentencelen #Длина обрабатываемого предложения
step = 3 #Шаг анализа
sentences = []
next_chars = []

index_dict = {}
dict_index = {}
for i, key in enumerate(tw2v.vocab.keys()): #index_dict и dict_index нужны для преобразования слов в их порядковые номера (для i/o нейросети) и наоборот
	index_dict[key] = i
	dict_index[i] = key
dimensions = len(vecText[0])
n_symbols = len(index_dict) + 1

for i in range(0, len(wText) - maxlen, step): #Sentences -- список придложений (вход нейроосети), next_chars -- список их окончаний (ожидаемый выход)
	sentences.append(wText[i: i + maxlen])
	next_chars.append(wText[i + maxlen])
print(colored("Получено шаблонов: " + str(len(sentences)), "green"))

embedding_weights = np.zeros((n_symbols+1,dimensions)) #Переводим словарь word2vec в удобный для Keras формат
for word,index in index_dict.items():
    embedding_weights[index,:] = tw2v[word] 
X = np.zeros((len(sentences), maxlen), dtype=np.int) #Формируем входные и выходные массивы для обучения
y = np.zeros((len(sentences)), dtype=np.int)
for i, sentence in enumerate(sentences):
	for t, word in enumerate(sentence):
		X[i,t] = index_dict[word]
	y[i] = index_dict[next_chars[i]]

print(colored("Строим модель", "green"))

model = Sequential()
model.add(Embedding(output_dim=dimensions, input_dim=n_symbols + 1, input_length=maxlen, weights=[embedding_weights])) #Слой векторизации загружается из word2vec
for i in range(namespace.layers - 1): #Добавляем рекуррентные LSTM
	model.add(LSTM(dimensions, return_sequences=True))
	model.add(Dropout(0.07))
model.add(LSTM(dimensions, return_sequences=False))
model.add(Dense(n_symbols + 1)) #Одномерный выход
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)

if namespace.generate: #В режиме генерации
	filename = namespace.hdf5
	model.load_weights(filename)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
	if namespace.seed == '': #Получаем начало для будущего текст
		start_index = random.randint(0, len(vecText) - maxlen - 1)
		sentence = wText[start_index: start_index + maxlen]
	else:
		if len(seed.split()) < maxlen:
			print(colored("Длина начального текста должна равняться " + str(maxlen) + " словам. Для случайного начала оставьте параметр seed пустым","red"))
			sys.exit()
		sentence = seed.split()[:maxlen:]
	generated = " ".join(sentence) 
	print("-"*15, " НАЧАЛО ", "-"*15)
	sys.stdout.write(generated)

	while len(generated) < namespace.genlength: #Сама генерация текст
		x = np.zeros((1, maxlen), dtype=np.int)
		for i, word in enumerate(sentence): #Заполняем x словами из начала предложения
			x[0, i] = index_dict[word]
		preds = model.predict(x, verbose=0)[0] #Предсказываем следующее слово
		next_word =  dict_index[preds.tolist().index(max(preds))] #Выбираем самое вероятное
		generated = generated + " " +next_word
		del(sentence[0])
		sentence.append(next_word)
		sys.stdout.write(" " + next_word) #И выводим в консоль
		sys.stdout.flush()
	print()
	print("-"*15, " КОНЕЦ ", "-"*15)
	sys.exit()

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)


for iteration in range(1, namespace.epoch + 1):
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(X, y, batch_size=namespace.batchsize, nb_epoch=1) #Прогоняем текстовые данные по одной эпохе, после каждой пишем пробный текст
	model.save_weights(namespace.source + "-" + str(iteration) + ".hdf5") #И сохраняем результаты
	start_index = random.randint(0, len(vecText) - maxlen - 1)
	generated = ''
	sentence = wText[start_index: start_index + maxlen]
	generated = generated + " ".join(sentence)
	print('----- Example with seed: "' + " ".join(sentence) + '"')
	sys.stdout.write(generated)

	while len(generated) < 300:
		x = np.zeros((1, maxlen), dtype=np.int)
		for i, word in enumerate(sentence):
			x[0, i] = index_dict[word]
		preds = model.predict(x, verbose=0)[0] #Как и в режиме генерации, выбираем самые вероятные слова
		next_word =  dict_index[preds.tolist().index(max(preds))]
		generated = generated + " " +next_word
		del(sentence[0])
		sentence.append(next_word)
		sys.stdout.write(" " + next_word)
		sys.stdout.flush()
	print()
print(colored("Обучение завершено!", "green"))

