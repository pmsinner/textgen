from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import argparse
import sys

def createParser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument ('-h', '--help', action='store_const', const=True, default=False)
	parser.add_argument ('-g', '--generate', action='store_const', const=True, default=False)
	parser.add_argument ('--seed', type = int, default=42)
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

parser = createParser()
namespace = parser.parse_args()

if namespace.help:
	print("\n textgen.py - programm, that generates special hdf5 bases, that can be used for making new text, based on old ones. \n\n -h, --h for help \n -s, --source [source.txt] path to source .txt file \n -g, --generate if you use this flag, programm will generate text using existing hdf5 file, else it will generate hdf5 file. If you use this flag, you need to set the --seed, the --hdf5 file, the number of --layers, the --genlength and the --source file \n --seed [42] is used, when the -g flag is on. Use [42] and program will get a random seed from the text, it's an integer. \n --hdf5 [input.hdf5] the path to the hdf5 file, if you use the -g flag \n --erasepunctuation (maybe, it will impprove the result) \n --wordsonly will fill you output only with words. It must be used wiith --generate flag \n --epoch [100] the bigger, the better (and slower) \n -b, --batchsize [128] less is better (and slower) \n -o, --output [weights-improvement-{epoch:02d}-{loss:.4f}.hdf5] the path to the output file, don't change if you don't understand, how does it work \n -l --layers [1] number of layers, that will neural network have. 1-3 is optimal \n --genlength length of generated text if -generate flag is set. \n\n by Peter Sinner \n")
	sys.exit()

path = namespace.source
commas = [",", ".", "/", "\\", ",",  "\'", "\"", "(", ")", "-", "!", "?", "{", "}", '*', ':', ';', '[', ']', '_']
text = open(path).read().lower()
if namespace.erasepunctuation:
	text = filter(lambda x: not x in commas, text)
if namespace.generate:
	if namespace.seed >= len(text):
		print ("Sorry, but seed is very big. For this text it must be smaller then", len(text))
		print ("Try again with another seed")
		sys.exit()
print('Chars in Text:', len(text))

chars = sorted(list(set(text)))
print('Total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
print('Total Patterns:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1



print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)

def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds)
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

if namespace.generate:
	filename = namespace.hdf5
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	generation = ""
	if namespace.seed == 42:
		start_index = random.randint(0, len(text) - maxlen - 1)
	else:
		start_index = namespace.seed	
	sentence = text[start_index: start_index + maxlen]
	generation += sentence
	print('----- Example with seed: "' + sentence + '"')
	sys.stdout.write(generation)
	for i in range(namespace.genlength):
		x = np.zeros((1, maxlen, len(chars)))
		for t, char in enumerate(sentence):
			x[0, t, char_indices[char]] = 1.
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, 1.0)
		next_char = indices_char[next_index]

		generation += next_char
		sentence = sentence[1:] + next_char

		sys.stdout.write(next_char)
		sys.stdout.flush()
	print()
	sys.exit()


model.compile(loss='categorical_crossentropy', optimizer=optimizer)



for iteration in range(1, namespace.epoch + 1):
	print()
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(X, y, batch_size=namespace.batchsize, nb_epoch=1)
	start_index = random.randint(0, len(text) - maxlen - 1)
	generated = ''
	sentence = text[start_index: start_index + maxlen]
	generated += sentence
	print('----- Example with seed: "' + sentence + '"')
	sys.stdout.write(generated)
	for i in range(200):
		x = np.zeros((1, maxlen, len(chars)))
		for t, char in enumerate(sentence):
			x[0, t, char_indices[char]] = 1.
		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, 1.0)
		next_char = indices_char[next_index]
		generated += next_char
		sentence = sentence[1:] + next_char
		sys.stdout.write(next_char)
		sys.stdout.flush()
	print()
	model.save_weights(namespace.source + "-" + str(iteration) + ".hdf5")
print("Learning finished successful!")

