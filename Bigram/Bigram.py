#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import collections
from sklearn.utils import shuffle


class Bigram:
	def __init__(self, V, D, eta=0.01):
		self.V = V # vocab size
		self.D = D # hidden layer size
		self.W_ih = np.random.normal(0,0.1,size=(V,D)) # input to hidden weight matrix(VxD)
		self.W_ho = np.random.normal(0,0.1,size=(D,V)) # hidden to output weight matrix(DxV)
		self.eta = eta

	# X : input com N ids que serão transformados em one hot vectors (NxV) 
	# y : output com N ids que serão transformados em one hot vectors (NxV)
	# X = w(t-1) e y = w(t)
	def fit(self,X,y,batch_size=None, epochs=10000):
		#Caso não tenha um valor para o batch, utiliza todos os dados
		(samples, features) = X.shape
		if batch_size is None or batch_size > samples:
			batch_size = samples

		#Inicializa na época 0
		epoch = 0
		while epoch < epochs:

			#Permutar input e output (Treinar aleatóriamente a cada epoca)
			X, y = shuffle(X, y, random_state=0)
			#Extrai batch_size samples
			train_data = X[0:batch_size+1,:]
			train_label = y[0:batch_size+1,:]

			#Foward propagation
			z1 = np.dot(train_data,self.W_ih)
			z1 = np.where(z1>0.0, 1.0, 0.0)
			z2 = np.dot(z1,self.W_ho)
			a2 = self.activation(z2, "softmax")

			#Calcular o erro
			loss = 0.0
			for sample, idx in enumerate(train_label.argmax(axis=1)):
				loss -= np.log(a2[sample,idx])
			loss = loss.sum()/batch_size
			print('epoch: {}    loss: {}'.format(str(epoch), str(loss)))

			#Backprop
			error_a2 = a2 - train_label
			dW_ho = z1.T.dot(error_a2)
			error_z1 = error_a2.dot(self.W_ho.T)
			dW_ih = train_data.T.dot(error_z1)

			self.W_ho -= self.eta*dW_ho/batch_size
			self.W_ih -= self.eta*dW_ih/batch_size

			epoch += 1
		print self.W_ho

	def activation(self, x, function="logistic"):
		if function == "logistic":
			return 1 / (1 + np.exp(-x))
		if function == "softmax":
			return np.exp(x)/np.exp(x).sum(axis=0)

#COMENTÁRIOOOOO



def onehot(keys, V):
	samples = len(keys)
	vectors = np.zeros((samples,V))

	for i in xrange(samples):
		vectors[i,keys[i]] = 1.0

	return vectors


def build_dataset(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
	  		index = dictionary[word]
		else:
	  		index = 0  # dictionary['UNK']
	  		unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary


# Get corpus text
with open('input.txt', 'r') as f:
	read_data = f.read()

V = 2000
data_id, count, dictionary, reverse_dictionary = build_dataset(read_data, V)

# data = ids2onehot(data, V)
data = onehot(data_id, V)
X = data[:-1]
y = data[1:]

print X.shape
print y.shape
D = 200
eta = 0.0001
batch_size = 10
epochs = 100
Model = Bigram(V,D,eta)
Model.fit(X,y,batch_size,epochs)