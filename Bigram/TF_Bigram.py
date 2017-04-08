#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np
import collections
from sklearn.utils import shuffle


class Bigram:
	def __init__(self, V, D, eta=1e-4):
		self.V = V
		self.D = D
		self.eta = eta
		

	def train(self, train_data, train_label, batch_size=None, epochs=10000):
		#Definir o grafo

		#input da rede(realimentação)
		x  = tf.placeholder(tf.float32, [None, self.V])
		y_ = tf.placeholder(tf.float32, [None, self.V]) #gabarito
		keep_prob = tf.placeholder(tf.float32)

		#definindo a estrutura da rede neural
		W1 = tf.Variable(tf.random_normal([self.V,self.D], stddev=0.05, dtype=tf.float32), name="W1")
		W2 = tf.Variable(tf.random_normal([self.D,self.V], stddev=0.05, dtype=tf.float32), name="W2")

		#definindo as operações entre as layers
		h = tf.matmul(x,W1)
		h = tf.clip_by_value( h, -1, 1 )
		# h_drop = tf.nn.dropout(h, keep_prob)
		y = tf.nn.softmax(tf.matmul(h,W2))

		
	

		#funcao de custo
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))
		train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


		#Caso não tenha um valor para o batch, utiliza todos os dados
		samples = len(train_data)
		if batch_size is None or batch_size > samples:
			batch_size = samples
		
		#Para garantir uma divisão exata dos dados
		batch_size += samples%batch_size


		#indice que marca o inicio do batch
		idx = 0

		init = tf.global_variables_initializer()
		with tf.Session() as session:
			init.run()
			print("Initialized")

			for i in range(epochs):
				inputs = self.onehot(train_data[idx:idx+batch_size])
				outputs = self.onehot(train_label[idx:idx+batch_size])
				#Resetar o contador quando idx == samples
				idx += batch_size
				idx = idx%samples

				if i%1 == 0:
					train_accuracy = accuracy.eval(feed_dict={
				        x:inputs, y_: outputs, keep_prob: 1.0})
					ce = cross_entropy.eval(feed_dict={
				        x:inputs, y_: outputs, keep_prob: 1.0})
					out = y.eval(feed_dict={
				        x:inputs, y_: outputs, keep_prob: 1.0})
					print("step %d, training accuracy %g"%(i, train_accuracy))
					print("step %d, cross entropy %g"%(i, ce))
					print(out)

				train_step.run(feed_dict={x: inputs, y_: outputs, keep_prob: 1.0})

			print("test accuracy %g"%accuracy.eval(feed_dict={
	    		x: inputs, y_: outputs, keep_prob: 1.0}))


	def onehot(self, keys):
		samples = len(keys)
		vectors = np.zeros((samples,self.V)).astype(np.float32)
		for i in xrange(samples):
			vectors[i,keys[i]] = 1.0
		return vectors



# Passo 1º: Criar o banco de dados
#1.1 definir função de preprocessamento
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


#1.2 ler o arquivo
import zipfile
with zipfile.ZipFile('input.zip', 'r') as myzip:
    read_data = myzip.read('input').split()

# Tamanho do vocabulário:
V = 1000

# Preprocessamento do corpus
data, count, dictionary, reverse_dictionary = build_dataset(read_data, V)

# Transformar data em numpy array para otimizar as contas
data = np.array(data)

# Como o output é a palavra seguinte y é a próxima palavra
train_data = data[:-1]
train_label = data[1:]

# Hiperparametros
D = 100
eta = 1e-2
batch_size = 200
epochs = 500

# Criando o modelo
Model = Bigram(V,D,eta)
Model.train(train_data,train_label,batch_size,epochs)




