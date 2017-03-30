#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import collections
from sklearn.utils import shuffle


class Bigram:
	def __init__(self, V, D, eta=0.01):
		self.V = V # vocab size
		self.D = D # hidden layer size
		self.W_ih = np.random.normal(0,0.1,size=(V,D)).astype(np.float64) # input to hidden weight matrix(VxD)
		self.W_ho = np.random.normal(0,0.1,size=(D,V)).astype(np.float64) # hidden to output weight matrix(DxV)
		self.eta = eta

	# X : input com N ids que serão transformados em one hot vectors (NxV) 
	# y : output com N ids que serão transformados em one hot vectors (NxV)
	# X = w(t-1) e y = w(t)
	def fit(self,X,y,batch_size=None, epochs=10000):
		#Caso não tenha um valor para o batch, utiliza todos os dados
		samples = len(X)
		if batch_size is None or batch_size > samples:
			batch_size = samples

		#Para garantir uma divisão exata dos dados
		batch_size += samples%batch_size

		#Inicializa na época 0
		epoch = 0
		idx = 0
		while epoch < epochs:

			#Permutar input e output (Treinar aleatóriamente a cada epoca)
			# LOL neste tipo de problema não faz sentido permutar
			# visto que a ordem importa
			# X, y = shuffle(X, y, random_state=0)
			
			#para treinar a rede nessa época
			data = X[idx:idx+batch_size]
			label = y[idx:idx+batch_size]
			train_data = self.onehot(data)
			train_label = self.onehot(label)
	
			#Resetar o contador quando idx == samples
			idx += batch_size
			idx = idx%samples

			#Foward propagation
			z1 = self.W_ih[data,:]
			z2 = np.dot(z1,self.W_ho)
			a2 = self.activation(z2, "softmax")

			#Calcular o erro
			loss = 0.0
			for sample, idx in enumerate(label):
				loss -= np.log(a2[sample,idx])
				a2[sample,idx] -= 1
			loss /= batch_size
			print('epoch: {}    loss: {}'.format(str(epoch), str(loss)))

			#Backprop
			error_a2 = a2
			dW_ho = z1.T.dot(error_a2)
			error_z1 = error_a2.dot(self.W_ho.T)
			#Nao sei como mapear esta conta, entao ainda
			#tive que utilizar o train_data
			dW_ih = train_data.T.dot(error_z1)

			self.W_ho -= self.eta*dW_ho/batch_size
			self.W_ih -= self.eta*dW_ih/batch_size
			# #Truncar para evitar overflow
			# self.W_ho = np.clip(self.W_ho, -0.5, 0.5)
			# self.W_ih = np.clip(self.W_ih, -0.5, 0.5)

			epoch += 1
		print self.W_ih

	def activation(self, x, function="logistic"):
		if function == "logistic":
			return 1 / (1 + np.exp(-x))
		if function == "softmax":
			for idx, sample in enumerate(x):
				exp = np.exp(sample)
				x[idx,:] = exp / exp.sum()
			return x			

	def onehot(self, keys):
		samples = len(keys)
		vectors = np.zeros((samples,self.V)).astype(np.float64)
		for i in xrange(samples):
			vectors[i,keys[i]] = 1.0
		return vectors

#COMENTÁRIOOOOO

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(24, 24))  #in inches

  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y, color="green")
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig("imageTSNE/"+filename)

def saveTSNE(final_embeddings, reverse_dictionary):
	try:
		from sklearn.manifold import TSNE

		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		plot_only = 500
		low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
		labels = [reverse_dictionary[i] for i in xrange(plot_only)]
		plot_with_labels(low_dim_embs, labels, filename="tsne.png")

	except ImportError:
		print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")






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
X = data[:-1]
y = data[1:]
# Hiperparametros
D = 200
eta = 3
batch_size = 200
epochs = 500
# Criando o modelo
Model = Bigram(V,D,eta)
Model.fit(X,y,batch_size,epochs)
saveTSNE(Model.W_ih, reverse_dictionary)