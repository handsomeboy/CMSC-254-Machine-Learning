import numpy as np

#helper functions
def split_data(X, Y, split):
	permutation = np.random.permutation(len(X))
	split = int(len(X)*split)

	X = X[permutation]
	Y = Y[permutation]

	train_X = X[:split]
	train_Y = Y[:split]

	test_X = X[split:]
	test_Y = Y[split:]

	return train_X, test_X, train_Y, test_Y

def sigmoid(activation):
	return 1.0/(1.0 + np.exp(-activation))

def sigmoidDerivative(x):
	return x*(1-x)

class NeuralNetwork():
	def __init__(self, layer_1_size=64, layer_2_size=32,nb_epochs=10, lr=.15, 
				verbose=False,decay=.1,momentum=.4,nesterov=True):

		self.verbose = verbose
		self.layer_1 = np.random.randn(785, layer_1_size)/np.sqrt(785)
		self.layer_2 = np.random.randn(layer_1_size, layer_2_size)/np.sqrt(layer_1_size)
		self.layer_3 = np.random.randn(layer_2_size, 10)/np.sqrt(layer_2_size)
		self.nb_epochs = nb_epochs
		self.lr = lr
		self.decay = decay
		self.momentum = momentum
		self.m1 = 0
		self.m2 = 0
		self.m3 = 0
		self.nesterov = nesterov

	def forwardPropagate(self, example):
		z1 = example.dot(self.layer_1)
		a1 = sigmoid(z1)
		z2 = a1.dot(self.layer_2)
		a2 = sigmoid(z2)
		z3 = a2.dot(self.layer_3)
		exp_scores = np.exp(z3)
		a3 = exp_scores/np.sum(exp_scores)

		return a1, a2, a3

	def backPropagate(self, example, label, a1, a2, a3):
		delta3 = (a3 - label)
		delta2 = sigmoidDerivative(a2)*np.matmul(self.layer_3, delta3)
		delta1 = sigmoidDerivative(a1)*np.matmul(self.layer_2, delta2)
		
		if self.nesterov:
			self.m1 = self.m1*(self.momentum**2) - (1+self.momentum)*self.lr*np.outer(example, delta1) 
			self.m2 = self.m2*(self.momentum**2) - (1+self.momentum)*self.lr*np.outer(a1, delta2)
			self.m3 = self.m3*(self.momentum**2) - (1+self.momentum)*self.lr*np.outer(a2, delta3)

		else:
			self.m1 = self.m1*(self.momentum) - self.lr*np.outer(example, delta1) 
			self.m2 = self.m2*(self.momentum) - self.lr*np.outer(a1, delta2)
			self.m3 = self.m3*(self.momentum) - self.lr*np.outer(a2, delta3)

		self.layer_1 += self.m1 
		self.layer_2 += self.m2
		self.layer_3 += self.m3

	def score(self, x, y):
		predictions = self.predict(x)
		accuracy = (predictions == np.argmax(y, axis=1)).sum()
		return accuracy/(len(x)*1.0)

	def fit(self, x, y, validation_split=0, early_stopping=False):
		self.accuracyHistory = []
		if validation_split:
			train_X, test_X, train_Y, test_Y = split_data(x, y, (1-validation_split))

		for epoch in xrange(self.nb_epochs):
			self.lr /= 1. + self.decay * epoch
			for example, label in zip(x, y):
				a1, a2, a3 = self.forwardPropagate(example)
				self.backPropagate(example, label, a1, a2, a3)

			accuracy = self.score(x, y)

			if self.verbose:
		 		print 'epoch ', epoch + 1
		 		print  'accuracy: ', accuracy

			if validation_split:
				val_accuracy = self.score(test_X, test_Y)
				self.accuracyHistory.append((accuracy, val_accuracy))

				if self.verbose:
					print  'val accuracy: ', val_accuracy

		self.accuracyHistory = np.array(self.accuracyHistory)

	def plot_history(self):
		import matplotlib.pyplot as plt

		plt.figure()
		train = plt.plot(self.accuracyHistory[:,0],c='b',label='train')
		test = plt.plot(self.accuracyHistory[:,1],c='r',label='test')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		plt.legend(loc=4)
		plt.show()

	def predict(self, x):
		probabilities = self.forwardPropagate(x)[2]
		return np.argmax(probabilities,axis=1).astype(int)