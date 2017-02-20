import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

class Multiclass_Perceptron():
	def __init__(self):
		data = []
		for line in open('train01234.digits'):
			data.append([int(x) for x in line.split()])

		self.train_data = np.stack(data)

		labels = []
		for line in open('train01234.labels'):
			labels.append(int(line))

		self.train_labels = labels
		self.w = np.zeros((len(self.train_data[0]),5))

	def predict(self, data):
		predictions = []
		for observation in np.array([data]):
			norm_observation = observation/np.linalg.norm(observation)
			y_hat = np.argmax([self.w[:,digit].dot(norm_observation) for digit in xrange(0, 5)])
			predictions.append(y_hat)

		return predictions

	def train(self,epochs=20):
		mistakes = []
		for x in xrange(0, epochs):
			for i in xrange(0, len(self.train_data)):
				y_hat = self.predict(self.train_data[i])[0]
				if y_hat != self.train_labels[i]:
					self.w[:,y_hat] -= self.train_data[i]/2.0
					self.w[:,self.train_labels[i]] += self.train_data[i]/2.0
					mistakes.append(1)
				else:
					mistakes.append(0)

		plt.plot(np.cumsum(mistakes))
		plt.xlabel('number of examples seen')
		plt.ylabel('cumulative error')
		#plt.show()

	def test(self):
		output = open('test01234.predictions','w')
		for line in open('test01234.digits'):
			data = ([int(x) for x in line.split()])
			y_hat = self.predict(data)[0]
			output.write("%s\n" % y_hat)

if __name__ == '__main__':
	perceptron = Multiclass_Perceptron()
	perceptron.train()
	perceptron.test()
