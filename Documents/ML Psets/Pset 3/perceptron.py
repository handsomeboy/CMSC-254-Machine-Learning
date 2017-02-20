import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
	def __init__(self):
		data = []
		for line in open('train35.digits'):
			data.append([int(x) for x in line.split()])

		self.train_data = np.stack(data)

		labels = []
		for line in open('train35.labels'):
			labels.append(int(line))

		self.train_labels = labels
		self.w = np.zeros(len(self.train_data[0]))

	def predict(self, data):
		predictions = []
		for observation in np.array([data]):
			norm_observation = observation/np.linalg.norm(observation)
			y_hat = 1 if self.w.dot(norm_observation) >= 0 else -1
			predictions.append(y_hat)

		return predictions

	def train(self,epochs=5):
		mistakes = []
		for x in xrange(0, epochs):
			for i in xrange(0, len(self.train_data)):
				y_hat = self.predict(self.train_data[i])[0]
				if y_hat != self.train_labels[i]:
					self.w += self.train_data[i]*(-np.sign(y_hat))
					mistakes.append(1)
				else:
					mistakes.append(0)

		plt.plot(np.cumsum(mistakes))
		plt.xlabel('number of examples seen')
		plt.ylabel('cumulative error')
		#plt.show()

	def test(self):
		output = open('test35.predictions','w')
		for line in open('train35.digits'):
			data = ([int(x) for x in line.split()])
			y_hat = self.predict(data)[0]
			output.write("%s\n" % y_hat)

if __name__ == '__main__':
	perceptron = Perceptron()
	perceptron.train()
	perceptron.test()