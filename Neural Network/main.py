import numpy as np
from pandas import read_csv
from neuralnetwork import NeuralNetwork
from itertools import izip, product
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt

def to_categorical(y):
	n_classes = len(np.unique(y))
	n_examples = len(y)
	categorical = np.zeros((n_examples, n_classes))
	categorical[np.arange(n_examples), y.flatten().astype(int)] = 1
	return categorical

def load_training_data():
	train_X = read_csv('TrainDigitX.csv.gz',header=None,dtype=np.float64).values
	train_X = np.c_[train_X, np.ones(len(train_X))]
	train_Y = read_csv('TrainDigitY.csv.gz',header=None,dtype=np.float64).values
	train_Y = to_categorical(train_Y)
	return train_X, train_Y

def load_test_data():
	test_X = read_csv('TestDigitX.csv.gz',header=None,dtype=np.float64).values
	test_X = np.c_[test_X, np.ones(len(test_X))]
	test_Y = read_csv('TestDigitY.csv.gz',header=None,dtype=np.float64).values
	test_Y = to_categorical(test_Y)
	return test_X, test_Y

def load_test_data2():
	test_X = read_csv('TestDigitX2.csv.gz',header=None,dtype=np.float64).values
	test_X = np.c_[test_X, np.ones(len(test_X))]
	return test_X

def create_folds(X, Y, n_folds):
	permutation = np.random.permutation(len(X))
	X = X[permutation]
	Y = Y[permutation]

	X_folds = np.array(np.array_split(X, n_folds))
	Y_folds = np.array(np.array_split(Y, n_folds))
	return X_folds, Y_folds

def cross_validate(X, Y, n_folds, params):
	X_folds, Y_folds = create_folds(X, Y, n_folds)
	mask = np.array([True for x in xrange(n_folds)])
	accuracy = []

	for i in xrange(n_folds):
		mask[i] = False
		train_X = np.concatenate(X_folds[mask])
		train_Y = np.concatenate(Y_folds[mask])
		ann = NeuralNetwork(**params)
		ann.fit(train_X, train_Y)
		mask[i] = True

		fold_accuracy = ann.score(X_folds[i], Y_folds[i])
		accuracy.append(fold_accuracy)

	accuracy = np.array(accuracy)
	print params, 'val accuracy: ', accuracy.mean()#, '+/-', 2*accuracy.std()
	return accuracy.mean(), params

def grid_search(X, Y, n_folds=3, show_best=10, **param_grid):
	pool = Pool()
	params = (dict(izip(param_grid, x)) for x in product(*param_grid.itervalues()))
	loop = partial(cross_validate, X, Y, n_folds)
	scores = pool.map(loop, params)

	scores = np.array(scores)
	best_params = scores[np.argmax(scores[:,0])]
	print 'best accuracy: ', best_params[0]
	print best_params[1]

	if show_best:
		print 'top scores: ', scores[np.argsort(scores[:,0])][::-1][:show_best]

	return scores[:,0]

if __name__ == '__main__':
	X, Y = load_training_data()
	test_X, test_Y = load_test_data()
	test_X2 = load_test_data2()
	np.random.seed(0)

	X = np.concatenate([X, test_X])
	Y = np.concatenate([Y, test_Y])

	scores = []
	for decay in np.arange(0, .2, .01):
		params = {'decay':decay}
		accuracy, params = cross_validate(X, Y, 3, params)
		momentum = params['decay']
		scores.append((decay, accuracy))

	scores = np.array(scores)
	plt.plot(scores[:,0], scores[:,1])
	plt.xlabel('decay')
	plt.ylabel('cv accuracy')
	plt.show()

	ann = NeuralNetwork(verbose=True,lr=.15, decay=0.04, 
		nesterov=True,momentum=.4,layer_1_size=64, layer_2_size=128, nb_epochs=10)

	ann.fit(X, Y,validation_split=.2)

	print ann.score(test_X, test_Y)
	predictions = ann.predict(test_X)
	np.savetxt('testX2predictions', predictions)
	ann.plot_history()

