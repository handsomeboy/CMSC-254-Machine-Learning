import matplotlib.pyplot as plt
import matplotlib.patches as patches
from adaboost import BoostingClassifier
from PIL import Image
import numpy as np

#helper functions
def load_test():
	test_image = Image.open('class.jpg')
	return np.array(test_image)

def computeIntegralImages(data):
	length = len(data)
	iimages = [np.cumsum(np.cumsum(data[i], axis=0), axis=1) for i in xrange(length)]

	return np.array(iimages).astype(int)

def load_training(n_examples):
	data = []
	labels = []
	for x in xrange(1, n_examples):
		for folder in ['background', 'faces']:
			condition = 'face' if folder == 'faces' else ''
			filename = folder + '/' + condition + str(x) + '.jpg'
			img = Image.open(filename)
			data.append(np.array(img.convert('L')))

			target = 1 if folder == 'faces' else 0
			labels.append(target)

	iimages = computeIntegralImages(data)
	labels = np.array(labels)

	return iimages, labels

def generateFeatures(stride):
	featuretbl = []
	sizes = [(height, length) for height in xrange(1, 2) for length in xrange(2, 3)]
	for height, length in sizes:
		for i in xrange(0, 64-length):
			c1 = np.array((0, i))
			c2 = np.array((0, i+length))
			c3 = np.array((height, i))
			c4 = np.array((height, i+length))
			c5 = np.array((2*height, i))
			c6 = np.array((2*height, i+length))
			featuretbl.append(np.array([c1, c2, c3, c4, c5, c6]))

			for j in xrange(0, 64 - stride - height*2, stride):
				c1 += (stride, 0)
				c2 += (stride, 0)
				c3 += (stride, 0)
				c4 += (stride, 0)
				c5 += (stride, 0)
				c6 += (stride, 0)
				featuretbl.append(np.array([c1, c2, c3, c4, c5, c6]))

			c1 = np.array((i, 0))
			c2 = np.array((i+length, 0))
			c3 = np.array((i, height))
			c4 = np.array((i+length, height))
			c5 = np.array((i, 2*height))
			c6 = np.array((i+length, 2*height))
			featuretbl.append(np.array([c1, c2, c3, c4, c5, c6]))

			for j in xrange(0, 64 - stride - height*2, stride):
				c1 += (0, stride)
				c2 += (0, stride)
				c3 += (0, stride)
				c4 += (0, stride)
				c5 += (0, stride)
				c6 += (0, stride)

				featuretbl.append(np.array([c1, c2, c3, c4, c5, c6]))

	return np.array(featuretbl)

def train(n_classifiers, iimages,labels, featuretbl):
	classifiers = []
	for i in xrange(n_classifiers):
		clf = BoostingClassifier(featuretbl, iimages, labels)
		print 'fitting classifier', i + 1
		FPR = clf.fit()
		classifiers.append(clf)
		# positive_predictions = (clf.predict(iimages) == 1)
		# if (sum(positive_predictions) == 0) | (FPR == 0):
		# 	break
		# else:
		# 	print sum(positive_predictions)
		# 	iimages = iimages[positive_predictions]
		# 	labels = labels[positive_predictions]

	return classifiers

def non_maximal_suppression(boxes, length, height, ax):
	x = boxes[:,1]
	y = boxes[:, 0]
	for i in xrange(0, height - 70, 70):
		for j in xrange(0, length - 127, 70):
			smaller_x = x[(i <= x) & (x <= i + 70)]
			smaller_y = y[(j <= y) & (y <= j + 70)]
			if smaller_x.size and smaller_y.size:
				new_x_1 = max(smaller_x)
				new_x_2 = min(smaller_x)
				new_y_1 = max(smaller_y)
				new_y_2 = min(smaller_y)
				rect = patches.Rectangle((new_x_1,new_y_1),new_x_1 - new_x_2,new_y_1-new_y_2,linewidth=1,edgecolor='r',facecolor='none')
				ax.add_patch(rect)

def test(classifiers, test_image, stride):
	fig,ax = plt.subplots(1)
	ax.imshow(test_image,cmap='gray')
	dropped = [0 for x in classifiers]
	faces = []
	height = test_image.shape[0]
	length = test_image.shape[1]
	for j in xrange(0, length - 63, stride):
		for i in xrange(0, height - 63, stride):
			test_square = test_image[i:i+64, j:j+64]
			test_iimage = computeIntegralImages([test_square])
			
			#cycle through cascade
			face = True
			for k in xrange(len(classifiers)):
				clf = classifiers[k]
				prediction = clf.predict(test_iimage)[0]
				if ~prediction:
					dropped[k] += 1
					face = False
					break
			
			if face:
				faces.append([i, j])
				# rect = patches.Rectangle((j,i),64,64,linewidth=1,edgecolor='r',facecolor='none')
				# ax.add_patch(rect)

	print dropped
	non_maximal_suppression(np.array(faces), length, height, ax)
	plt.show()

if __name__ == '__main__':
	n_examples = 300
	iimages, labels = load_training(n_examples)
	featuretbl = generateFeatures(stride=4)
	testImage = load_test()
	classifiers = train(4, iimages, labels, featuretbl)
	test(classifiers, testImage, 2)