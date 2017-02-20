import numpy as np
from decision_stump import DecisionStump
import matplotlib.pyplot as plt

#AdaBoost class
class BoostingClassifier():

	def __init__(self, featuretbl, iimages, labels):
		self.featuretbl = featuretbl
		self.iimages = iimages
		self.n_examples = len(iimages)
		self.weights = np.empty(len(iimages))
		self.weights[labels == 1] = 1./(2*(labels == 1).sum())
		self.weights[labels == 0] = 1./(2*(labels == 0).sum())
		self.labels = labels

	def boostRound(self):
		weakLearners = []
		self.weights /= self.weights.sum()

		for feature in self.featuretbl:
			rec1 = feature[:4]
			rec2 = feature[2:]

			stump = DecisionStump(rec1, rec2)
			stump.fit(self.iimages, self.labels, self.weights)
			weakLearners.append(stump)

		errors = np.array([learner.error for learner in weakLearners])
		bestLearner = weakLearners[errors.argmin()]
		error = bestLearner.error

		beta = error/(1-error)
		alpha = np.log(1/beta)

		predictions = bestLearner.predict(self.iimages)
		self.featuretbl = np.delete(self.featuretbl, np.argmin(errors), 0)
		self.weights *= np.power(beta, 1 - np.equal(predictions, self.labels))

		return alpha, bestLearner

	def fit(self):
		self.predictors = []
		FPR = 1

		while (FPR > .3):
			new_round = self.boostRound()
			self.predictors.append(new_round)
			FPR = (self.predict(self.iimages[::2]) == 1).sum()/(len(self.labels[::2])*1.)
			print FPR

		return FPR

	def predict(self, x):
		terms = np.array([alpha*(predictor.predict(x) - .8) for alpha, predictor in self.predictors])
		prediction = terms.sum(axis=0) >= 0
		return prediction