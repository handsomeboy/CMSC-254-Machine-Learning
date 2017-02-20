import numpy as np

class DecisionStump():
	def __init__(self, rec1, rec2):
		self.rec1 = rec1
		self.rec2 = rec2

	def computeIntensity(self, iimage):
		#split up rectangle into points 
		p1 = self.rec1[0]
		p2 = self.rec1[1]
		p3 = self.rec1[2]
		p4 = self.rec1[3]
		intensity = iimage[p1[0],p1[1]] - iimage[p2[0], p2[1]] - iimage[p3[0], p3[1]] + iimage[p4[0], p4[1]]

		p1 = self.rec2[0]
		p2 = self.rec2[1]
		p3 = self.rec2[2]
		p4 = self.rec2[3]

		intensity2 = iimage[p1[0],p1[1]] - iimage[p2[0], p2[1]] - iimage[p3[0], p3[1]] + iimage[p4[0], p4[1]]

		return intensity - intensity2

	def calcThreshold(self, values, labels, weights):
		combined = np.array([values, labels])		
		permutation = combined[0].argsort()
		permuted_weights = weights[permutation]
		permuted_combined = combined[:,permutation]
		
		weighted_faces = weights*(permuted_combined[1] == 1)
		weighted_background = weights*(permuted_combined[1] == 0)

		splus = np.cumsum(weighted_faces)
		sminus = np.cumsum(weighted_background)
		tplus = splus[-1]
		tminus = sminus[-1]

		f = lambda x: min(splus[x] + (tminus - sminus[x]), sminus[x] + (tplus - splus[x]))
		errors = map(f, xrange(len(labels)))


		best_threshold = permuted_combined[0][np.argmin(errors)]
		polarity = -1 if splus[np.argmin(errors)] + tminus - sminus[np.argmin(errors)] > sminus[np.argmin(errors)] + tplus - splus[np.argmin(errors)] else 1

		return best_threshold, polarity 

	def fit(self, iimages, labels, weights):
		self.values = [self.computeIntensity(iimage) for iimage in iimages]
		self.threshold, self.polarity = self.calcThreshold(self.values, labels, weights)
		self.error = np.sum(weights*np.abs(self.predict(iimages) - labels))

	def predict(self, iimages):
		intensities = np.array([self.computeIntensity(iimage) for iimage in iimages])
		predictions = self.polarity*(self.threshold - intensities) > 0
		return predictions
