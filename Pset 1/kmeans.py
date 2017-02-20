import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class Cluster():
	def __init__(self, centroid=None, points=[]):
		self.points = points
		self.centroid = centroid
	
	def find_centroid(self):
		self.centroid = np.sum(self.points,axis=0)/len(self.points) if self.points else [0, 0]

class Algorithm():

	def __init__(self, n_clusters):
		self.data = []
		self.n_clusters = n_clusters

	def load_data(self):
		for line in open('toydata.txt'):
			line = line.split()
			x = float(line[0])
			y = float(line[1])
			self.data.append([x, y])

		self.data = np.array(self.data)

	def load_data2(self):
		self.data = np.array([[-30, -1], 
								[-30, 1],
								[30, -1],
								[30, 1]])

	def calculate_centroids(self):
		for cluster in self.clusters:
			cluster.find_centroid()

	def initialize(self):
		centroids = np.random.randint(1, len(self.data),size=self.n_clusters)
		self.clusters = [Cluster(centroid=self.data[centroid]) for centroid in centroids]

	def initialize_plusplus(self):
		initial_center = self.data[np.random.randint(0, len(self.data))]
		centroids = np.array([initial_center,])
		for k in xrange(0, self.n_clusters-1):
			distances = []
			for point in self.data:
				d1 = min([np.linalg.norm(point - centroid) for centroid in centroids])
				distances.append(d1)

			probabilities = distances/sum(distances)
			indices = np.arange(len(self.data))
			choice = np.random.choice(indices, p=probabilities)
			new_centroid = np.array([self.data[choice]])
			centroids = np.concatenate((centroids, new_centroid),axis=0)

		self.clusters = [Cluster(centroid=centroid) for centroid in centroids]

	def cluster_points(self):
		centroids = [cluster.centroid for cluster in self.clusters]
		self.clusters = [Cluster() for x in xrange(0, self.n_clusters)]
		new_clusters = defaultdict(list)
		for point in self.data:
			new_assignment = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])
			new_clusters[new_assignment].append(point)
			self.clusters[new_assignment].points = new_clusters[new_assignment]

	def calc_distortion(self):
		distortion = 0
		for cluster in self.clusters:
			for point in cluster.points:
				distortion += np.linalg.norm(point - cluster.centroid)**2

		return distortion

	def run_kmeans(self, max_iter=50):
		distortion = []
		for i in xrange(1, max_iter):
			old_centroids = np.array([cluster.centroid for cluster in self.clusters])
			self.cluster_points()
			self.calculate_centroids()
			distortion.append(self.calc_distortion())
			new_centroids = np.array([cluster.centroid for cluster in self.clusters])
			if np.array_equal(new_centroids, old_centroids):
				break

		return distortion

	def plot_results(self):
		plt.figure()
		for cluster in self.clusters:
			x = np.vstack(cluster.points)[:,0]
			y = np.vstack(cluster.points)[:,1]
			plt.plot(x, y,'o')

		plt.show()

	def plot_distortion(self):
		plt.figure()
		for run in xrange(1, 20):
			self.initialize()
			distortion = self.run_kmeans()
			plt.plot(distortion)

		plt.title('Iteration vs Distortion')
		plt.xlabel('Iterations')
		plt.ylabel('Distortion')
		plt.show()

	def compare_initialization(self):
		distortions_ratio = []
		dist_rand = np.array([])
		dist_pp = np.array([])

		for run in xrange(1, 500):
			self.initialize()
			self.run_kmeans()
			dist_rand = np.append(dist_rand, self.calc_distortion())
			self.initialize_plusplus()
			self.run_kmeans()
			dist_pp = np.append(dist_pp,self.calc_distortion())

		ratios = dist_rand/dist_pp
		correction_term = np.cov(ratios, dist_pp)[0, 1]/dist_pp.mean()
		mean = sum(dist_rand)/sum(dist_pp) - correction_term
		variance = dist_rand.mean()/dist_pp.mean() * (dist_rand.std()/dist_pp.mean()**2 + dist_pp.std()/dist_rand.mean()**2 - 2*correction_term/dist_rand.mean())
		print 'Improvement: ', mean, '+/-', 1.96*variance
			
if __name__ == '__main__':
	clustering = Algorithm(n_clusters=3)
	clustering.load_data()
	#clustering.load_data2()

	# run one of the following two initializations
	#clustering.initialize()
	clustering.initialize_plusplus()

	#main execution
	clustering.run_kmeans()

	#plotting functions
	clustering.plot_distortion()
	#clustering.plot_results()

	#extra credit
	#clustering.compare_initialization()
