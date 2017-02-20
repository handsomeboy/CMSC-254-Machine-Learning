import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

class Isomap():
	def __init__(self):
		data = []
		for line in open('3Ddata.txt'):
			line = line.split()
			x = float(line[0])
			y = float(line[1])
			z = float(line[2])
			color = int(line[3])
			data.append([x, y, z, color])

		data = np.stack(np.array(data))

		self.coordinates = data[:,[0,1,2]]
		self.colors = data[:,3].reshape(len(data),1)

	def construct_graph(self):
		self.distances = np.empty([500, 500])
		for i in xrange(0, 500):
			for j in xrange(0, 500):
				self.distances[i, j] = np.linalg.norm(self.coordinates[i]-self.coordinates[j])

			no_edges = np.argsort(self.distances[i])[11:]
			self.distances[i][no_edges] = np.inf

	def floyd_warshall(self):
		n = len(self.coordinates)
		for k in xrange(0, n):
			for j in xrange(0, n):
				for i in xrange(0, n):
					if self.distances[i, j] > self.distances[i, k] + self.distances[k, j]:
						self.distances[i, j] = self.distances[i, k] + self.distances[k, j]
						
		self.distances = np.square(self.distances)

	def MDS(self): 
		gram = np.empty([500, 500])
		P = np.identity(500) - 1.0/500*np.ones(500)*np.ones(500).T
		G = -.5*P.dot(self.distances).dot(P)
		D, V = np.linalg.eig(G)
		D = np.diagflat(np.append(D[:2], np.zeros(498)))
		self.projected = np.matrix(V)*np.sqrt(D)

	def project_data(self):
		plt.scatter(self.projected[:,0],self.projected[:,1],c=self.colors)
		plt.show()

if __name__ == '__main__':
	isomap = Isomap()
	isomap.construct_graph()
	isomap.floyd_warshall()
	isomap.MDS()
	isomap.project_data()
