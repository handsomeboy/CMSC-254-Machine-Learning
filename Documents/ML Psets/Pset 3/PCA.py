import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PCA():
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
		self.coordinates = self.coordinates - np.mean(self.coordinates,axis=0)
		self.colors = data[:,3].reshape(len(data),1)

	def calc_components(self):
		sample_mean = np.mean(self.coordinates,axis=0)
		point_var = [np.reshape(x, (3, 1)) for x in self.coordinates]
		sample_covariance = np.cov(self.coordinates.T)
		self.vectors = np.linalg.eig(sample_covariance)[1].T

	def project_data(self):
		projected = np.dot(self.coordinates,np.asmatrix(self.vectors).T)
		plt.scatter(projected[:,0],projected[:,1],c=self.colors)
		plt.show()

	def plot_results(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.coordinates[:,0],self.coordinates[:,1],self.coordinates[:,2],c=self.colors)
		
		for vector in self.vectors:
			ax.quiver(vector[0],vector[1],vector[2],0)
		plt.show()

if __name__ == '__main__':
	pca = PCA()
	pca.calc_components()
	pca.plot_results()
	pca.project_data()
