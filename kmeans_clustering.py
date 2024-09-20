import numpy as np
import argparse
import scipy.io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class MykmeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        
        self.dataset_file = dataset_file
        self.read_mat()

    def read_mat(self):
        # Load the .mat file
        mat = scipy.io.loadmat(self.dataset_file)
        print("Keys in the .mat file:", mat.keys())

        self.data = mat['X']

        
    def model_fit(self):
        '''
        initialize self.model here and execute kmeans clustering here
        '''
        
        n_clusters = 3  
        max_iter = 200 

        # Initialize the KMeans model 
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        
        # Fit model w/ data
        self.model.fit(self.data)
        
        # Get the cluster centers
        cluster_centers = np.array(self.model.cluster_centers_)
        return cluster_centers
    
    def plot_clusters(self):
        cluster_centers = self.model_fit()  
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], s=50, c='gray', label='Data points')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='pink', marker='*', label='Centroids') 
        plt.title('Visualization of Clusters and Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kmeans clustering')
    parser.add_argument('-d','--dataset_file', type=str, default = "dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MykmeansClustering(args.dataset_file)
    clusters_centers = classifier.model_fit()
    print(clusters_centers)
    classifier.plot_clusters()
    