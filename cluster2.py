"""
K MEANS CLUSTERING ALGORITHM
-> Clustering the data into k different clusters
-> using unsupervised learning as the dataset is unlabeled
-> Each sample is assigned to the cluster with the nearest mean

"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA

# so that we can reproduce the data later
np.random.seed(42)

# to find th Euclidean Distance between two vectors
# this will help in calculating the distance between each data point and cluster centers


def Euclidean_Dist(x1, x2):
    #d = np.sqrt(np.sum((x1-x2)**2))
    d = np.linalg.norm(x1-x2) 
    return d

# takes two vectors and returns the cosine similarity between them
def Cosine_Sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product/(norm_a*norm_b)


class KMeans:
    num_plots = 0
    # setting default values for the class
    # if k value not provided by user, k will be 5 and no of iterations will be 100

    def __init__(self, K=5, max_iters=100, plot_steps=False):

        # initializing values
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps


        # list of sample indices for each cluster
        # for each cluster we initialize an empty list
        self.clusters = [[] for _ in range(self.K)]
        # mean feature vector for each cluster (actual samples)
        self.centroids = []

    # this method will input a list of chosen centroids
    # and assign each data point/sample to its nearest centroid and return the clusters created
    def _create_clusters(self, centroids):
        # creating an empty list of K-lists for clusters
        clusters = [[] for _ in range(self.K)]
        for index, sample in enumerate(self.X):
            # find the nearest centroid for current sample
            #print(index,sample)
            centroid_index = self._nearest_centroid(sample, centroids,index)
            clusters[centroid_index].append(index)
        print("Clusters = ",clusters)
        return clusters

    # returns the index of centroid in list which is nearest to the given sample
    def _nearest_centroid(self, sample, centroids,index):
        dist = [Euclidean_Dist(sample, c) for c in centroids]
        print("sample",sample)   
        print("dist = ",dist)     
        nearest_centroid = np.argmin(dist)
        print("min dist = ",min(dist))
        print('centroid for this sample with index = ',nearest_centroid,sample,index)
        return nearest_centroid

    def get_new_centroids(self, clusters):
        # creating an array filled with zeroes with tuple of K and number of features
        centroids = np.zeros((self.K, self.n_features))
        # calculating the new centroid as the mean of all samples in the cluster
        # clusters is the list of lists
        for cluster_index, cluster in enumerate(clusters):
            # finding mean of samples in the current cluster
            cluster_mean = np.mean(self.X[cluster], axis=0)
            print("Xcluster =",self.X[cluster])
            centroids[cluster_index] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        dist = [Euclidean_Dist(old_centroids[i], new_centroids[i]) for i in range(self.K)]
        # if there is no change in the distances of the old and new centroids, means the algorithm has converged
        if(sum(dist) == 0): 
            return True
        return False

    def _get_cluster_labels(self, clusters):
        # creating an empty NumPy array for storing the label of each sample
        cluster_labels = [-1 for _ in range(self.n_samples)]
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                cluster_labels[sample_index] = int(cluster_index)
        #for i in range(len(cluster_labels)):
         #   cluster_labels[i] = "Label-"+str(cluster_labels[i])
        return cluster_labels

    def predict(self, X):
        # for storing data
        self.X = X.toarray()
        self.Y = X
        # number of samples and features
        #print(X.shape)
        self.n_samples, self.n_features = X.shape
        print("SAMPLES:", self.n_samples, self.n_features)


        # initialize the centroids
        # To randomly pick some samples
        # it will pick a random choice between 0 and number of samples
        # In KMeans algorithm initially, random samples are made centroids and gradually optimization is done and new centroids selected
        # this will be an array of size self.K
        # print(self.X)        
        random_sample_indices = np.random.choice( self.n_samples, self.K, replace=False)       
        print("RANDOM SAMPLE INDICES = ", random_sample_indices)
        self.centroids = [self.X[i] for i in random_sample_indices]
        print("CENTROIDS = ",self.centroids)
       
        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            

            if self.plot_steps:
                self.plot_data()

            # update centroids
            old_centroids = self.centroids
            self.centroids = self.get_new_centroids(self.clusters)
            print("NEW CENTROIDS : ",self.centroids)
            
            if self.plot_steps:
                self.plot_data()

            # checking for convergence of algorithm
            if self._is_converged(old_centroids, self.centroids):
                # we can end the clustering algorithm now
                print("CONVERGED")
                break

        # return cluster_labels
        return self._get_cluster_labels(self.clusters)

    def plot_data(self):
        labels_color_map = {
            0: 'blue', 1: 'green', 2: 'purple', 3: 'red', 4: 'yellow',
            5: 'orange', 6: 'pink', 7: 'cyan', 8: 'magenta', 9: 'black' 
        }
        pca_num_components = 2
        reduced_data = PCA(n_components=pca_num_components).fit_transform(self.Y.todense())
        reduced_d2 = PCA(n_components=pca_num_components).fit_transform(self.centroids)
        figure, ax = plt.subplots(figsize=(8,12))
        for i,index in enumerate(self.clusters):            
            point  = reduced_data[index].T
            cen = reduced_d2[i].T
            ax.scatter(*point,c=labels_color_map[i])     
            #ax.scatter(*cen,c=labels_color_map[i],marker="x",linewidth=2,legend="Centroid") 
            ax.scatter(*cen,marker="x",c='black',linewidth=2)  
        #plt.legend(num_points=1)
        plt.show()        
        
        
        
        

       

    
   
       
