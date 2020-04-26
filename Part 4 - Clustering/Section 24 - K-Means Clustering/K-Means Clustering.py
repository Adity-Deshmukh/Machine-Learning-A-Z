# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:07:59 2020

@author: adity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#using elbow method

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

#applying kmeans with proper no of clusters
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualizing observations
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100,c='yellow',label='Cluster4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100,c='purple',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='cyan',label='ClusterCenter')

plt.title('Client Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score') 
plt.legend()
plt.show()