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

#using dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method ='ward'))
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.title('Dendrogram')
plt.show()

#fitting data
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#visualizing observations
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s=100,c='blue',label='Cluster2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s=100,c='yellow',label='Cluster4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s=100,c='purple',label='Cluster5')


plt.title('Client Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score') 
plt.legend()
plt.show()