# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:27:51 2020

@author: adity
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
N=10000
d=10
number_of_selections = [0]*d;
sums_of_rewards = [0]*d;
ads_selected = []
total_rewards = 0
import math

for i in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if(number_of_selections[i]>0):
            average_reward =sums_of_rewards[i]/number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound>max_upper_bound):
            max_upper_bound=upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad]+=1
    reward = dataset[i,ad]
    sums_of_rewards[ad] += reward
    total_rewards += reward

plt.hist(ads_selected)
plt.title('UCB ads selected')
plt.ylabel('Number of times each ad was selected')
plt.xlabel('Ads')
plt.show()