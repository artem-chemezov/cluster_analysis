import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, NullFormatter


dataset = pd.read_csv('../../bronze.csv').values[:,1:]
types = pd.read_csv('../../bronze.csv').values[:,:1]
sarr = [0] * 100
k_out = 0
indexes = [0] * 100
#print(dataset)
for k in range(1, 100):
	cl = DBSCAN(eps=k, min_samples=6).fit(dataset)
	lb = cl.labels_
	if (1 in lb):
		sarr[k] = silhouette_score(dataset, lb, metric="euclidean", sample_size=None, random_state=None)
		indexes[k] = np.unique(lb).shape[0]
	else:
		sarr[k] = 0
		indexes[k] = 1
	if sarr[k] > sarr[k_out]:
		lb_out = lb
		k_out = k
	#print(k, "     ", silhouette_score(dataset, lb, metric="euclidean", sample_size=None, random_state=None))

output = np.vstack((range(100), sarr, indexes)).T
writer = csv.writer(open("output_DBSCAN_evaluation.csv", 'w'))
for k in output:
    writer.writerow(k)
output = np.vstack((dataset.T, lb_out.T)).T
output = np.hstack((types, output))
writer = csv.writer(open("output_DBSCAN_labels.csv", 'w'))
for k in output:
    writer.writerow(k)


