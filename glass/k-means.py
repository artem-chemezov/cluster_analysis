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


dataset = pd.read_csv('../../glass.csv').values[:,1:]
types = pd.read_csv('../../glass.csv').values[:,:1]
sarr = [0] * dataset.shape[0]
k_out = 0
#print(dataset)
for k in range(2, dataset.shape[0]):
	cl = KMeans(k).fit(dataset)
	lb = cl.labels_
	sarr[k] = silhouette_score(dataset, lb, metric="euclidean", sample_size=None, random_state=None)
	if sarr[k] > sarr[k_out]:
		lb_out = lb
		k_out = k
	#print(k, "     ", silhouette_score(dataset, lb, metric="euclidean", sample_size=None, random_state=None))

output = np.vstack((range(dataset.shape[0]), sarr)).T
writer = csv.writer(open("output_kmeans_evaluation.csv", 'w'))
for k in output:
    writer.writerow(k)
output = np.vstack((dataset.T, lb_out.T)).T
output = np.hstack((types, output))
writer = csv.writer(open("output_kmeans_lables.csv", 'w'))
for k in output:
    writer.writerow(k)


