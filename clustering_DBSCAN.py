

from sklearn.cluster import DBSCAN

from cluster_labelling import labelling,accuracy_analysis


import numpy as np
from sklearn.datasets import load_digits       

from sklearn.metrics import accuracy_score


#data, labels = load_digits(return_X_y=True)  

dataset = "Dataset_N1200.txt"
print(dataset)
data = np.loadtxt(dataset)
labels = data[:, -1].astype(int) 
data = data[:, 0:-1]



(n_samples, n_features),n_clusters = data.shape, np.unique(labels).size
print(f"# digits: {n_clusters}; # samples: {n_samples}; # features {n_features}")



accuracy_list = []
no_epochs = 5

for _ in range(no_epochs):
	
  
  #DBSCAN: 
  
  #clustering = DBSCAN(eps=200000, min_samples=20).fit(data)  
  #clustering = DBSCAN(eps=0.00000000000001, min_samples=50).fit(data)
  #clustering = DBSCAN(eps=0.5, min_samples=10).fit(data)
  #clustering = DBSCAN(eps=0.001, min_samples=40).fit(data)
  #clustering = DBSCAN(eps=1.00002, min_samples=344444).fit(data)
  #clustering = DBSCAN(eps=10, min_samples=50).fit(data)
  #clustering = DBSCAN(eps=30, min_samples=50).fit(data)     ------------------------> 13%
  #clustering = DBSCAN(eps=20, min_samples=50).fit(data)     ------------------------> 19%
  #clustering = DBSCAN(eps=40, min_samples=50).fit(data)
  #clustering = DBSCAN(eps=30, min_samples=60).fit(data)     ------------------------> 22%
  #clustering = DBSCAN(eps=30, min_samples=90).fit(data)      ------------------------> 37%
  clustering = DBSCAN(eps=30, min_samples=85).fit(data)       #-------------------------> 41%
  #clustering = DBSCAN(eps=25, min_samples=85).fit(data)       ---------------------------> 27%
  #clustering = DBSCAN(eps=30, min_samples=70).fit(data)       ----------------------------> 21%
  #clustering = DBSCAN(eps=60, min_samples=80).fit(data) 


predicted_labels = labelling(clustering.fit_predict(data), labels, n_clusters, n_samples)
accuracy = accuracy_score(labels, predicted_labels)
accuracy_list.append(accuracy)

accuracy_analysis(accuracy_list)


clustering.labels_

