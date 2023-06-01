
from sklearn.cluster import AgglomerativeClustering 

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
	
 
  #Hierarchical Clustering: 
  clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(data)   


predicted_labels = labelling(clustering.fit_predict(data), labels, n_clusters, n_samples)
accuracy = accuracy_score(labels, predicted_labels)
accuracy_list.append(accuracy)

accuracy_analysis(accuracy_list)

clustering.labels_






