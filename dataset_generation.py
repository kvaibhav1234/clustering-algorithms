
import numpy as np
from sklearn.decomposition import PCA
import math

#Variables - Clustering
N = 60   #edges = n*dl = m*dr = N
k = 4    #num of clusters ---> n = k 
#Variables - Bipartite Graph
dl = 15   #degree left side
dr = 2   #degree right side
n = 4    #left side
m = 30   #right side


data = np.zeros((N, n+m+1)) # last column for labels

right_deg = np.array([dr]*m) 
left_deg = np.array([dl]*n) 

# print("Left Degree:", left_deg)
# print("Right Degree:", right_deg)


for i in range(N): 
    y = np.random.choice(n, p=left_deg / np.sum(left_deg)) # random vertex from left set
    x = np.random.choice(m, p=right_deg / np.sum(right_deg)) # random vertex from right set
    # print("y, x+N: ", y,x+m)
    
    data[i][x+n] = 1
    data[i][y] = 1
    data[i][-1] = y

    right_deg[x] -= 1
    left_deg[y] -= 1
    

""""   
row = np.array([N*d])
data = np.append(right_deg,[row],axis=0)

col = np.array([Nd*2*N+1])
data = np.append(left_deg,[col],axis=0)
"""

# print("Dataset:", data)

filename = "dataset"
np.savetxt(filename + ".txt", data, fmt = '%u', delimiter=" ")


##### Dimensionality reduction
PCA_dim = 20
reduced_data = PCA(n_components= PCA_dim).fit_transform(data[:, 0:-1])

filename = "reduced_dataset"
np.savetxt(filename + ".txt", np.concatenate((reduced_data, (data[:, -1]).reshape(-1, 1)), axis = 1), delimiter=" ")

