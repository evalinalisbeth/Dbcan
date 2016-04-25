import numpy as np
import pandas as pd
from scipy.spatial import distance
import Queue

def dbscan(point_no):
    global epsilon
    global minpts
    global df_norm
    global n_rows
    global clusters
    global checklist
    global cluster_count
    global core_points_list

    if clusters[point_no] > 0:
        return

    epsilon_neighbors = []

    for i in xrange(n_rows):
        if i != point_no:
            if distance.euclidean(df_norm.iloc[[point_no]], df_norm.iloc[[i]]) <= epsilon:
                epsilon_neighbors.append(i)
    if len(epsilon_neighbors) == 0:
        clusters[point_no] = -1
    else:

        neighbors_with_core_point = False
        for i in epsilon_neighbors:
            if core_points_list[i] == 1:
                neighbors_with_core_point = True
                core_point = i
                break

        if (neighbors_with_core_point or len(epsilon_neighbors) >= minpts):
            if neighbors_with_core_point:
                clusters[point_no] = clusters[core_point]
            else:
                cluster_count += 1
                clusters[point_no] = cluster_count

            if len(epsilon_neighbors) >= minpts:
                core_points_list[point_no] = 1
                for i in epsilon_neighbors:
                    if clusters[i] <= 0:
                        checklist.put(i)

        else:
            clusters[point_no] = -1

    print point_no, clusters[point_no], len(epsilon_neighbors), core_points_list[point_no]
    #print epsilon_neighbors

epsilon = 0.07
minpts = 9

df = pd.read_csv('data/wholesale.csv')
n_rows = len(df)
df.drop(["Channel", "Region"], axis = 1, inplace = True)

df_norm = (df - df.mean()) / (df.max() - df.min())

clusters = [0 for i in xrange(n_rows)]
core_points_list = [0 for i in xrange(n_rows)]
cluster_count = 0

checklist = Queue.Queue()
checklist.put(0)

while 0 in clusters or not checklist.empty():
    if not checklist.empty():
        dbscan(checklist.get())
    else:
        i = 0
        while clusters[i] != 0:
            i+=1
        dbscan(i)

df['clusters'] = clusters
df.to_csv('data/clustering.csv',index=False)