import numpy as np
import pandas as pd
import plotly
from scipy.spatial import distance
import Queue
import plotly.graph_objs as go


def DBSCAN(D, eps, MinPts):
    C = 0
    NeighborPts = []
    global clusters
    visited = [0 for i in xrange(len(D))]
    for P in xrange(len(D)):
        if visited[P] != 0:
            continue
        visited[P]=1
        NeighborPts = regionQuery(P,eps,D)
        if len(NeighborPts)<MinPts:
            visited[P] = 1
            clusters[P] = -1
            #print P, len(NeighborPts), clusters[P]
        else:
            C += 1
            expandCluster(P,NeighborPts,C,eps,MinPts,visited,D)

def expandCluster(P,NeighborPts,C,eps,MinPts,visited,D):
    global clusters
    clusters[P]=C
    i = 0
    #print P, len(NeighborPts), clusters[P]
    while i < len(NeighborPts):
        p = NeighborPts[i]
        neighborpts = []
        if visited[p] == 0:
            visited[p] = 1
            neighborpts = regionQuery(p,eps,D)
            if len(neighborpts) >= MinPts:
                NeighborPts = list(set(NeighborPts + neighborpts))
        if clusters[p]<=0:
            clusters[p]=C
            #print p, len(neighborpts), clusters[p]
        i+=1

def regionQuery(P, eps, D):
    NeighborPts = []
    for i in xrange(len(D)):
        if distance.euclidean(D.iloc[[P]], D.iloc[[i]]) <= eps:
            NeighborPts.append(i)
    return NeighborPts

epsilon = 0.025
minpts = 5

df = pd.read_csv('data/wholesale.csv')
n_rows = len(df)
df.drop(["Channel", "Region"], axis = 1, inplace = True)
df_norm = (df - df.mean()) / (df.max() - df.min())

clusters = [0 for i in xrange(len(df_norm))]

DBSCAN(df_norm,epsilon,minpts)
df['clusters'] = clusters
df.to_csv('clustering.csv',index=False)
print len(set(df['clusters']))

noise=[(df.clusters==-1)]
c=[(df.clusters==1)]

trace_comp0 = go.Scatter(
    x=noise[0],
    y=noise[1],
    mode='markers',
    marker=dict(size=12,
                line=dict(width=1),
                color="navy"
               ),
    name='Americas',

    )

trace_comp1 = go.Scatter(
    x=c[0],
    y=c[1],
    mode='markers',
    marker=dict(size=12,
                line=dict(width=1),
                color="red"
               ),
    name='Americas',

    )

data_comp = [trace_comp0, trace_comp1]

#fig_comp = go.Figure(data=data_comp, layout=layout_comp)
plotly.offline.plot(data_comp, filename='life-expectancy-per-GDP-2007')
