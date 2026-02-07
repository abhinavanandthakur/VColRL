#this file contains all the utility functions required to run the Fastcolor algorithm


import random
import math
import networkx as nx  

#function which takes a graph as input and returns the adjacency list representation of the graph
def convert_to_list_of_neighbors(G):
    return {node: list(G.neighbors(node)) for node in G.nodes()}

#function which takes a graph as input and returns the mapping of nodes to 1,2,...,n
def node_mapping(G):
    old_labels=G.nodes()
    mapping={old:new for new,old in enumerate(old_labels,start=1)}
    G_new=nx.relabel_nodes(G, mapping)
    return G_new, mapping

#function which takes a graph as input and returns the mapping of nodes to 0,1,...,n-1
def node_mapping_kernel(G):
    old_labels=G.nodes()
    mapping={old:new for new,old in enumerate(old_labels,start=0)}
    G_new=nx.relabel_nodes(G, mapping)
    return G_new, mapping

#function which takes a graph as input and returns the core decomposition of the graph
def core_decomposition(g):

    "Assumes that the vertex are numbered from 1 to n"   
    vert=[0]*(g.number_of_nodes()+1) #indexing will start with 1
    pos=[0]*(g.number_of_nodes()+1) #indexing will start with 1
    n=g.number_of_nodes()
    
    deg=[0]*(g.number_of_nodes()+1) #indexing will start with 1
    bin=[0]*(g.number_of_nodes()) #degree will start with 0 bins of degree
    md=0
    for node, degree in g.degree():
        deg[node] = degree #calculating degree for each node
        bin[degree] += 1 #incrementing the bin for the degree
        if degree>md:
            md=degree
    bin=bin[:md+1] #slicing the bin uptill the maximum degree
    start=1
    for d in range(0,md+1):
        num=bin[d]
        bin[d]=start
        start+=num #incrementing the start for the next bin
        
    for v in range(1,n+1):
        pos[v]=bin[deg[v]] #position of the vertex in the bin
        vert[pos[v]]=v #vertex at the position of the vertex in the bin
        bin[deg[v]]+=1 #incrementing the bin for the degree
    
    for d in range(md,0,-1):
        bin[d]=bin[d-1] #shifting the bin
    bin[0]=1 #setting the first bin to 1
    
    for i in range(1,n+1):
        v=vert[i] #vertex at the position of the vertex in the bin
        for u in g.neighbors(v):
            if deg[u]>deg[v]: 
                du=deg[u] 
                pu=pos[u] 
                pw=bin[du]
                w=vert[pw] 
                
                if u != w:
                    pos[u]=pw
                    vert[pu]=w
                    pos[w]=pu
                    vert[pw]=u
                bin[du]+=1
                deg[u]-=1
                
    return deg[1:] #sliced the first positionas we now map from node starting from 0

#function which takes a graph and a lower bound as input and returns the lowest bounded of the graph
def FindLBIS(G,lb):
    
    LBIS=set()
    
    for i in G.nodes():
        if G.degree(i)<lb and len(LBIS.intersection(set(G.neighbors(i))))==0:
            LBIS.add(i)
    return LBIS

#fuction to find a clique as described in the paper      
def FindClq(G,lb,t=1):
    lb_old=lb
    startset_size=math.ceil(G.number_of_nodes()/100)
    startset=set(random.sample(G.nodes(), startset_size))
    t_max=64
    t=1
    while len(startset)>0:
        u=startset.pop()
        C=set()
        C.add(u)
        CS=set(G.neighbors(u))
        
        while len(CS)>0:
            sample_CS=random.sample(list(CS), t)
            v=sample_CS[0]
            max_int=0
            for i in sample_CS:
                NV=set(G.neighbors(i))
                CS_NV=CS.intersection(NV)
                if len(CS_NV)>max_int:
                    max_int=len(CS_NV)
                    v=i
            NV=set(G.neighbors(v))
            if len(C)+1+len(CS.intersection(NV))<=lb:
                break
            C.add(v)
            CS.remove(v)
            CS=CS.intersection(NV)
            
        if len(C)>lb:
            lb=len(C)
            
    if lb_old==lb:
        return True, lb   
    else:    
        return False, lb
    
    
