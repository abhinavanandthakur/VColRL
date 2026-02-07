#This version of FastColor is implemented based on the pseudocode provided in the original paper and may not be its exact impementation
#Lin, Jinkun, et al. "A reduction based method for coloring very large graphs." IJCAI-17. International Joint Conferences on Artifical Intelligence (IJCAI), 2017.

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from time import time
from utils import  node_mapping, node_mapping_kernel, core_decomposition, FindLBIS, FindClq
import os
from tqdm import tqdm
import random
import glob
import statistics
from load_dataset import read_data
import argparse
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='er_50_100')
args=parser.parse_args()

cutoff_time=60

#recolor function of fastcolor to avoid increasing the color number
def recolor(G,v,coloring):
    new_color=coloring[v]
    conflicts=[0]*(new_color+1)
    used=[-1]*(new_color+1) 
    for w in G.neighbors(v):
        if coloring[w]!=0:
            conflicts[coloring[w]]+=1
            used[coloring[w]]=w 
    
    for i in range(1,new_color):
        if conflicts[i]==1:
            w=used[i]
            for u in G.neighbors(w):
                if coloring[u]!=0:
                    used[coloring[u]]=w     
            c=0 
            for j in range(i+1,new_color):
                if (used[j]!=w):
                    c=j
                    break
                
            if c!=0:
                coloring[v]=coloring[w]
                coloring[w]=c
                return True, coloring

    return False, coloring


#function to find the saturation degree
def find_saturation_degree(G, coloring):
    sat_degree = [0] * G.number_of_nodes()
    for node in G.nodes():
        color_neighbors=set()
        for neighbor in G.neighbors(node):
            if coloring[neighbor] != 0:
                color_neighbors.add(coloring[neighbor])
        sat_degree[node] = len(color_neighbors)
    return sat_degree

#function to update saturarion degree as we do not want to find the saturation degree of all the nodes again and again 
def update_saturation_degree(G, coloring, node, sat_degree):
    # Update the saturation degree of the neighbors of the given nodemax_color_used
    for neighbor in G.neighbors(node):
        color_used=set()
        for nodes in G.neighbors(neighbor):
            if coloring[nodes] != 0:
                color_used.add(coloring[nodes])
        sat_degree[neighbor] = len(color_used)  
    return sat_degree

#color kernel function as described in the fastcolor paper
def ColorKernel(G_main, is_colored=False):
    G,_=node_mapping_kernel(G_main)
    G_core,_=node_mapping(G)
    coloring= [0]*G.number_of_nodes()
    max_color_used=0
    #G_core,_=node_mapping(G)
    if is_colored==False:
        cores=core_decomposition(G_core)
        sorted_cores = sorted(range(len(cores)), key=lambda i: cores[i], reverse=True)
        for v in sorted_cores:
            #find the value i>0 such that no neighbor of u has taken color i 
            ct=1
            neighbor_coloring=set()
            for u in G.neighbors(v):
                if coloring[u]!=0:
                    neighbor_coloring.add(coloring[u])
            for i in range(1, G.number_of_nodes()+1):
                if i not in neighbor_coloring:
                    ct=i
                    break
                
            coloring[v]=ct
            if ct>max_color_used:
                status,coloring=recolor(G,v,coloring)
                if status==False:
                    max_color_used=ct
        
    else:
        V_set=list(G.nodes())
        saturation_degree=[0]*G.number_of_nodes()
        while len(V_set)>0:
            #saturation_degree=find_saturation_degree(G, coloring)
            max_saturaion_deg=0
            max_indices=[]
            for i in V_set:
                if saturation_degree[i]>max_saturaion_deg:
                    max_saturaion_deg=saturation_degree[i]
                    max_indices=[]
                    max_indices.append(i)
                elif saturation_degree[i]==max_saturaion_deg:
                    max_indices.append(i)
                     
            v=random.choice(max_indices)
            V_set.remove(v)

            ct=1
            neighbor_coloring=set()
            for u in G.neighbors(v):
                if coloring[u]!=0:
                    neighbor_coloring.add(coloring[u])
            for i in range(1, G.number_of_nodes()+1):
                if i not in neighbor_coloring:
                    ct=i
                    break
                
            coloring[v]=ct   
            if ct>max_color_used:
                status,coloring=recolor(G,v,coloring)
                if status==False:
                    max_color_used=ct
            saturation_degree=update_saturation_degree(G,coloring,v,saturation_degree)
                    
    return coloring



#construction function to construct the new solution after reduction
def construct_solution(G_k, G_m, e_m, alpha, lb_tracker, num_nodes):
    #this constructrion takes assumption that alpha is the best solution for G_k
    if len(G_m)==0: #if there is no reduction alpha will be the constructed solution
        return alpha
    
    alpha_plus=[0]*num_nodes #initializing the solution
    #making copies sot that the original values are not changed
    alpha=alpha.copy()
    lb_tracker=lb_tracker.copy()
    G_k=G_k.copy()
    G_m=G_m.copy()
    e_m=e_m.copy()
    
    #reoving the last entry of the lower bound tracker
    if len(G_m)<len(lb_tracker):
        lb_tracker=lb_tracker[:len(G_m)]
    
    assert len(G_m)==len(e_m)==len(lb_tracker), f"length of G_m and e_m lb_tracker should be same ====> {len(G_m)} {len(e_m)} {len(lb_tracker)}"
    
    for i in G_k.nodes():
        alpha_plus[i]=alpha.pop(0)
        
    for _ in range(len(G_m)):
        
        node_set = G_m.pop()
        edge_set = e_m.pop()
        lower_bound_prev=lb_tracker.pop()
        
        chromatic_num_current=max(alpha_plus)
        
        #construction of graph
        G_k.add_nodes_from(node_set)
        G_k.add_edges_from(edge_set)
        
        #assert chromatic_num_current<=lower_bound_prev
        
        if chromatic_num_current<lower_bound_prev:
            new_color=max(alpha_plus)+1
            for node in node_set:
                alpha_plus[node]=new_color
                
        elif chromatic_num_current>=lower_bound_prev:
            for node in node_set:
                color_ued_by_neighbors=set()
                for neighbor in G_k.neighbors(node):
                    if alpha_plus[neighbor]!=0:
                        color_ued_by_neighbors.add(alpha_plus[neighbor])
            
                c=1
                
                for color_iter in range(1, chromatic_num_current+1):
                    if color_iter not in color_ued_by_neighbors:
                        c=color_iter
                        break
                alpha_plus[node]=c
        
    return alpha_plus

#main fastcolor function
def FastColor(G, cutoff_time=60):
    
    G_k=G.copy()
    G_m=[]
    e_m=[]
    lb_G=0
    lb_tracker=[]
    ub_G=G.number_of_nodes()
    lb_tracker.append(0)
    alpha_best=[0]*G.number_of_nodes()
    lb_k=0
    isColored=False
    t=1
    time_start=time()
    ub_tracker=[]
    ub_tracker.append(ub_G)
    while 1:
        bms_param_adjustment_required,lb_k=FindClq(G_k, lb=lb_k, t=t)
        if bms_param_adjustment_required:
            t=2*t
        if t>64:
            t=1
        if lb_k>lb_G:
            lb_G=lb_k
            
        lb_tracker[-1]=lb_k #we are overriding the current reduced graph's lower bound
        I=FindLBIS(G_k, lb_k)
        edges_before_reduction=G.edges()
        removed_edges=set() #keeps track for removed edges in this iteration
        remaining_nodes=set(G_k.nodes()).difference(I)
        G_k=G_k.subgraph(remaining_nodes) #this is new Gk but the nodes will not be renumbered
      
        for ed in edges_before_reduction:
            if ed not in G_k.edges():
                removed_edges.add(ed)
                
        #if there is an independent set independednt set, then we need to reset some values as the graph is reduces 
        if len(I)!=0:
            lb_k=0
            isColored=False
            G_m.append(I)
            e_m.append(removed_edges)
            lb_tracker.append(lb_k) #adding a new entry for the lower bound of reduced graph 
        
        alpha=ColorKernel(G_k, is_colored=isColored) 
        alpha_plus=construct_solution(G_k, G_m, e_m, alpha,lb_tracker,G.number_of_nodes()) 
        isColored=True
       
        if max(alpha_plus)<ub_G:  
            alpha_best=alpha_plus.copy()
            ub_G=max(alpha_plus)
            ub_tracker.append(max(alpha_plus))
            
            
        time_elapsed=time()-time_start
        if ub_G==lb_G: 
            return max(alpha_best), time_elapsed
        
        if time_elapsed>cutoff_time:
            break
        
    return max(alpha_best), time()-time_start
            
            
#fuction to load dimacs graphs
def load_dimacs_col_file(filepath):
    """
    Reads a .col file in DIMACS format and returns a NetworkX graph.
    """
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('c'):
                continue  # comment
            elif line.startswith('p'):
                parts = line.strip().split()
                num_nodes = int(parts[2])
                # optionally add nodes explicitly
                G.add_nodes_from(range(num_nodes))
            elif line.startswith('e'):
                #print('count', count)
                parts = line.strip().split()
                u = int(parts[1])-1
                v = int(parts[2])-1
                G.add_edge(u, v)
    return G


file_path= f"dataset/{args.dataset}.txt"
print(file_path)
test_graphs,_=read_data(file_path)
#test_graphs=test_graphs[:5]
num_graphs=len(test_graphs)
print('num_graphs:', num_graphs)
result_file_name=f'results_fastcolor_{args.dataset}.txt'
result_file= open(result_file_name, 'w')
graph_count=0

colors=[]
timee=[]
for nx_graph in test_graphs:
    graph_count+=1
    nodes=nx_graph.number_of_nodes()
    edges=nx_graph.number_of_edges()
    print(f'\nProcessing******* {graph_count}/{num_graphs} ********nodes:{nodes}********edges:{edges}*******************************************************')
    sol, time_exec=FastColor(nx_graph, cutoff_time=cutoff_time)
    colors.append(sol)
    timee.append(time_exec)
    result_file.write(f'{sol} {time_exec}\n')
    result_file.flush()

print(f'\nFastColor performance: {statistics.mean(colors)} {statistics.mean(timee)}')
