import numpy as np
import networkx as nx
from copy import deepcopy
from time import time
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
import dgl
from vcolmis.actor_critic import ActorCritic as ActorCriticMis
from vcolmis.graph_net import PolicyGraphConvNet as PolicyGraphConvNetMis
from vcolmis.graph_net import ValueGraphConvNet as ValueGraphConvNetMis
from vcolmis.env import MaximumIndependentSetEnv
from pulp import *
import pulp
from random import randint
from itertools import combinations, chain
from timeit import default_timer
import os
from argparse import ArgumentParser
import statistics
import scipy.stats as stats
import glob




device = 'cpu'

# env
hamming_reward_coef = 0.1

# actor critic
num_layers = 4
hidden_dim = 128

#optimiazation
max_epi_t = 128 #hp
episode_length= 128 #hp

# dataset specific
min_num_nodes = 50 
max_num_nodes = 100 

num_colors=15

# construct everything for mis
env_mis = MaximumIndependentSetEnv(
    max_epi_t = max_epi_t,
    max_num_nodes = max_num_nodes,
    hamming_reward_coef = hamming_reward_coef,
    device = device
    )

# construct actor critic network
actor_critic_mis = ActorCriticMis(
    actor_class = PolicyGraphConvNetMis,
    critic_class = ValueGraphConvNetMis, 
    max_num_nodes = max_num_nodes, 
    hidden_dim = hidden_dim,
    num_layers = num_layers,
    device = device
    )

# Load the saved state dictionary
model_path_mis = './model_mis.pth'
state_dict_mis = torch.load(model_path_mis)

# Load the state dictionary into the model
actor_critic_mis.load_state_dict(state_dict_mis)



filepath= f'./results_vcolmis.txt'
result_file=open(filepath,'w')


#function to compute mean, std

def compute_stats(data, confidence=0.95):
    """Compute mean, standard deviation, and 95% confidence intervals with MoE."""
    
    n = len(data)
    if n < 2:
        raise ValueError("At least two data points are required.")

    mean = statistics.mean(data)
    std_dev = statistics.stdev(data)
    #z_ci = (mean, z_margin, (mean - z_margin, mean + z_margin))

    return mean, std_dev

# define evaluate function for mis
def evaluate_mis(g, actor_critic):
    actor_critic.eval()
    cum_cnt = 0
    cum_eval_sol = 0.0
    
    g.set_n_initializer(dgl.init.zero_initializer)
    ob = env_mis.register(g, num_samples = 1)
    while True:
        with torch.no_grad():
            action = actor_critic.act(ob, g)

        ob, reward, done, info = env_mis.step(action)
        if torch.all(done).item():
            cum_eval_sol += info['sol'].max(dim = 1)[0].sum().cpu()
            cum_cnt += g.batch_size
            break
    
    ob_temp=ob.select(2,0)
    return cum_eval_sol, ob_temp.flatten()



def colors_used_mis(graph):
    count=0
    while graph.number_of_nodes()!=0:
        _,ob=evaluate_mis(graph,actor_critic_mis)
        if torch.any(ob==1):
            count+=1
    
        else:
            count+=torch.sum(ob==0).item()
            break

        graph=graph.subgraph(ob==0)

    return count



def load_dimacs_col_file(file_path):
    G = nx.Graph()
    max_node=0
    with open(file_path, 'r') as f:
        for line in f:
            line=line.strip()
            if line.startswith('e'):  # 'e' lines represent edges
                _, node1, node2 = line.split()
                # Subtract 1 from node labels to make them zero-indexed
                G.add_edge(int(node1) - 1, int(node2) - 1)
                max_node=max(int(node1)-1, int(node2)-1, max_node)
    for node in range(max_node+1):
        if node not in G:
            G.add_node(node)
    return G



def compare(graph,file_name):
    
    nx_graph=graph
    dgl_graph_main=dgl.from_networkx(nx_graph)
                  
    color=[]
    timee=[]
    
    for j in range(100):
        e=time()
        soln_mis=colors_used_mis(dgl_graph_main)
        time_mis=time()-e
        color.append(soln_mis)
        timee.append(time_mis)
        print(f'vcolmis agent done with {j+1} trial with {soln_mis} colors')
        
    
    mean, std_dev=compute_stats(color)
    mean_t, std_dev_t=compute_stats(timee)
    result_file.write(f'{file_name}: {min(color)} {timee[color.index(min(color))]} {mean} {std_dev} {mean_t} {std_dev_t}\n')
    result_file.flush()
    print('vcolmis completed')
            




directory_path='benchmarks'
benchmarks=glob.glob('benchmarks/*.col')
num_graphs = len(benchmarks)
graph_count=0

for file_name in benchmarks:
    graph_count+=1
    print(f'\nProcessing********************{graph_count}/{num_graphs} ****************{file_name}*********************************************************************************************')
    #file_path=os.path.join(directory_path,file_name)
    nx_graph = load_dimacs_col_file(file_name)
    nodes=nx_graph.number_of_nodes()
    edges=nx_graph.number_of_edges()
    print(nodes,edges)
    a=compare(nx_graph,file_name.split('/')[-1])
 


result_file.close()
