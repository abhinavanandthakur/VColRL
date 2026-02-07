import numpy as np
import networkx as nx
from copy import deepcopy
from time import time
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
import dgl
from vcolrl.ppo.actor_critic import ActorCritic as ActorCriticCb
from vcolrl.ppo.graph_net import PolicyGraphConvNet as PolicyGraphConvNetCb
from vcolrl.ppo.graph_net import ValueGraphConvNet as ValueGraphConvNetCb
from vcolrl.env import VCP
from random import randint
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
episode_length= 1024 #hp
num_parallel_graph=1
sample_mis=100#hp

# dataset specific
min_num_nodes = 50 
max_num_nodes = 100 

num_colors=15



#constructing everything for cb

env_cb=VCP(max_epi_t=episode_length,device=device,num_colors=num_colors)

#construct actor critic network
actor_critic_cb =ActorCriticCb(
    actor_class= PolicyGraphConvNetCb,
    critic_class=ValueGraphConvNetCb,
    max_num_nodes=max_num_nodes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_channels=num_colors,
    device=device
)

model_path_cb = './model_sw.pth'
state_dict_cb = torch.load(model_path_cb,map_location=torch.device('cpu'))

# Load the state dictionary into the model
actor_critic_cb.load_state_dict(state_dict_cb)

filepath= 'results_sw.txt'
result_file=open(filepath,'w')


def compute_stats(data, confidence=0.95):
    """Compute mean, standard deviation, and 95% confidence intervals with MoE."""
    
    n = len(data)
    if n < 2:
        raise ValueError("At least two data points are required.")

    mean = statistics.mean(data)
    std_dev = statistics.stdev(data)
    std_err = std_dev / (n ** 0.5)  # Standard error

    # t-distributed CI (for small samples)
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    t_margin = t_critical * std_err
    #t_ci = (mean, t_margin, (mean - t_margin, mean + t_margin))

    # Normal-distributed CI (for large samples)
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    z_margin = z_critical * std_err
    #z_ci = (mean, z_margin, (mean - z_margin, mean + z_margin))

    return mean, std_dev, std_err, t_margin, z_margin



def evaluate_cb(graph, actor_critic):
    num_vertex= graph.number_of_nodes()
    actor_critic.eval()
    progress_bar = tqdm(total=num_vertex*5, desc="Validation Progress", unit="iteration")
    t1=time()
    graph_list = [graph.clone() for _ in range(num_parallel_graph)]
    g=dgl.batch(graph_list)
    g.set_n_initializer(dgl.init.zero_initializer)
    ob = env_cb.register(num_vertex*5,g,1)
    ob=ob.to(device)
    while True:
        with torch.no_grad():
            action = actor_critic.act(ob, g).to(device)
        ob,_,done = env_cb.step(action+1)
        progress_bar.update(1)
        if torch.all(done).item():
            break
    state=ob.select(2,0).int().squeeze().tolist()
    sat_best=0
    solution_best=10000
    ob_best=[]
    for graph in dgl.unbatch(g):
        num_nodes=graph.number_of_nodes()
        local_state=state[:num_nodes]
        state[:num_nodes]=[]
        satisfied=round(100-local_state.count(0)*100/num_nodes,2)
        my_soln=len(set(x for x in local_state if x != 0))

        if satisfied>sat_best:
            sat_best=satisfied
            solution_best=my_soln
            ob_best=deepcopy(local_state)
        elif satisfied==sat_best and my_soln<solution_best:
            solution_best=my_soln
            ob_best=deepcopy(local_state)
    solution_time=time()-t1 
    return sat_best,torch.tensor(ob_best),solution_best,solution_time

    



def load_dimacs_col_file(file_path):
    G = nx.Graph()
    max_node=0
    with open(file_path, 'r') as f:
        for line in f:
            line=line.strip()
            # split_list=line.split()
            # if len (split_list)==2:
            #     G.add_edge(int(split_list[0]) - 1, int(split_list[1]) - 1)
            #     max_node=max(int(split_list[0])-1, int(split_list[1])-1, max_node)
            if line.startswith('e'):  # 'e' lines represent edges
                _, node1, node2 = line.split()
                # Subtract 1 from node labels to make them zero-indexed
                G.add_edge(int(node1) - 1, int(node2) - 1)
                max_node=max(int(node1)-1, int(node2)-1, max_node)
    for node in range(max_node+1):
        if node not in G:
            G.add_node(node)
    return G

num_samples=30  #the number of trials for mean and stdev is set to 30

def compare(graph,file_name):
    
    nx_graph=graph
    dgl_graph_main=dgl.from_networkx(nx_graph)
    
    print('RL agent.......')
    color=[]
    timee=[]
    for i in range(num_samples):
        dgl_graph=deepcopy(dgl_graph_main)
        agent_time_begin=time()
        soln_cb_sat,ob_best,soln_cb_colors,time_cb=evaluate_cb(dgl_graph,actor_critic_cb)
        Flag=1
        while soln_cb_sat<100:
            Flag+=1
            if Flag>10:
                break
            dgl_graph=dgl_graph.subgraph(ob_best==0)
            if dgl_graph.num_nodes()==1:
                soln_cb_colors+=1
                break
            soln_cb_sat,ob_best,soln_cb_colors_ad,time_cb_ad=evaluate_cb(dgl_graph,actor_critic_cb)
            soln_cb_colors+=soln_cb_colors_ad
            time_cb+=time_cb_ad
        agent_time=time()-agent_time_begin
        color.append(soln_cb_colors)
        timee.append(agent_time)
        print(f'RL agent done with {i+1} trial with time {agent_time}')
    best_color=min(color)
    index= color.index(best_color)
    best_time=timee[index]
    mean_color, std_dev_color, _, _, _=compute_stats(color)
    mean_time, std_dev_time, _, _, _=compute_stats(timee)
    result_file.write(f'{file_name}: {best_color} {best_time} {mean_color} {std_dev_color} {mean_time} {std_dev_time}\n')
    result_file.flush()
    print('RL agent done')
    return 0
            
        
benchmarks=glob.glob('benchmarks/*.col')
    
num_graphs = len(benchmarks)
graph_count=0
for file_name in benchmarks:
    graph_count+=1
    print(f'\nProcessing********************{graph_count}/{num_graphs} ****************{file_name}*********************************************************************************************')

    nx_graph = load_dimacs_col_file(file_name)
    nodes=nx_graph.number_of_nodes()
    edges=nx_graph.number_of_edges()
    print(nodes,edges)
    a=compare(nx_graph,file_name.split('/')[-1])
    try:
        pass
    except:
        print("Some error occured")
        continue
 


result_file.close()
