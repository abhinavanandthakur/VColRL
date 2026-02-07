import numpy as np
import networkx as nx
from copy import deepcopy
from time import time
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
import dgl
from mis.ppo.actor_critic import ActorCritic as ActorCriticMis
from mis.ppo.graph_net import PolicyGraphConvNet as PolicyGraphConvNetMis
from mis.ppo.graph_net import ValueGraphConvNet as ValueGraphConvNetMis
from mis.env import MaximumIndependentSetEnv

from load_dataset import data_set,read_data, split_list
from pulp import *
import pulp
from random import randint
from itertools import combinations, chain
from Coloring_networkx_addons import ThinGraph, is_coloring_feasible
from timeit import default_timer
import os
from load_dataset import read_data
import argparse

device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='er_50_100')

args=parser.parse_args()


# env
hamming_reward_coef = 0.1

# actor critic
num_layers = 4
hidden_dim = 128

#optimiazation
max_epi_t = 64 #hp
episode_length= 64 #hp
num_parallel_graph=100 #hp
gurobi_limit=10
# dataset specific
min_num_nodes = 100 #def 20 40 
max_num_nodes = 150 #def 50 300

eval_num_samples = 1


sat_pulp=0
sat_mis=0
sat_greedy=0
sat_tabucol=0

avg_soln_pulp=0
avg_time_pulp=0
avg_soln_greedy=0
avg_time_greedy=0
avg_soln_mis=0
avg_time_mis=0
avg_soln_tabucol=0
avg_time_tabucol=0

num_channels=15

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
state_dict_mis = torch.load(model_path_mis, weights_only=True)

# Load the state dictionary into the model
actor_critic_mis.load_state_dict(state_dict_mis)


# define evaluate function for mis
def evaluate_mis(g, actor_critic):
    actor_critic.eval()
    cum_cnt = 0
    cum_eval_sol = 0.0
    
    g.set_n_initializer(dgl.init.zero_initializer)
    ob = env_mis.register(g, num_samples = eval_num_samples)
    while True:
        with torch.no_grad():
            action = actor_critic.act(ob, g)

        ob, reward, done, info = env_mis.step(action)
        if torch.all(done).item():
            cum_eval_sol += info['sol'].max(dim = 1)[0].sum().cpu()
            cum_cnt += g.batch_size
            break
    
    ob_temp=ob.select(2,0)
    #avg_eval_sol = cum_eval_sol / cum_cnt

    return cum_eval_sol, ob_temp.flatten()# ob_temp[:,torch.argmax(torch.sum(ob_temp,dim=0))]


def channels_used_mis(graph,actor_critic):
    count=0
    subgraph=graph.subgraph(graph.ndata['value']!=0)
    if subgraph.num_nodes()==0:
        return 0
    while subgraph.num_nodes()!=0:
        _,ob=evaluate_mis(subgraph,actor_critic_mis)
        if torch.any(ob==1):
            count+=torch.max(subgraph.ndata['value'][ob==1]).item()
            subgraph.ndata['value'][ob==1]=0
        else:
            count+=torch.sum(subgraph.ndata['value'][ob==0]).item()
            subgraph.ndata['value'][ob==0]=0
            break
        subgraph=subgraph.subgraph(subgraph.ndata['value']!=0)

    return count




def check_solution(G, demand,best_time_cb=5000):
    num_nodes=G.number_of_nodes()
    #pulp
    # Create a PuLP problem instance
    problem = LpProblem("CB_Optimization", LpMinimize)
    # Defining variables
    btot=15
    b=LpVariable.matrix("b",(range(2),range(num_nodes)),1,btot,LpInteger)
    w=LpVariable("w",1,btot,LpInteger)
    y=LpVariable.matrix("y",(range(num_nodes),range(num_nodes)),cat="Binary")

    #adding objective
    problem+=w

    #adding constraints
    for node in G.nodes():
        #problem+= (b[1][node]-b[0][node]+1) >= demand[node] #consr 1
        problem+= b[1][node]>=b[0][node]
        problem+= w>=b[1][node] #constr 4
        
        for neighbor in G.neighbors(node):
            #ypk init
            problem+= b[1][node]>=b[0][neighbor]+0.0001-3000*(1-y[node][neighbor])
            problem+=b[1][node]<=b[0][neighbor]+3000*(y[node][neighbor])
            problem+=(b[1][node]-b[0][neighbor])<= btot*y[node][neighbor]-1 #constr 2
            problem+=y[node][neighbor]+y[neighbor][node]<=1 #constr 3
            
    #problem.solve(PULP_CBC_CMD(msg=0,timeLimit=15))
    #problem.solve(GUROBI_CMD(msg=0,timeLimit=best_time_cb,threads=1))
    problem.solve(GUROBI_CMD(msg=0,timeLimit=gurobi_limit))
    try:
        return LpStatus[problem.status], int(w.varValue)
    except:
        return LpStatus[problem.status], 0

#tabucol
def Tabucol_opt(G, k, C_L=6, C_lambda=0.6, C_maxiter=100000, verbose=False):
    '''Tabucol_opt provides the graph coloring with the smallest number of
    colors'''
    assert len(G) > 0
    # print(k,len(G))
    # assert k > 0 and k <= len(G)
    best_colors, best_niter = {}, 0
    # compute length of max clique as number of colors cannot be less
    length_max_clique = 1
    ncolors = k
    while ncolors >= length_max_clique:
        colors, niter = Tabucol(G, ncolors, C_L, C_lambda, C_maxiter)
        if is_coloring_feasible(G, colors):
            best_colors = dict(colors)
            best_niter = niter
            ncolors = max(colors.values())
            if verbose:
                print('Tabucol_opt found a solution with %d colors' % (max(colors.values())+1))
        else:
            if not best_colors:
                print('Tabucol_opt did not find a solution with %d colors' % k)
            break

    return best_colors, best_niter


def Tabucol(G, k, C_L=6, C_lambda=0.6, C_maxiter=100000, verbose=False):
    '''Tabucol provides a vertex k-coloring if such coloring is found
    using a tabu search scheme. Tabucol features are inspired from
    "A survey of local search methods for graph coloring" by Philippe Galiniera
    and Alain Hertzb" '''

    def is_tabu_allowed(tabu_d, nc, n_iter):
        ''' assess whether candidate is in tabu dictionary'''
        return (nc not in tabu_d) or (nc in tabu_d and tabu_d[nc] < n_iter)

    # generate random coloring and compute color classes
    colors = {i: randint(0, k-1) for i in G.nodes()}
    color_classes = {col: set(i for i in G.nodes()
                     if colors[i] == col) for col in range(k)}
    # compute actual violations and number of violated edges as F
    viol_edges = {col: [edge for edge in combinations(col_set, 2)
                        if edge in G.edges()]
                  for col, col_set in color_classes.items()}
    #  viol_nodes contains nodes involved in violated edges, a node
    #  being counted as many times it appears in viol_edges
    viol_nodes = {col: list(chain.from_iterable(viol_edges[col]))
                  for col in color_classes.keys()}
    # F is the total number of violations (violated edges)
    F = sum(len(v) for v in viol_edges.values())

    # initiate local search in 1-move neighborhood
    # create tabu dictionary with (node, col) as key and niter as value
    tabu = {}
    niter = 0
    restrictive = False
    while F > 0 and niter < C_maxiter:

        # generate candidates with violation variation
        delta = {}
        for col, node_list in viol_nodes.items():
            for node in node_list:
                old_count = viol_nodes[col].count(node)
                for col_cand in range(k):
                    nc = (node, col_cand)
                    if col_cand != col and is_tabu_allowed(tabu, nc, niter):
                        new_count = len(set(G[node]).intersection(color_classes[col_cand]))
                        delta[nc] = new_count-old_count
        if not delta:
            # skip current iteration
            if not restrictive:
                if verbose:
                    print('tabu scheme is probably too restrictive')
                restrictive = True
        else:
            # select a candidate among the ones with lowest delta
            delta_c = min(delta.values())
            final_cand = [(n, c) for (n, c), value in delta.items()
                          if value == delta_c]
            # choose a candidate with lowest value at random
            (node_c, col_c) = final_cand[randint(0, len(final_cand)-1)]
            # update tabu dictionary
            deleting_tabu_list = [key_t for key_t, iter_t in tabu.items()
                                  if iter_t <= niter]
            for key_t in deleting_tabu_list:
                del tabu[key_t]
            old_col = colors[node_c]
            tabu[(node_c, old_col)] = niter + int(C_L + C_lambda*F)
            # update number of violations
            F += delta_c
            # update modified violation edge and node classes
            viol_edges[old_col] = [edge for edge in viol_edges[old_col]
                                   if node_c not in edge]
            viol_nodes[old_col] = list(chain.from_iterable(viol_edges[old_col]))
            new_viol_edges = ((node_c, u) for u in set(G[node_c]).intersection(color_classes[col_c]))
            viol_edges[col_c].extend(new_viol_edges)
            viol_nodes[col_c] = list(chain.from_iterable(viol_edges[col_c]))
            # update colors and color classes
            colors[node_c] = col_c
            color_classes[old_col] -= {node_c}
            color_classes[col_c] = color_classes[col_c].union({node_c})

        niter += 1

    if verbose and niter == C_maxiter:
        print('exiting loop as max iterations exceeded')

    # get rid of empty color classes to speed up next search
    if not all(col_class for col_class in color_classes.values()):
        colors = {}
        count = 0
        for col_class in color_classes.values():
            if col_class:
                for n in col_class:
                    colors[n] = count
                count += 1

    return colors, niter


def tabu_main(G, initial):
    sols=[]
    time=0
    colors = {}
    start_time = default_timer()
    colors, niter = Tabucol_opt(G,
                                initial,
                                C_L=6,
                                C_lambda=0.6,
                                C_maxiter=100000,
                                verbose=False)
    elapsed = default_timer() - start_time
    try:
        return max(colors.values())+1, elapsed
    except:
        return -1 , -1


def greedy_coloring(G):
    V = len(G.nodes)
    result = [-1] * V

    # Assign the first color to the first vertex
    result[0] = 0

    # A temporary array to store the available colors.
    available = [False] * V

    # Track the number of colors used
    max_color = 0

    # Assign colors to remaining V-1 vertices
    for u in range(1, V):
        # Process all adjacent vertices and
        # flag their colors as unavailable
        for i in G.neighbors(u):
            if result[i] != -1:
                available[result[i]] = True

        # Find the first available color
        cr = 0
        while cr < V:
            if not available[cr]:
                break
            cr += 1

        # Assign the found color
        result[u] = cr

        # Update the maximum color used
        if cr > max_color:
            max_color = cr

        # Reset the values back to False for the next iteration
        for i in G.neighbors(u):
            if result[i] != -1:
                available[result[i]] = False

    # Print the result
    # for u in range(V):
    #     print(f"Vertex {u} ---> Color {result[u]}")

    # Return the number of colors used
    return max_color + 1  # +1 because colors start from 0



def compare(graph,demands):
    global avg_soln_pulp
    global avg_time_pulp
    global avg_soln_greedy
    global avg_time_greedy
    global avg_soln_mis
    global avg_time_mis 
    global sat_pulp
    global sat_mis
    global sat_greedy
    global sat_tabucol
    global avg_soln_tabucol
    global avg_time_tabucol

    nx_graph=graph
    nx_demand=deepcopy(demands)
    dgl_graph=dgl.from_networkx(nx_graph)
    dgl_graph.ndata['value']=torch.tensor(nx_demand).float()
    
   #checking solution for gurobi
    a=time()
    status, channel_pulp= check_solution(nx_graph,demands)
    #status, channel_pulp= chromatic_number(nx_graph)
    b=time()-a

    #checking solution for greedy
    c=time()
    channel_greedy=greedy_coloring(nx_graph)
    d=time()-c


    #checking solution for mis
    e=time()
    soln_mis=channels_used_mis(dgl_graph,actor_critic_mis)
    time_mis=time()-e

    #tabucol
    channel_tabucol, f=tabu_main(nx_graph,15)


    #pulp
    if status=='Optimal':
        sat_pulp+=1
        avg_soln_pulp+=channel_pulp
    avg_time_pulp+=b
    best_soln_pulp=(channel_pulp,b)

    
    #greedy
    best_soln_greedy=(channel_greedy,d)
    if channel_greedy<=num_channels:
        sat_greedy+=1
        avg_soln_greedy+=channel_greedy
    avg_time_greedy+=d

    #tabucol
    best_soln_tabucol=(channel_tabucol,f)
    if channel_tabucol<=num_channels:
        sat_tabucol+=1
        avg_soln_tabucol+=channel_tabucol
    avg_time_tabucol+=f 

    #mis
    best_soln_mis=(soln_mis,time_mis)
    if soln_mis<=num_channels:
        sat_mis+=1
        avg_soln_mis+=soln_mis
    avg_time_mis+=time_mis
    #writing to file
    result_file.write(str(nx_graph.number_of_nodes())+'\n')
    result_file.write(f"{best_soln_pulp[0]} {best_soln_pulp[1]}\n")
    result_file.write(f"{best_soln_greedy[0]} {best_soln_greedy[1]:.8f}\n")
    result_file.write(f"{best_soln_tabucol[0]} {best_soln_tabucol[1]:.8f}\n")
    result_file.write(f"{best_soln_mis[0]} {best_soln_mis[1]}\n")
    result_file.write('\n')
    result_file.flush()




min_nodes= 100#def 70
max_nodes= 150
count=0
cb_performance=0

file_path= f"dataset/{args.dataset}.txt"
print(file_path)
test_graphs,_=read_data(file_path)
#test_graphs=test_graphs[:5]
num_graphs=len(test_graphs)
print('num_graphs:', num_graphs)

filepath=f'results_{args.dataset}.txt'
result_file=open(filepath,'w')

for nx_graph in test_graphs:
    nodes=nx_graph.number_of_nodes()
    edges=nx_graph.number_of_edges()
    count+=1
    print(f'Sample: {count}/{num_graphs} , Num_Nodes: {nodes}, Num_Edges:{edges}*************************************************')
    demand=list(np.ones(nodes).astype(int))
    compare(nx_graph,demand)
    

#average solution
try:    
    avg_soln_pulp/=sat_pulp
except:
    avg_soln_pulp=15

avg_soln_greedy/=sat_greedy
avg_soln_tabucol/=sat_tabucol
avg_soln_mis/=sat_mis
#average time
avg_time_pulp/=num_graphs
avg_time_greedy/=num_graphs
avg_time_tabucol/=num_graphs
avg_time_mis/=num_graphs
print('\n')
print(file_path)
print(num_graphs)
print(f"number of nodes: {min_nodes} to {max_nodes} ")
print('Total number of samples tested:',count)
print('Gurobi Performance:',sat_pulp*100/num_graphs,avg_soln_pulp,f'{avg_time_pulp:.8f}')
print("FF Performance:",sat_greedy*100/num_graphs,avg_soln_greedy,f'{avg_time_greedy:.8f}')
print("TabucolMin Performance:",sat_tabucol*100/num_graphs,avg_soln_tabucol,f'{avg_time_tabucol:.8f}')
print("VColMIS performance:",sat_mis*100/num_graphs,avg_soln_mis,f'{avg_time_mis:.8f}')
result_file.close()
