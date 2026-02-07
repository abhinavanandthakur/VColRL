import numpy as np
import networkx as nx
from copy import deepcopy
from time import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dgl
from vcolrl.actor_critic import ActorCritic as ActorCriticCb
from vcolrl.graph_net import PolicyGraphConvNet as PolicyGraphConvNetCb
from vcolrl.graph_net import ValueGraphConvNet as ValueGraphConvNetCb
from vcolrl.env import VCP
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


#constructing everything for cb

env_cb=VCP(max_epi_t=episode_length,device=device,num_colors=num_colors)

#construct actor critic network
actor_critic_cb =ActorCriticCb(
    actor_class= PolicyGraphConvNetCb,
    critic_class=ValueGraphConvNetCb,
    max_num_nodes=max_num_nodes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_colors=num_colors,
    device=device
)

model_path_cb = './model_vcolrl.pth'
state_dict_cb = torch.load(model_path_cb,map_location=torch.device('cpu'))

# Load the state dictionary into the model
actor_critic_cb.load_state_dict(state_dict_cb)

#result file to store results
filepath= f'./results_vcolrl.txt'
result_file=open(filepath,'w')

#functionto compute meana and std dev
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

    return mean, std_dev



#this function takes a graph and an actor critic and returns the solution found by the actor critic on the graph

def evaluate_vcolrl(graph, actor_critic):
    
    actor_critic.eval()
    progress_bar = tqdm(total=episode_length, desc="Validation Progress", unit="iteration")
    t1=time()
    g=graph
    num_nodes=g.number_of_nodes()
    g.set_n_initializer(dgl.init.zero_initializer)
    ob = env_cb.register(episode_length,g,1)
    ob=ob.to(device)
    while True:
        with torch.no_grad():
            action = actor_critic.act(ob, g).to(device)
        ob,_,done = env_cb.step(action+1)
        progress_bar.update(1)
        if torch.all(done).item():
            break
    state=ob.select(2,0).int().squeeze().tolist()
    satisfied=round(100-state.count(0)*100/num_nodes,1)
    my_soln=len(set(x for x in state if x != 0))
    solution_time=time()-t1 
    return satisfied,torch.tensor(state),my_soln,solution_time

    
#function to load a dimacs col file and return a networkx graph

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


#this function runs vcolrl on a given graph for 100 trials and writes the results to a file
def run_vcolrl(graph,file_name):
    
    nx_graph=graph
    dgl_graph_main=dgl.from_networkx(nx_graph)
    
    color=[]
    timee=[]
    for i in range(100):
        dgl_graph=deepcopy(dgl_graph_main)
        agent_time_begin=time()
        soln_cb_sat,ob_best,soln_cb_colors,time_cb=evaluate_vcolrl(dgl_graph,actor_critic_cb)
        Flag=1
        while soln_cb_sat<100:
            Flag+=1
            if Flag>10:
                break
            dgl_graph=dgl_graph.subgraph(ob_best==0)
            if dgl_graph.num_nodes()==1:
                soln_cb_colors+=1
                break
            soln_cb_sat,ob_best,soln_cb_colors_ad,time_cb_ad=evaluate_vcolrl(dgl_graph,actor_critic_cb)
            soln_cb_colors+=soln_cb_colors_ad
            time_cb+=time_cb_ad
        agent_time=time()-agent_time_begin
        color.append(soln_cb_colors)
        timee.append(agent_time)
        print(f'VcolRl agent done with {i+1} trial with {soln_cb_colors} colors for {file_name}')
        
    
    mean_col, std_dev_col=compute_stats(color)
    mean_time, std_dev_time=compute_stats(timee)
    #writing best solution and mean and std dev for colors and time
    result_file.write(f'{file_name}: {min(color)} {timee[color.index(min(color))]} {mean_col} {std_dev_col} {mean_time} {std_dev_time}\n')
    #writing mean and std dev for colors
    result_file.flush()
    print('done')
            

            
#evaluating vcolrl on test data
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
    a=run_vcolrl(nx_graph,file_name.split('/')[-1])
 


result_file.close()
