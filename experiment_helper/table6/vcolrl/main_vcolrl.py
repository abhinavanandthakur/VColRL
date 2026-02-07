import numpy as np
from copy import deepcopy
from time import time
from tqdm import tqdm
import torch
import dgl
from vcolrl.ppo.actor_critic import ActorCritic as ActorCriticCb
from vcolrl.ppo.graph_net import PolicyGraphConvNet as PolicyGraphConvNetCb
from vcolrl.ppo.graph_net import ValueGraphConvNet as ValueGraphConvNetCb
from vcolrl.env import ChannelBonding

from load_dataset import read_data
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str, default="er_50_100")
args=parser.parse_args()

device = 'cpu'

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
min_num_nodes = 50 #def 20 40 
max_num_nodes = 100 #def 50 300

eval_num_samples = 1

sat_cb=0
avg_soln_cb=0
num_channels=15
avg_time_cb=0

#constructing everything for cb

env_cb=ChannelBonding(max_epi_t=episode_length,device=device,num_channels=num_channels)
#construct actor critic network
actor_critic_cb =ActorCriticCb(
    actor_class= PolicyGraphConvNetCb,
    critic_class=ValueGraphConvNetCb,
    max_num_nodes=max_num_nodes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_channels=num_channels,
    device=device
)

#model_path_cb = './er_50_100.pth'
model_path_cb = './model_vcolrl.pth'
state_dict_cb = torch.load(model_path_cb,weights_only=True,map_location=torch.device('cpu'))

# Load the state dictionary into the model
actor_critic_cb.load_state_dict(state_dict_cb)

# define evaluate function for mis


def evaluate_cb(graph, actor_critic):
    
    actor_critic.eval()
    progress_bar = tqdm(total=episode_length, desc="Validation Progress", unit="iteration")
    t1=time()
    graph_list = [graph.clone() for _ in range(num_parallel_graph)]
    g=dgl.batch(graph_list)
    g.set_n_initializer(dgl.init.zero_initializer)
    ob = env_cb.register(episode_length,g,1)
    ob=ob.to(device)
    ob_prev=ob.select(2,0).squeeze()
    break_counter=0
    while True:
        with torch.no_grad():
            action = actor_critic.act(ob, g).to(device)
        ob,_,done = env_cb.step(action+1)
        progress_bar.update(1)
        if torch.norm(ob.select(2,0).squeeze()-ob_prev,p=1)==0:
            break_counter+=1
        else:
            break_counter=0
        ob_prev= ob.select(2,0).int().squeeze()
        if torch.all(done).item() or break_counter==5:
            break
    ob[:,:,0][ob[:,:,0]==num_channels+1]=0
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
    solution_time=time()-t1 #sorted till here
    return sat_best,torch.tensor(ob_best),solution_best,solution_time





def compare(graph,demands):
    global avg_soln_cb
    global avg_time_cb
    global sat_cb 

    nx_graph=graph
    nx_demand=deepcopy(demands)
    dgl_graph=dgl.from_networkx(nx_graph)
    dgl_graph.ndata['value']=torch.tensor(nx_demand).float()
    
   
    #rl agent cb
    
    
    agent_time_begin=time()
    soln_cb_sat,ob_best,soln_cb_channels,time_cb=evaluate_cb(dgl_graph,actor_critic_cb)
    Flag=1
    
    while soln_cb_sat<100:
        Flag+=1
        if Flag>100:
            break
        #chacking correctness of solution, this time will be subtracted as it is not a part of solution
        dgl_graph=dgl_graph.subgraph(ob_best==0)
        soln_cb_sat,ob_best,soln_cb_channels_ad,time_cb_ad=evaluate_cb(dgl_graph,actor_critic_cb)
        soln_cb_channels+=soln_cb_channels_ad
        time_cb+=time_cb_ad
    agent_time=time()-agent_time_begin
    best_soln_cb=(100 if Flag<=100 else 0,soln_cb_channels,agent_time)
    if best_soln_cb[0]==100:
        sat_cb+=1
        avg_soln_cb+=soln_cb_channels
        avg_time_cb+=time_cb
    

    #greedy
    #writing to file
    result_file.write(str(nx_graph.number_of_nodes())+'\n')
    result_file.write(f"{best_soln_cb[0]} {best_soln_cb[1]} {best_soln_cb[2]}\n")
    result_file.write('\n')
    result_file.flush()


    #temporary
    return 0

count=0

file_path= f'dataset/{args.dataset}.txt'
print(file_path)
test_graphs,_=read_data(file_path)
#test_graphs=test_graphs[:10]
num_graphs=len(test_graphs)
print('num_graphs:', num_graphs)

result_file=open(f"result_{args.dataset}.txt",'w')

for nx_graph in test_graphs:
    nodes=nx_graph.number_of_nodes()
    count+=1
    print(f'Sample: {count}/{num_graphs} , Num_Nodes: {nodes}  *********************************************************')
    demand=list(np.ones(nodes).astype(int))
    compare(nx_graph,demand)

print(sat_cb)
#average solution
avg_soln_cb/=sat_cb
avg_time_cb/=sat_cb

print('\n')
print(file_path)
print(num_graphs)
print('Total number of samples tested:',count)
print('VColRL Model Performance:',sat_cb*100/num_graphs,avg_soln_cb,f'{avg_time_cb:.8f}')
result_file.close()
