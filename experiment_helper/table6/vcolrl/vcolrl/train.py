import os
import logging
import random
import numpy as np
import networkx as nx
import argparse
from time import time
import torch
from torch.utils.data import DataLoader
from load_dataset import read_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from load_dataset import split_list
from load_dataset import data_set
from tqdm import tqdm
import dgl
import ppo
from ppo.framework import ProxPolicyOptimFramework
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from ppo.storage import RolloutStorage
from pulp import *
from env import ChannelBonding

device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
num_channels=15
# actor critic
num_layers = 4 #default:4
input_dim = num_channels+1 #def num_channels+1
output_dim = num_channels+1
hidden_dim = 128 #def:128

# optimization
init_lr = 1e-4 #def:1e-4
max_epi_t = 32 #def 32
max_rollout_t = 32 #def 32
vali_epi_t=256
max_update_t = 1000000

# ppo
gamma = 1 #def:1
clip_value = 0.2 #def:0.2
optim_num_samples = 4 #4
critic_loss_coef = 0.25#0.5 #0.5
reg_coef = 0.01 #def:0.1
max_grad_norm =1# 0.5 #def 0.5



# dataset specific
min_num_nodes =50
max_num_nodes = 100

# main
rollout_batch_size = 32
eval_batch_size = 1000
optim_batch_size = 16
init_anneal_ratio = 1.0
max_anneal_t = - 1
anneal_base = 0.
train_num_samples = 1 #2
eval_num_samples = 1

# generate and save datasets
#num_eval_graphs = 1000

hamming_reward_coef = 0.1

#importing data
filepath='./dataset/graph_dataset.txt'
graphs,optimum_soln=read_data(filepath,device)

print('train')
for i in set(optimum_soln):
    print(i,optimum_soln.count(i))
#split dataset into train, validation and test

train_graphs,test_graphs,val_graphs=split_list(graphs)
print(len(graphs))
optimum_soln=optimum_soln[-eval_batch_size:]
print('vali')
for i in set(optimum_soln):
    print(i,optimum_soln.count(i))
# number of batches per epochs
num_batch_per_epoch = len(train_graphs)//rollout_batch_size
print(num_batch_per_epoch)

#files and variables for saving models
model_dir='./models'
os.makedirs(model_dir,exist_ok=True)
index=0 #helper variable to name the model for saving

#utility files for logging validation stats and losses
vali_file=open('validation_stats.txt','w')
loss_file=open('loss.txt','w')
datasets={
    'train': data_set(train_graphs),
    'vali': data_set(val_graphs),
    'test': data_set(test_graphs)
}
print(len(datasets['train']),len(datasets['vali']))

def collate_fn(graphs):
    return dgl.batch(graphs)

data_loaders={
    'train': DataLoader(
        datasets['train'],
        batch_size=rollout_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True
    ),

    "vali": DataLoader(
        datasets["vali"],
        batch_size = eval_batch_size,
        shuffle = False,
        collate_fn=collate_fn,
        num_workers = 0,
        #drop_last=True
        )
}

#environment construction
env=ChannelBonding(max_epi_t=max_epi_t,hamming_reward_coef=hamming_reward_coef,device=device)

#constructing rollout storage
rollout=RolloutStorage(
    max_t=max_rollout_t,
    batch_size=rollout_batch_size,
    num_samples=train_num_samples
    )

#construct actor critic network and upload the best model
actor_critic =ActorCritic(
    actor_class= PolicyGraphConvNet,
    critic_class=ValueGraphConvNet,
    max_num_nodes=max_num_nodes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_channels=num_channels,
    device=device
)

##code to start the training from the state of the specified model in the path

#model_path_cb = './model_233.pth'
#state_dict_cb = torch.load(model_path_cb)
# Load the state dictionary into the model
#actor_critic.load_state_dict(state_dict_cb)

# construct PPO framework
framework = ProxPolicyOptimFramework(
    actor_critic = actor_critic,
    init_lr = init_lr,
    clip_value = clip_value, 
    optim_num_samples = optim_num_samples,
    optim_batch_size = optim_batch_size,
    critic_loss_coef = critic_loss_coef, 
    reg_coef = reg_coef, 
    max_grad_norm = max_grad_norm,
    num_channels=num_channels,
    device = device
    )    


def evaluate(mode, actor_critic):
    actor_critic.eval()
    sat=0
    counter=0
    total_rew=torch.zeros((eval_batch_size,1),device=device)
    progress_bar = tqdm(total=vali_epi_t, desc="Validation Progress", unit="iteration")
    for g in data_loaders[mode]:
        g.set_n_initializer(dgl.init.zero_initializer)
        ob= env.register(vali_epi_t,g,num_samples=eval_num_samples)
        ob=ob.to(device)
        while True:
            with torch.no_grad():
                action = actor_critic.act(ob, g).to(device)
            ob,reward,done = env.step(action+1)
            total_rew+=reward
            #print(reward.size())
            if torch.all(done).item():
                state=ob.select(2,0).int().squeeze().tolist()
                optimum_iter=0
                for graph in dgl.unbatch(g):
                    num_nodes=graph.number_of_nodes()
                    demand=graph.ndata['value'].tolist()
                    local_state=state[:num_nodes]
                    state[:num_nodes]=[]
                    channel=[1 for _ in range(num_channels+1) ]
                    optimum=optimum_soln[optimum_iter]
                    optimum_iter+=1

                    for i in range(len(local_state)):
                        id1=local_state[i]
                        if (id1):
                            id2=id1+demand[i]
                            if id2>num_channels+1:
                                channel[id1:]=[0 for _ in range(num_channels+1-id1)]
                            else:
                                channel[id1:int(id2)]=[0 for _ in range(int(id2)-id1)]
                    channel[0]=0
                    satisfied=round(100-local_state.count(0)*100/num_nodes,2)
                    if(int(satisfied)==100):
                        sat+=1
                    my_soln=num_channels-channel.count(1)
                    vali_file.write(f"{satisfied} {my_soln} {optimum}\n")
                    #print(channel.count(1), 100-local_state.count(0)*100/num_nodes)
                #vali_file.write("\n")
                #vali_file.write(f"{total_rew.mean().item()}\n")
                break
            counter+=1
            progress_bar.update(1)
    actor_critic.train()
    return sat,total_rew.mean().item()

#variables to log losses after every epoch
actor_loss_avg=0
critic_loss_avg=0
entropy_loss_avg=0
objective_avg=0

for update_t in range(max_update_t):

    if update_t == 0 or torch.all(done).item():
        try:
            g = next(train_data_iter)
        except:
            train_data_iter = iter(data_loaders["train"])
            g = next(train_data_iter)
        
        g.set_n_initializer(dgl.init.zero_initializer)
        ob = env.register(max_epi_t,g,num_samples=train_num_samples)
        rollout.insert_ob_and_g(ob, g)

    for step_t in range(max_rollout_t):
        # get action and value prediction
        with torch.no_grad():
            (action, 
            action_log_prob, 
            value_pred, 
            ) = actor_critic.act_and_crit(ob, g)

        # print(g.ndata['ch'])
        # a=input('ch')
        # step environments
        ob,  reward, done = env.step(action+1)

        # insert to rollout
        rollout.insert_tensors(
            ob,
            action,
            action_log_prob, 
            value_pred, 
            reward, 
            done
            )
        
        if torch.all(done).item():
            break
    
    # compute gamma-decayed returns and corresponding advantages
    rollout.compute_rets_and_advantages(gamma)

    # update actor critic model with ppo
    actor_loss, critic_loss, entropy_loss,objective = framework.update(rollout)
    actor_loss_avg+=actor_loss.item()
    critic_loss_avg+=critic_loss.item()
    entropy_loss_avg+=entropy_loss.item()
    objective_avg+=objective.item()
    loss_file.write(f"{actor_loss.item()} {critic_loss.item()} {entropy_loss.item()} {objective.item()}\n")
    

            
# log stats
    #a=input()
    print("update_t: {:05d}".format(update_t + 1))
    print("train stats...")
    print(
        "Satisfied%: {:.4f}, "
        "Objective%: {:.4f}, "
        "actor_loss: {:.4f}, "
        "critic_loss: {:.4f}, "
        "entropy: {:.4f}".format(
            (ob[:,0,0]!=0).sum().item()*100/ob.size(0),
            objective.item(),
            actor_loss.item(),
            critic_loss.item(),
            entropy_loss.item()
            )
        )
        

    if (update_t + 1) % num_batch_per_epoch == 0:
        actor_loss_avg/=num_batch_per_epoch
        critic_loss_avg/=num_batch_per_epoch
        entropy_loss_avg/=num_batch_per_epoch
        objective_avg/=num_batch_per_epoch
        loss_file.write(f"{actor_loss_avg} {critic_loss_avg} {entropy_loss_avg} {objective_avg}\n\n")
        actor_loss_avg=0
        critic_loss_avg=0
        entropy_loss_avg=0
        objective_avg=0
        index+=1
        #saving models after each epoch
        torch.save(actor_critic.state_dict(), os.path.join(model_dir,'model_'+str(index)+'.pth'))
        #torch.save(actor_critic.state_dict(), os.path.join(model_dir,'model_'+str(index)+'.pth'))
        sol,total_rew = evaluate("vali", actor_critic)
        sol=sol*100/eval_batch_size
        #vali_file.write(f"{sol:.2f}\n")
        print("vali stats...")
        print("Validation satisfaction:",sol,"return",total_rew)
        vali_file.write(f"{sol:.2f}\n")
        vali_file.write(f"{total_rew:.2f}\n")
        vali_file.write('\n')
#        framework.cosine_anneal()


vali_file.close()
loss_file.close()
plt.close()
