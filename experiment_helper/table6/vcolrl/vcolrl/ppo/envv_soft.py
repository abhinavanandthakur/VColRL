import torch
import dgl
import networkx as nx
from collections import namedtuple
import dgl.function as fn
from copy import deepcopy as dc
import random
import time
from time import time
from torch.utils.data import DataLoader
import numpy as np
#from numba import njit

class ChannelBonding(object):
    
    def __init__(self,max_epi_t=32,hamming_reward_coef=0.1,device='cpu',num_channels=15):
        
        self.max_epi_t= max_epi_t
        self.device=device
        self.hamming_reward_coef = hamming_reward_coef
        self.num_channels=num_channels

    
    # def demand_value(self,g):
    #     norm=dgl.sum_nodes(g,'dem')
    #     obs=g.ndata['x_'].clone()
    #     mask=(obs!=0) & (obs!=self.num_channels+1)
    #     g.ndata['mask']=mask
    #     val=dgl.sum_nodes(g,'dem','mask')
    #     g.ndata.pop('mask')
    #     return val/norm

    # def channel_value(self,g):
        
    #     obs=g.ndata['x_'].clone()
    #     obs[obs==self.num_channels+1]=0
    #     mask=obs!=0
    #     max_channel=torch.where(mask,obs+self.demand-1,obs)
    #     g.ndata['max_channel'] =max_channel
    #     val=dgl.max_nodes(g,'max_channel')
    #     g.ndata.pop('max_channel')
    #     return val/self.num_channels

    def step(self,action):
        _,done=self._take_action(action)
        ob=self._build_ob()
        return ob,0,done
    

    def _channel_occupancy(self,inc,ob,dem):
    
        ch=np.zeros((self.num_nodes,self.num_samples,self.num_channels+1))
        for i in range(self.num_nodes):
            for j in range(self.num_samples):
                if inc[i,j]:
                    ch[i,j,ob[i,j]:ob[i,j]+dem[i,j]]=1
        return ch
    
    
    def _take_action (self,action):

        self.graph.ndata['x_']= self.graph.ndata['x'].clone()

        # current_sol_ch=self.channel_value(self.graph)
        # current_sol_dem=self.demand_value(self.graph)

        #taking action
        undecided=(self.graph.ndata['x_']==self.num_channels+1)
        self.graph.ndata['x_'][undecided] = (action)[undecided].float()
        self.t+=1
       
        #cleanup begins

        #removing overflows
        obs=self.graph.ndata['x_'].clone()
        max_channel=torch.where((obs!=0) & (obs!=self.num_channels+1),obs+self.demand-1,obs)
        self.graph.ndata['x_'][max_channel>self.num_channels]=self.num_channels+1
        #overflows complete

        included=(self.graph.ndata['x_']!=0) & (self.graph.ndata['x_']!=self.num_channels+1)
        expanded_nodes=self.graph.nodes().unsqueeze(1).expand_as(self.graph.ndata['x_'])
        index=expanded_nodes*(self.num_channels+1)+self.graph.ndata['x_']

        if self.num_samples==2:
            ch1=torch.zeros(self.num_nodes*(self.num_channels+1)+1,device=self.device)
            ch1[index[:,0].int()]=1
            ch1=torch.reshape(ch1[:-1],(self.num_nodes,self.num_channels+1))
            ch1[~included[:,0]]=0

            ch2=torch.zeros(self.num_nodes*(self.num_channels+1)+1,device=self.device)
            ch2[index[:,1].int()]=1
            ch2=torch.reshape(ch2[:-1],(self.num_nodes,self.num_channels+1))
            ch2[~included[:,1]]=0
            self.graph.ndata['ch']=torch.cat((ch1.unsqueeze(1),ch2.unsqueeze(1)),dim=1)
        else:
            ch1=torch.zeros(self.num_nodes*(self.num_channels+1)+1,device=self.device)
            ch1[index.flatten().int()]=1
            ch1=torch.reshape(ch1[:-1],(self.num_nodes,self.num_channels+1))
            ch1[~included.flatten()]=0
            self.graph.ndata['ch']=ch1.unsqueeze(1)
        self.graph.ndata['ch'][:,:,0]=0

        #removing overlaps
        self.graph.update_all(message_func=fn.copy_u('ch','m'),reduce_func=fn.sum('m','sum'))
        self.graph.ndata['sum']=self.graph.ndata['ch']*self.graph.ndata['sum']
        clashed=torch.any(self.graph.ndata['sum']>0,axis=2)*(included)
        self.graph.ndata['x_'][clashed&undecided]=self.num_channels+1 #remove undecided condition for hard rollback
        self.graph.ndata['ch'][clashed&undecided]=0

        #recalculating features
        self.graph.update_all(message_func=fn.copy_u('ch','m'),reduce_func=fn.sum('m','sum'))
        self.graph.ndata['sum']=(self.graph.ndata['sum'])>0
        self.graph.ndata['ch']=1-self.graph.ndata.pop('sum').float()
        self.graph.ndata['ch'][self.graph.ndata['ch']==0]=-1

        #filling timeout with 0, can't be decided
        still_undecided=self.graph.ndata['x_']==self.num_channels+1
        timeout=self.t==self.max_epi_t
        self.graph.ndata['x_'][still_undecided&timeout]=0
        
        
        #calculating reward
        # rew_dem= self.demand_value(self.graph)-current_sol_dem
        # rew_channel= current_sol_ch-self.channel_value(self.graph)
        # rew = 0.7*rew_dem+ 0.3*rew_channel
        self.graph.ndata['x'] = self.graph.ndata.pop('x_')

        if self.hamming_reward_coef > 0.0 and self.num_samples == 2:
            xl, xr = self.graph.ndata['x'].split(1, dim = 1)
            undecidedl, undecidedr = undecided.split(1, dim=1)
            hamming_d = torch.abs(xl.float() - xr.float())
            hamming_d[(xl == self.num_channels+1) | (xr == self.num_channels+1) | (xl == 0) | (xr == 0)] = 0.0
            hamming_d[~undecidedl & ~undecidedr] = 0.0
            #print(hamming_d)
            self.graph.ndata['h'] = hamming_d
            hamming_reward = dgl.mean_nodes(self.graph, 'h').expand_as(rew)
            self.graph.ndata.pop('h')
            # print(hamming_d)
            # print('mean',hamming_reward)
            hamming_reward=hamming_reward/self.num_channels
            # print('normalized',hamming_reward)
            # print(rew)
            # a=input()
            rew += self.hamming_reward_coef * hamming_reward
    
        #handling batch dones
        done=self._check_done()

        return 0, done
    
        
    def _check_done(self):
        
        self.graph.ndata['ud']=(self.graph.ndata['x']==self.num_channels+1).float()
        done_graphs=(1-dgl.max_nodes(self.graph,'ud')).bool()
        self.graph.ndata.pop('ud')
        return done_graphs
    
    
    def _build_ob(self):
        ob_x = self.graph.ndata['x'].unsqueeze(2).float()
        ob_t = self.t.unsqueeze(2).float() / self.max_epi_t
        ob = torch.cat([ob_x, ob_t,self.graph.ndata['ch']], dim = 2)
        return ob
    
        
    def register(self,max_epi_t,g,num_samples=1):
        self.max_epi_t= max_epi_t
        self.graph=g
        self.num_nodes=self.graph.number_of_nodes()
        self.num_samples=num_samples
        self.demand=self.graph.ndata['value'].unsqueeze(1).expand(-1,self.num_samples)
        self.graph.ndata['dem']=self.demand
        #print(self.demand)
        self.graph.ndata['x']=torch.ones((self.num_nodes,self.num_samples),device=self.device)*(self.num_channels+1) #state
        self.t = torch.zeros((self.num_nodes,self.num_samples), dtype = torch.long, device = self.device)
        self.graph.ndata['ch']=torch.ones((self.num_nodes,self.num_samples,self.num_channels+1),device=self.device)
        ob=self._build_ob()
        return ob

if __name__=='__main__':

    graph = nx.Graph()
    # Add nodes to the graph
    graph.add_nodes_from([0,1, 2, 3,4])
    # Add edges to the graph
    graph.add_edges_from([(0,1), (1, 2),(3, 4)])
    #demand=np.array([4,2,4,4,2])
    demand=[1,1,1,1,1]
    dgl_graph_1 = dgl.from_networkx(graph)
    dgl_graph_1.ndata['value']= torch.tensor(demand).float()


    graph2=nx.Graph()
    graph2.add_nodes_from([0,1,2,3])
    graph2.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    dgl_graph_2 = dgl.from_networkx(graph2)
    #dgl_graph_2.ndata['value']= torch.tensor([5,2,3,2]).float()
    dgl_graph_2.ndata['value']= torch.tensor([1,1,1,1]).float()

    graph_list=[dgl_graph_1,dgl_graph_2]
    batched_graph = dgl.batch(graph_list)
    env=ChannelBonding()
    ob=env.register(3,batched_graph,1)
    #action1=[1,5,16,1,16,1,6,8,9,1,6,14,1,1,2,1,4,7]
    #action1=[1,5,16,16,10,1,6,8,10,1,5,16,1,10,1,6,8,10]
   # action2=[1,5,101,1,101,1,6,8,9]
    #action=torch.tensor(action1).reshape(2,9).T
    action=torch.tensor([1,5,16,16,10,1,6,8,10]).unsqueeze(1)
    print(action)
    ob,reward,done=env.step(action)
    print('observation',ob)
    print(reward)
    print(done)
    action=torch.tensor([1,5,5,16,10,1,6,8,10]).unsqueeze(1)
    print(action)
    ob,reward,done=env.step(action)
    print('observation',ob)
    print(reward)
    print(done)
    # action2=torch.tensor([101,101,5,101,5,1,2,3,101]).reshape(9,1)
    # ob,feat2,reward,done,gr=env.step(action2)
    # print(reward)
    # print(ob)
    # print(feat2)
    # print(done)


    # print(env.step(action))
    # state,reward,done,flag,info=env.step(action)
    # print(state,reward,done,info)
    # action2=torch.tensor([0,5,1,-1,-1,1,2,3,4]).reshape(9,1)
    # print(env.step(action2))
    # state,reward,done,flag,info=env.step(action2)
    # print(state,reward,done,info)
    # print("soln:",env.solution)
