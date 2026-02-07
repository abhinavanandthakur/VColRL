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

# environment for vertex coloring problem
class VCP(object):
    
    def __init__(self,max_epi_t=32,hamming_reward_coef=0.1,device='cpu',num_colors=15):
        
        self.max_epi_t= max_epi_t
        self.device=device
        self.hamming_reward_coef = hamming_reward_coef
        self.num_colors=num_colors

    #function to calculate the vertex satisafaction value for a state, used in reward calculation
    def vertex_sat(self,g):
        norm=dgl.sum_nodes(g,'helper')
        obs=g.ndata['x_'].clone()
        mask=(obs!=0) & (obs!=self.num_colors+1)
        g.ndata['mask']=mask
        val=dgl.sum_nodes(g,'helper','mask')
        g.ndata.pop('mask')
        return val/norm

    #function to calculate the color usage value for a state, used in reward calculation
    def color_value(self,g):
        
        obs=g.ndata['x_'].clone()
        obs[obs==self.num_colors+1]=0
        max_color=obs
        g.ndata['max_color'] =max_color
        val=dgl.max_nodes(g,'max_color')
        g.ndata.pop('max_color')
        return val/self.num_colors

    #step function to take an action and return the next observation, reward and done
    def step(self,action):
        reward,done=self._take_action(action)
        ob=self._build_ob()
        return ob,reward,done
    
    
    def _take_action (self,action):

        self.graph.ndata['x_']= self.graph.ndata['x'].clone()

        current_sol_ch=self.color_value(self.graph)
        current_sol_dem=self.vertex_sat(self.graph)

        #taking action
        undecided=(self.graph.ndata['x_']==self.num_colors+1)
        self.graph.ndata['x_'][undecided] = (action)[undecided].float()
        self.t+=1

        #making binary feature vector
        included=(self.graph.ndata['x_']!=0) & (self.graph.ndata['x_']!=self.num_colors+1)
        expanded_nodes=self.graph.nodes().unsqueeze(1).expand_as(self.graph.ndata['x_'])
        index=expanded_nodes*(self.num_colors+1)+self.graph.ndata['x_']

        ch1=torch.zeros(self.num_nodes*(self.num_colors+1)+1,device=self.device)
        ch1[index.flatten().int()]=1
        ch1=torch.reshape(ch1[:-1],(self.num_nodes,self.num_colors+1))
        ch1[~included.flatten()]=0
        self.graph.ndata['ch']=ch1.unsqueeze(1)
        self.graph.ndata['ch'][:,:,0]=0

        #removing overlaps
        self.graph.update_all(message_func=fn.copy_u('ch','m'),reduce_func=fn.sum('m','sum'))
        self.graph.ndata['sum']=self.graph.ndata['ch']*self.graph.ndata['sum']
        clashed=torch.any(self.graph.ndata['sum']>0,axis=2)*(included)
        self.graph.ndata['x_'][clashed]=self.num_colors+1 #remove undecided condition for hard rollback
        self.graph.ndata['ch'][clashed]=0

        #recalculating features
        self.graph.update_all(message_func=fn.copy_u('ch','m'),reduce_func=fn.sum('m','sum'))
        self.graph.ndata['sum']=(self.graph.ndata['sum'])>0
        self.graph.ndata['ch']=1-self.graph.ndata.pop('sum').float()
        self.graph.ndata['ch'][self.graph.ndata['ch']==0]=-1

        #filling timeout with 0, can't be decided
        still_undecided=self.graph.ndata['x_']==self.num_colors+1
        timeout=self.t==self.max_epi_t
        self.graph.ndata['x_'][still_undecided&timeout]=0
        
        
        #calculating reward
        rew_dem= self.vertex_sat(self.graph)-current_sol_dem
        rew_color= current_sol_ch-self.color_value(self.graph)
        rew = 2*rew_dem+ rew_color
        self.graph.ndata['x'] = self.graph.ndata.pop('x_')
    
        #handling batch dones
        done=self._check_done()

        return rew, done
    
    #function to check the done status 
    def _check_done(self):
        
        self.graph.ndata['ud']=(self.graph.ndata['x']==self.num_colors+1).float()
        done_graphs=(1-dgl.max_nodes(self.graph,'ud')).bool()
        self.graph.ndata.pop('ud')
        return done_graphs
    
    #taking the state and time step as input and returning the observation (including features to be passed to the GNN)
    def _build_ob(self):
        ob_x = self.graph.ndata['x'].unsqueeze(2).float()
        ob_t = self.t.unsqueeze(2).float() / self.max_epi_t
        ob = torch.cat([ob_x, ob_t,self.graph.ndata['ch']], dim = 2)
        return ob
    
    #register function to register the graph and other parameters for the environment   
    def register(self,max_epi_t,g,num_samples=1):
        self.max_epi_t= max_epi_t
        self.graph=g
        self.num_nodes=self.graph.number_of_nodes()
        self.num_samples=num_samples
        self.graph.ndata['helper']=self.graph.ndata['helper_value'].unsqueeze(1).expand(-1,self.num_samples)
        self.graph.ndata['x']=torch.ones((self.num_nodes,self.num_samples),device=self.device)*(self.num_colors+1) #state
        self.t = torch.zeros((self.num_nodes,self.num_samples), dtype = torch.long, device = self.device)
        self.graph.ndata['ch']=torch.ones((self.num_nodes,self.num_samples,self.num_colors+1),device=self.device)
        ob=self._build_ob()
        return ob

    