import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import dgl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#fuction to read the dataset file
def read_data(filepath,device):
    graph_ob=[]
    optimum_soln=[]
    with open(filepath,'r') as file:
        current_list=[]
        for line in file:
            line=line.strip()
            if line:
                current_list.append(line)
            else:
                helper=list(map(int,current_list.pop(-1).split()))
                optimum_soln.append(helper[-1])
                helper=helper[:-1]
                g=nx.Graph()
                for temp in current_list:
                    node,*neighbors=temp.split()
                    g.add_node(int(node))
                    g.add_edges_from((int(node),int(neighbor)) for neighbor in neighbors)
                dgl_graph = dgl.from_networkx(g)
                dgl_graph.ndata['helper_value']= torch.tensor(helper).float()
                dgl_graph=dgl_graph.to(device)
                graph_ob.append(dgl_graph)
                current_list=[]

    return graph_ob, optimum_soln

#function to split the dataset into training and validation sets
def split_list(graphs,train_ratio=0.8,shuffle=0):
    graphs=graphs
    if shuffle:
        random.shuffle(graphs)

    train_graphs=graphs[:14000]
    val_graphs=graphs[14000:15000]

    return train_graphs,val_graphs

#this class will be used to create dataloaders for training and validation
class data_set(Dataset):

    def __init__(self,data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]





