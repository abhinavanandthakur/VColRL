import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import dgl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
                demand=list(map(int,current_list.pop(-1).split()))
                optimum_soln.append(demand[-1])
                demand=demand[:-1]
                g=nx.Graph()
                for temp in current_list:
                    node,*neighbors=temp.split()
                    g.add_node(int(node))
                    g.add_edges_from((int(node),int(neighbor)) for neighbor in neighbors)
                dgl_graph = dgl.from_networkx(g)
                dgl_graph.ndata['value']= torch.tensor(demand).float()
                dgl_graph=dgl_graph.to(device)
                graph_ob.append(dgl_graph)
                current_list=[]

    return graph_ob, optimum_soln

def split_list(graphs,train_ratio=0.8,shuffle=0):
    graphs=graphs
    if shuffle:
        random.shuffle(graphs)
    # Calculate the sizes of each portion
    # total_size = len(graphs)
    # train_size = int(total_size * train_ratio)
    # test_size=int(total_size*(1-train_ratio)/2)
    #splitting
    # train_graphs=graphs[:train_size]
    # test_graphs=graphs[train_size:train_size+test_size+1]
    # val_graphs=graphs[train_size+test_size+1:]
    train_graphs=graphs[:14000]

    #test_graphs=graphs[5000:6000]
    val_graphs=graphs[14000:15000]

    return train_graphs,[],val_graphs

class data_set(Dataset):

    def __init__(self,data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


    



# filepath='./dataset/graph_dataset.txt'

# _,graphs,demands=dataloader(filepath)

# # print(len(demands),len(graphs))
# # for i in range(5):
# #     print("A List=",graphs[i])
# #     print('Demand=',demands[i])
# #     print('\n')

# print(len(graphs))
# # Plot the graph
# nx.draw(graphs[4], with_labels=True)

# # Display the plot
# plt.show()



