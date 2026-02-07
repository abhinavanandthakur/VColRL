import torch
import torch.nn as nn
from torch.distributions import Categorical
import dgl


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor_class,
        critic_class, 
        max_num_nodes, 
        hidden_dim,
        num_layers,
        num_colors,
        device
        ):
        super(ActorCritic, self).__init__()
        self.num_colors=num_colors
        self.actor_net = actor_class(self.num_colors+1, hidden_dim, self.num_colors+1, num_layers)
        self.critic_net = critic_class(self.num_colors+1, hidden_dim, 1, num_layers)
        self.device = device
        self.to(device)
        self.max_num_nodes = max_num_nodes
        
    #this function gives the masks, idxs, subgraph and h for the given ob and g so that the action can be taken on a subgraph and mapped back to the original graph
    def get_masks_idxs_subg_h(self, ob, g):

        node_mask = (ob.select(2, 0).long() == self.num_colors+1)
        flatten_node_idxs = node_mask.reshape(-1).nonzero().squeeze(1)
        
        # num_subg_nodes
        subg_mask = node_mask.any(dim = 1)
        flatten_subg_idxs = subg_mask.nonzero().squeeze(1)
        
        # num_subg_nodes * batch_size
        subg_node_mask = node_mask.index_select(0, flatten_subg_idxs)
        flatten_subg_node_idxs = subg_node_mask.view(-1).nonzero().squeeze(1)

        g = g.to(self.device)
        subg = g.subgraph(flatten_subg_idxs)
     
        h = ob[:,:,2:].index_select(0, flatten_subg_idxs)
        return (
            (node_mask, subg_mask, subg_node_mask), 
            (flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs),
            subg, 
            h   
            )
    #this function sees the state of the graph and returns the action to be taken 
    def act(self, ob, g):
        num_nodes, batch_size = ob.size(0), ob.size(1)
        
        masks, idxs, subg, h = self.get_masks_idxs_subg_h(ob, g)
        node_mask, subg_mask, subg_node_mask = masks
        flatten_node_idxs, flatten_subg_idxs, flatten_subg_node_idxs = idxs

        # compute logits to get action
        logits = (
            self.actor_net(
                h, 
                subg,
                mask = subg_node_mask
                )
            .view(-1, self.num_colors+1)
            .index_select(0, flatten_subg_node_idxs)
            )
        
        # get actions
        action = torch.zeros(
            num_nodes * batch_size,
            dtype = torch.long, 
            device = self.device
            )   
        m = Categorical(
            logits = logits.view(-1, logits.size(-1))
            )
        action[flatten_node_idxs] = m.sample()        
        action = action.view(-1, batch_size)
        
        return action
    
    
