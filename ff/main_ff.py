
import networkx as nx
from time import time
import os
import glob





filepath='./results_firstfit.txt'
result_file=open(filepath,'w')

   
def first_fit(G):
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

    # Return the number of colors used
    return max_color + 1  # +1 because colors start from 0



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



def compare(graph,file_name):
  
    nx_graph=graph
    print('greedy.......')
    #checking solution for greedy
    c=time()
    color_greedy=first_fit(nx_graph)
    d=time()-c
    print('greedy completed with solution:',color_greedy)

    result_file.write(f"{file_name}: {color_greedy} {d:.8f}\n")
    result_file.flush()

    



benchmarks= glob.glob('benchmarks/*.col')


num_graphs = len(benchmarks)
graph_count=0
for file_name in benchmarks:
    graph_count+=1
    print(f'\nProcessing********************{graph_count}/{num_graphs} ****************{file_name}*********************************************************************************************')
    nx_graph = load_dimacs_col_file(file_name)
    nodes=nx_graph.number_of_nodes()
    edges=nx_graph.number_of_edges()
    print(nodes,edges)
    
    try:
        a=compare(nx_graph,file_name.split('/')[-1])
    except:
        print("Some error occured")
        continue

    

result_file.close()
