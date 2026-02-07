import networkx as nx
from time import time
from pulp import *
import pulp
import glob




num_colors=15
cutoff_time=10 #seconds

filepath='./results_gurobi.txt'
result_file=open(filepath,'w')

   
def minimum_graph_coloring(graph,best_time_cb=5,max_colors=15):
 
    # Number of vertices in the graph
    num_vertices = graph.number_of_nodes()

    # Create the ILP problem
    problem = pulp.LpProblem("MinimumGraphColoring", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.matrix("x",(range(num_vertices),range(max_colors)),cat="Binary")
    y = pulp.LpVariable.matrix("y", range(max_colors), cat='Binary')

    # Objective: Minimize the number of colors used
    problem += pulp.lpSum(y), "MinimizeColors"

    # Each vertex must receive exactly one color _clubbed
    for i in range(num_vertices):
        problem += pulp.lpSum(x[i]) == 1, f"ColorAssignment_{i}"

    # No two adjacent vertices can share the same color and if any color is given to a node, it is marked with y
    for i, j in graph.edges():
        for c in range(max_colors):
            problem += x[i][c] + x[j][c] <= y[c], f"AdjacentColorConflict_{i}_{j}_{c}"

    # Solve the problem
    problem.solve(pulp.GUROBI_CMD(msg=0,options=[('timeLimit',best_time_cb)]))
    colors_used = pulp.value(problem.objective)
    try:
        return LpStatus[problem.status], int(colors_used)
    except:
        return LpStatus[problem.status], 0






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
    print('gurobi.......')
    a=time()
    status, color_pulp= minimum_graph_coloring(nx_graph,cutoff_time) #cutoff time chosen as 900 seconds; change it for faster results 
    b=time()-a
    print('gurobi completed with solution:',color_pulp)
    result_file.write(f"{file_name}: {status} {color_pulp} {b}\n")
    result_file.flush()

    

benchmarks=glob.glob('benchmarks/*.col')

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
