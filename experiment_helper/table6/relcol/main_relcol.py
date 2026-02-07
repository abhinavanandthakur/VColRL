import pickle
import os
import glob
import networkx as nx
import gym_graph_colouring
from utils import construct_target, convert_networkx_to_graph
from dataset_functions import *
from test_policy_functions import *
from dqn_agent import DQNAgentGN
from time import time
from datetime import datetime
from load_dataset import read_data
import argparse
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='er_50_100')
args=parser.parse_args()

PRINT_OUTPUT = True
SAVE_OUTPUT = False

# A string of the filenames of the trained parameters, saved in outputs/training
TRAINING_DIRNAMES = ['learned_parameters_GN']



DATASET_FILENAME = 'lemos_20221026_1355.pickle'
dataset_filepath = os.path.join('datasets', DATASET_FILENAME)

# Note runs per graphs is fixed in test_using_learned_policy function (in file test_policy_functions.py)

TEST_STOCHASTIC_POLICY = False

def get_multiple_policy_stats(dataset, all_stats):
    graph_names=[]
    for target_graph in dataset.graphs:
        graph_names.append(target_graph['name'])
    
    colours_used_by_graph = {}
    for graph_name in graph_names:
        colours_used_by_graph[graph_name] = np.zeros(len(TRAINING_DIRNAMES))

    for ind, training_run_stats in enumerate(all_stats):
        if TEST_STOCHASTIC_POLICY:
            deterministic_stats_all_graphs, deterministic_stats_by_graph, stochastic_stats_all_graphs, stochastic_stats_by_graph = training_run_stats
        else:
            deterministic_stats_all_graphs, deterministic_stats_by_graph = training_run_stats
    
        for individual_graph_stats in deterministic_stats_by_graph:
            colours_used_by_graph[individual_graph_stats['name']][ind] = individual_graph_stats['avg_colours_used']

    stats_by_graph = {}
    for graph_name in graph_names:
        stats_by_graph[graph_name] = {}
        stats_by_graph[graph_name]['mean'] = np.mean(colours_used_by_graph[graph_name])
        stats_by_graph[graph_name]['std'] = np.std(colours_used_by_graph[graph_name])
        stats_by_graph[graph_name]['min'] = np.amin(colours_used_by_graph[graph_name])
        stats_by_graph[graph_name]['max'] = np.amax(colours_used_by_graph[graph_name])

    return graph_names, stats_by_graph

def print_baselines(dataset):

    print('Dataset results')
    print('===============')
    print('\n')
    
    print('Random policy avg reward, avg colours and std of colours used by graph: ')
    for graph_stats in dataset.random_policy_stats_by_graph:
        print(graph_stats['name'], ': ', graph_stats['avg_episode_reward'], ' | ', graph_stats['avg_colours_used'], ' | ', graph_stats['colours_used_std'])
    print('On average:')
    print(dataset.random_policy_stats_combined['avg_episode_reward'], ' | ', dataset.random_policy_stats_combined['avg_colours_used'])
    
    print('---')
    print('DSATUR avg reward and avg colours used by graph: ')
    for graph_stats in dataset.dsatur_stats_by_graph:
        print(graph_stats['name'], ': ', graph_stats['episode_reward'], ' | ', graph_stats['colours_used'])
    print('On average:')
    print(dataset.dsatur_stats_combined['avg_episode_reward'], ' | ', dataset.dsatur_stats_combined['avg_colours_used'])


def print_single_policy_results(dataset, all_stats):
    
    print_baselines(dataset)

    if TEST_STOCHASTIC_POLICY:
        deterministic_stats_all_graphs, deterministic_stats_by_graph, stochastic_stats_all_graphs, stochastic_stats_by_graph = all_stats
    else:
        deterministic_stats_all_graphs, deterministic_stats_by_graph = all_stats
    
    print('---')
    print('Learned policy (deterministic) avg reward and avg colours used: ')
    for graph_stats in deterministic_stats_by_graph:
        print(graph_stats['name'], ': ', graph_stats['avg_episode_reward'], ' | ', graph_stats['avg_colours_used'])
    print('On average:')
    print(deterministic_stats_all_graphs['avg_episode_reward'], ' | ', deterministic_stats_all_graphs['avg_colours_used'])
    
    if TEST_STOCHASTIC_POLICY:
        print('---')
        print('Learned policy (stochastic) avg reward, avg colours used and min colours used: ')
        for graph_stats in stochastic_stats_by_graph:
            print(graph_stats['name'], ': ', graph_stats['avg_episode_reward'], ' | ', graph_stats['avg_colours_used'], ' | ', graph_stats['min_colours_used'])
        print('On average:')
        print(stochastic_stats_all_graphs['avg_episode_reward'], ' | ', stochastic_stats_all_graphs['avg_colours_used'], ' | ', stochastic_stats_all_graphs['avg_min_colours_used'])


def print_multiple_policy_results(graph_names, dataset, stats_by_training_run, stats_by_graph):
    
    print_baselines(dataset)

    print('---')
    print('Learned policy (deterministic) colours used stats by run across graphs:')
    print(f"{'run_no':<10}{'|'}{'mean' :<6}{'|'}{'std':<6}")
    print('-'*(10+6*2))
    run_means = np.zeros(len(stats_by_training_run))
    for ind, elt in enumerate(stats_by_training_run):
        run_means[ind] = elt[0]['avg_colours_used']
        print(f"""{ind:<10}{'|'}{f'''{elt[0]['avg_colours_used']:.2f}'''.rstrip('0').rstrip('.'):<6}{'|'}{f'''{elt[0]['colours_used_std']:.2f}'''.rstrip('0').rstrip('.'):<6}""")
    print('-'*(10+6*2))
    print('On average: ', np.mean(run_means))
    print('With std: ', np.std(run_means))
    
    print('---')
    print('Learned policy (deterministic) colours used stats by graph across runs:')
    print(f"{'graph_name':<20}{'|'}{'mean' :<6}{'|'}{'std':<6}{'|'}{'min':<6}{'|'}{'max':<6}")
    print('-'*(20+6*4))
    graph_means = np.zeros(len(graph_names))
    for ind, graph_name in enumerate(graph_names):
        graph_means[ind] = stats_by_graph[graph_name]['mean']
        print(f"""{graph_name:<20}{'|'}{f'''{stats_by_graph[graph_name]['mean']:.2f}'''.rstrip('0').rstrip('.'):<6}{'|'}{f'''{stats_by_graph[graph_name]['std']:.2f}'''.rstrip('0').rstrip('.'):<6}{'|'}{stats_by_graph[graph_name]['min']:<6.0f}{'|'}{stats_by_graph[graph_name]['max']:<6.0f}""")
    print('-'*(20+6*4))
    print('On average: ', np.mean(graph_means))
    print('With std: ', np.std(graph_means))
    
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


if __name__ == "__main__":
    
    
    env = gym.make('graph-colouring-v0')

    dataset = load_dataset(dataset_filepath)

    example_target, example_current_graph = env.reset(target=dataset.graphs[0])

    example_data, example_global_features = env.GenerateData_v1(example_target, example_current_graph)
    len_node_features = example_data.x.shape[1]
    len_edge_features = example_data.edge_attr.shape[1]
    len_global_features = len(example_global_features)

    agent = DQNAgentGN(len_node_features, len_edge_features, len_global_features, test_mode=True)
    test_model_path='test_GN'
    agent.load_models(test_model_path)
    
    
    
    file_path= f'datasets/{args.dataset}.txt'
    result_file=open(f'relcol_{args.dataset}.txt','w')
    print(file_path)
    test_graphs,_=read_data(file_path)
    #test_graphs=test_graphs[:5]
    num_graphs=len(test_graphs)
    print('num_graphs:', num_graphs)
    graph_count=0
    colors=[]
    timee=[]
    for nx_graph in test_graphs : #benchmarks.keys():
        graph_count+=1
        print(f'\nProcessing********************{graph_count}/{num_graphs} *************************************************************************************************************')
        no_vertices = nx_graph.number_of_nodes()
        no_edges=nx_graph.number_of_edges()
        
        if no_vertices<=5000 and no_edges<=100000:
            
            print(f'Constructing target {datetime.now()}..........')
            print(f'converting with nodes {no_vertices} and edges {no_edges}..........')
            t1=time()
            target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices = convert_networkx_to_graph(nx_graph.nodes, nx_graph.edges)
            target = construct_target(no_vertices, target_edge_list, target_edge_attr, edge_ind_dict, target_edge_indices, name='g'+str(graph_count))
            construct_time= time()-t1
            print(f'construct time: {construct_time} seconds. Processing for coloring............')
            try:
                color_time_begin=time()
                min_colors = test_using_learned_policy(env, target, agent, stochastic=False)
                total_time_without_construct=time()-color_time_begin
                total_time=time()-t1
                colors.append(min_colors)
                timee.append(total_time)
                result_file.write(f'{min_colors} {total_time} {total_time_without_construct}\n')
                result_file.flush()
                print('colors used',min_colors)
            except:
                result_file.write(f'Some error occured\n')
                print("Some error occured")
                continue
           
        else:
            print("Too big for processing")
            
    print(f'\nReLCol performance: {statistics.mean(colors)} {statistics.mean(timee)}')
    
    result_file.close()
    
    
