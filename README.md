# VColRL: Learn to solve the Vertex Coloring Problem with Reinforcement Learning
VColRL is a reinforcement learning framework for solving the Vertex Coloring Problem (VCP), an NP-hard problem (the optimization version) that aims to color the vertices of a graph with the minimum number of colors such that no two 
adjacent vertices share the same color. The VCP has many important applications, such as frequency assignment in wireless communication, register allocation in compiler design, timetabling, map coloring, etc. 
In several scenarios, like ad-hoc networks and register allocation in Just-in-Time compilers, near-optimal solutions are sufficient but must be computed quickly. 
VColRL is designed to meet this requirement by leveraging reinforcement learning to learn effective coloring strategies. It employs Proximal Policy Optimization (PPO) for training and uses a Graph Neural Network (GraphSAGE architecture)
as the function approximation. This integration allows VColRL to handle graph structures efficiently while producing high-quality solutions within practical time constraints.


# Getting Started

## Cloning VColRL repository

To clone the VColRL repository.

```bash
git clone https://github.com/abhinavanandthakur/VColRL.git
```

## Setting up the virtual environment for VColRL

Create a conda environment from [vcolrl.yml](environments/vcolrl.yml)

```bash
conda env create -f environments/vcolrl.yml
```

Activate `VColRL` environment.

```bash
conda activate vcolrl
```

In the original paper, VColRL is compared against several baselines. All the baselines except ReLCol should be executed in the VolRL environment.

## Setting up the virtual environment for ReLCol

Create a conda environment from [relcol.yml](environments/relcol.yml)

```bash
conda env create -f environments/relcol.yml
```

Activate `ReLCol` environment.

```bash
conda activate relcol
```

## Getting the Academic License for Gurobi Optimizer
The Gurobi Solver will require a license to execute big optimizations. Visit [Gurobi Academic Program and Licenses](https://www.gurobi.com/academia/academic-program-and-licenses/) to know about the process.

# Training VColRL

VColRL is based on the MDP named HDM. The training code is present in [training_code](./training_code). 

To train the `VColRL` model, follow the steps given below:

```bash
cd training_code/vcolrl_hdm
conda activate vcolrl
python3 train.py
```

Something like this will appear on the terminal 

```bash
update_t: 00001
train stats...
Satisfied%: 68.0129, Objective%: -0.0338, actor_loss: -0.0196, critic_loss: 0.0528, entropy: 2.7448
update_t: 00002
train stats...
Satisfied%: 71.9032, Objective%: -0.0157, actor_loss: -0.0055, critic_loss: 0.0688, entropy: 2.7413
update_t: 00003
train stats...
Satisfied%: 75.8664, Objective%: -0.0144, actor_loss: -0.0067, critic_loss: 0.0787, entropy: 2.7431
...
```
Train the model for at least **300 epochs** (approximately **150,000 updates** for a dataset of 15,000 graphs with batch size 32).  

The following outputs will be generated during training:

- `validation_stats.txt` — validation statistics  
- `models` — trained model checkpoints saved after every epoch  

All files will appear in the same directory where the training is executed.

In order to find the best-performing model, the file `validation_stats.txt` needs to be analyzed. Copy this file to [best model inference](./best_model_inference) folder and run the evaluation script.

```bash
cp validation_stats.txt ../../best_model_inference
python3 analyze.py --samples 1000
```
There are 1000 graphs in our validation dataset, and therefore, the `sample` flag is set to 1000. The following can be seen on the terminal.

```bash
best model for atleast 95 % graphs satisfied with least color used is 482 with average color usage of 6.101 for validation graphs
```
Go to the models folder inside the training directory and fetch this model (model number 482 in this case). For convenience, the trained models are provided in [trained models](./trained_models) folder. 
To train other variants, a similar procedure can be followed in the corresponding directories present in [training_code](./training_code). 


# Evaluating VColRL

The code to evaluate **VColRL** is available in the [vcolrl](./vcolrl) folder. The folder must contain a trained model with the name `model_vcolrl.pth`. 
At its core, VColRL processes input graphs in the form of [DGL](https://www.dgl.ai/) graphs.  

For convenience, we provide a high-level function that accepts graphs in the **`DIMACS.col`** format from a specified folder `benchmarks` and stores the evaluation results in a text file named `results_vcolrl.txt`. Both are located in the same directory. Run the following command to evaluate VColRL on the graphs present in the `benchmarks` folder.

```bash
cd vcolrl
python3 main_vcolrl.py
```

The following output can be seen on the terminal upon execution of the script

```bash
Processing********************1/52 ****************benchmarks/le450_5d.col*********************************************************************************************
450 9757
Validation Progress:  51%|██████▌      | 65/128 [00:00<00:00, 377.29iteration/s]
```

The file `results_vcolrl.txt` contains the output in the following format.

```bash
le450_5d.col: 5 0.05399131774902344 6.55 1.3512059708858513 0.13013418674468993 0.09725692934436886
```
The entries for each graph correspond to the `color used (best_solution)`, `execution time`, followed by the means and standard deviations of both terms, respectively.

# Evaluating baselines
In the original paper, VColRL is compared against six baselines:  
[First Fit](./ff), [TabucolMin](./tabucolmin), [VColMIS](./vcolmis), [Gurobi](./gurobi), [FastColor](./fastcolor), and [ReLCol](./relcol).  

To evaluate any of these baselines, navigate to the corresponding directory and run the Python file starting with the name `main`.
An example procedure for [VColMIS](./vcolmis) is given below.

```bash
cd vcolmis
python3 main_vcolmis.py
```
The input–output procedure is the same as described in the [Evaluating VColRL](#evaluating-vcolrl) section. For some deterministic baselines, the output does not contain the mean and the standard deviation.

# Experiments

There are four major experiments presented in the VColRL paper:

a) Comparison of various MDP configurations. 
b) Comparison of VColRL with baselines on [Color02 & DIMACS](https://mat.tepper.cmu.edu/COLOR02/), [NDR](https://networkrepository.com/), and [SNAP](https://snap.stanford.edu/snap/) benchmarks.  
c) Comparison of VColRL with baselines on a wide range of synthetic graphs.  
d) Comparison of VColRL with two sequential models: one with 'defer' action and another without it.  

## Comparison of various MDP configurations. 

For this experiment, follow the training and best model inference procedure described in [Training VColRL](#training-vcolrl).  

During the best model inference stage, in addition to printing the best model details in the terminal, the script also generates a figure showing **Average Return**, **Average Colors Used**, and **Graph Satisfaction**. Example outputs for VColRL (left) and a model trained with SWC (right), trained up to 150 epochs, are shown below.
<!-- <img width="1000" height="500" alt="plot" src="https://github.com/user-attachments/assets/aa6c3b76-8cad-4f4a-a046-3b8602e0f09e" />-->

<img width="400" height="400" alt="plot_1" src="https://github.com/user-attachments/assets/8e5c08ff-90e9-40af-af04-9c3dead16980" />
<img width="400" height="400" alt="plot" src="https://github.com/user-attachments/assets/aa7cca58-46b7-4702-9832-be5307240d44" />




## Comparison of VColRL with baselines on benchmark graphs.

For this experiment, follow the steps to run VColRL and the baselines as described in [Evaluating VColRL](#evaluating-vcolrl) and [Evaluating Baselines](#evaluating-baselines).  The graphs present in the respective directories correspond to those evaluated in Tables 3, 4, and 5 of the original paper.  Note that for graphs with more than 10,000 nodes, an additional search step is applied in VColRL.  The code for evaluating large graphs is provided in [VColRL on Big Graphs](./experiment_helper/vcolrl_big). The procedure is the same as described in [Evaluating VColRL](#evaluating-vcolrl), but one should navigate to the [VColRL on Big Graphs](./experiment_helper/vcolrl_big) directory before running the scripts.

## Comparison of VColRL with baselines on a wide range of synthetic graphs.

For this experiment, we provide a helper directory present in [Synthetic Graph Experiment](./experiment_helper/table6). The `dataset` folder in the directories of the respective algorithm contains the dataset used to generate Table 6 of the original paper. 

To get the results for VColMIS, Gurobi, TabucolMin, and First-Fit, follow the steps.

```bash
cd experiment_helper/table6/vcolmis_gurobi_tabucolmin_ff
python3 main_compare.py --dataset test
```
Change the `dataset` flag as needed to evaluate on the respective datasets (e.g., er_50_100, ba_150_200, etc).

The following output can be seen on the terminal after completion

```bash
dataset/test.txt
5
number of nodes: 100 to 150 
Total number of samples tested: 5
Gurobi Performance: 100.0 5.2 5.79635334
FF Performance: 100.0 7.2 0.00008540
TabucolMin Performance: 100.0 5.2 5.85727673
VColMIS performance: 100.0 7.2 0.60046544
```
The three values represent the following:  
1. The percentage of the graphs in the dataset that are successfully evaluated
2. The average number of colors used.  
3. The average time required for evaluation.

Similar steps can be followed for all other algorithms in their respective directories present in [Synthetic Graph Experiment](./experiment_helper/table6). Some algorithms may not contain the first value in the output.

## Comparison of VColRL with sequential models.
VColRL is compared with two sequential models: `Sequential Defer (SD)` and `Sequential Without Defer (SW)`. The results of these sequential models on benchmark graphs, as presented in Table 7 of the original paper, can be obtained by running the code in [Sequential Models](./sequential_models). To run the experiments, navigate to the directory of the respective variant and follow the steps described in [Evaluating VColRL](#evaluating-vcolrl). The input–output format and procedure are the same.


# Acknowledgements

The following individuals have provided invaluable guidance and support for this project:
1) [Amitangshu Pal](https://www.cse.iitk.ac.in/users/amitangshu/)
2) [Subrahmanya Swamy Peruru](https://www.iitk.ac.in/new/subrahmanya-swamy-peruru)

# Bibliography

1) [Learning What to Defer for Maximum Independent Sets](https://proceedings.mlr.press/v119/ahn20a/ahn20a.pdf)
2) [A Reduction based Method for Coloring Very Large Graphs ](https://www.ijcai.org/proceedings/2017/73)
3) [Generating a graph colouring heuristic with deep Q-learning and graph neural networks](https://link.springer.com/chapter/10.1007/978-3-031-44505-7_33)
4) [On-line and first fit colorings of graphs](https://onlinelibrary.wiley.com/doi/abs/10.1002/jgt.3190120212)
5) [Using tabu search techniques for graph coloring](https://link.springer.com/article/10.1007/bf02239976)
6) [Gurobi Optimization](https://www.gurobi.com/solutions/gurobi-optimizer/)
7) [pchervi](https://github.com/pchervi/Graph-Coloring)







