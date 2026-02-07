import argparse
import os
import matplotlib.pyplot as plt
import copy


parser=argparse.ArgumentParser()
parser.add_argument(
    "--valfile",
    type=str,
    default="validation_stats.txt",
)


parser.add_argument(
    "--sat",
    help='graph satisfaction threshold',
    type=int,
    default=95

)

parser.add_argument(
    "--samples",
    help='number of validation graphs',
    type=int

)

args=parser.parse_args()
val_file=args.valfile

def normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    range_val = max_val - min_val

    if range_val == 0:
        # Avoid division by zero if all values are the same
        return [0 for _ in lst]
    
    normalized_lst = [(x - min_val) / range_val for x in lst]
    return normalized_lst

with open(val_file,"r") as file:
    sat=[]
    avg_loss=[]
    loss=0
    loss_counter=0
    optimum_soln=[]
    temp=[]
    best_soln=[]
    current_soln=[]
    best_flag=[]
    current_flag=[]
    color_loss=5000
    min_color_used=1000
    reward=-5000
    val=[]
    best_model=0
    best_model_colors=0
    count=0
    comp_val=0
    local_count=0
    avg_reward=[]
    color_used=[]
    average_color_used=0
    for line in file:
        line=line.strip()
        if line:
            val=line.split()
            if len(val)==1:
                local_count+=1
                if local_count%2!=0:
                    comp_val=int(float(val[0]))
                else:
                    avg_reward.append(float(val[0]))
            if len(val)==3 :
                average_color_used+=int(val[1])
                current_soln.append(int(val[1]))
                temp.append(int(val[2]))
                if int(float(val[0]))==100:
                    current_flag.append(True)
                    loss_counter+=1
                    loss+=int(val[1])-int(val[2])
                else:
                    current_flag.append(False)

        else:
            count+=1
            color_used.append(average_color_used/args.samples)
            average_color_used=0
            optimum_soln=temp
            sat.append(int(loss_counter*100/args.samples))
            try:
                avg_loss.append(loss/loss_counter)
            except:
                avg_loss.append(100)
            if comp_val>=args.sat and  avg_loss[-1]<color_loss:
            # if comp_val>=args.sat and  avg_reward[-1]>reward:
                best_soln=copy.deepcopy(current_soln)
                best_flag=copy.deepcopy(current_flag)
                best_model=count
                #reward=avg_reward[-1]
                color_loss=avg_loss[-1]
                
            if comp_val>=args.sat and  color_used[-1]< min_color_used:
                best_model_colors=count
                min_color_used=color_used[-1]
            
            #print(round(loss_counter*100/args.samples,2))
            
            loss=0
            loss_counter=0
            current_soln=[]
            current_flag=[]
            temp=[]
            
    
    print(f'best model for atleast {args.sat} % graphs satisfied with least color used is {best_model_colors} with average color usage of {min_color_used} for validation graphs')
    
    x=[i for i in range(1,len(sat)+1)]
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(20,10))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Graph Satisfaction %")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Average color Loss")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Average Reward")

    ax1.plot(x, sat, linestyle='-', color='blue',linewidth=2) #color=(0.5,0.5,0.5)
    ax2.plot(x, avg_loss, linestyle='-', color='blue',linewidth=2)
    ax3.plot(x, avg_reward, linestyle='-', color='blue',linewidth=2)
    plt.show()

    

    

    

