import math
import pickle

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def min_max_norm(lists):
    min_elem = min(lists[0])
    max_elem = max(lists[0])

    if len(lists) >1:
        for each in lists[1:]:
            min_elem = min(min(each),min_elem)
            max_elem = max(max(each),max_elem)

    res = []
    for each_list in lists:
        res.append([(each-min_elem)/(max_elem-min_elem) for each in each_list])

    return res

# seed = 3
mode = "dec"
seed = 3
'''
Model Utility vs Forget quality
'''
# retain_reward = [25.16008949499254] # dec -> grid world
# retain_reward = [30.615856195298097] # dec -> aircraft landing
# x = [25.16008949499254] if mode == "dec" else [25.140105583106703]
# y = [0]



x = []
y = []

epoch_list = [10]

for epoch in epoch_list:
    '''
    Unlearned model test performance on Retain Dataset
    '''
    # cumulative_reward_unlearn_original = pickle.load(open(f"trained_data/s{seed}-{mode}-cRewards-Retain.pkl","rb"))[50:]
    # cumulative_reward_unlearn_original = pickle.load(open(f"trained_data/s{seed}-{mode}-cRewards-Normal.pkl","rb"))[50:]
    cumulative_reward_unlearn_original = pickle.load(open(f"trained_data/s{seed}-{mode}-cRewards-Unlearn-epoch-{epoch}.pkl","rb"))[50:]
    norm_result = min_max_norm([cumulative_reward_unlearn_original])

    cumulative_reward_unlearn = norm_result[0]

    a = np.reshape(np.array(cumulative_reward_unlearn),(19,50)).tolist()

    b = []
    for each_row in a:
        temp = []
        for i in range(len(each_row)):
            temp.append(each_row[i]+sum(each_row[:i]))
        b.append(temp)

    '''
    cumulative_reward_retain shape 50,19
    test epoch 50
    '''
    cumulative_reward_unlearn = np.transpose(np.array(b),(1,0)).tolist()

    '''
    R_truth on forget set
    '''
    # r_truth_unlearn_origin = pickle.load(open(f"trained_data/s{seed}-{mode}-RTruth-Normal.pkl","rb"))[:50]
    r_truth_unlearn_origin = pickle.load(open(f"trained_data/s{seed}-{mode}-RTruth-Unlearn-epoch-{epoch}.pkl","rb"))[:50]
    r_truth_retain_origin = pickle.load(open(f"trained_data/s{seed}-{mode}-RTruth-Retain.pkl","rb"))[:50]

    r_truth_unlearn_sum = sum(r_truth_unlearn_origin[:50])
    r_truth_retain_sum = sum(r_truth_retain_origin[:50])

    r_truth_unlearn = [each/r_truth_unlearn_sum for each in r_truth_unlearn_origin[:50]]
    r_truth_retain = [each/r_truth_retain_sum for each in r_truth_retain_origin[:50]]

    for i in [50]:# select last epoch
        x.append(sum(cumulative_reward_unlearn[i-1])/len(cumulative_reward_unlearn[i-1])) # average reward for 19 env
        r_truth_unlearn_epoch = [sum(r_truth_unlearn[:j+1]) for j in range(i)]
        r_truth_retain_epoch = [sum(r_truth_retain[:j+1]) for j in range(i)]
        p_value = ks_2samp(r_truth_unlearn_epoch,r_truth_retain_epoch).pvalue
        y.append(math.log(p_value))

print("\"\"\"")
print(x)
print("======")
print(y)
print("\"\"\"")

data = {
    "Model Utility":x,
    "Forget Quality":y,
}

pd.DataFrame(data).to_csv(path_or_buf=f"trained_result/{mode}-result.csv",index=False)






