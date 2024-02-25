import numpy as np
import pandas as pd
import seaborn as sns
from temp_setpoint import Tref, time_list, n_tf
from reactor_model import max_temp, min_temp, max_conc, min_conc, num_temp, num_conc, temp_list, conc_list, tj_list, state_index, new_state, T, Ca
import time

# states, states, action
# Reactor temperature, Reactor concentration, Jacket temperature
Q_matrix = np.zeros([n_tf, num_temp, num_conc, num_temp])
# discount factor for q learning
# optimal values obtained: 
# learning_rate = 0.8, num_episodes = 2500
gamma = 1
learning_rate = 0.01
epsilon = 0.05

def calc_reward(state, action, prev_action, iter_num):
    """ Calculate reward of the action done on state for the current iteration 
    Parameters: 
        state: a 2 x 1 numpy array
        action: a scalar value
        prev_action: a scalar value
        iter_num: integer"""
    Q = np.array([[1, 0],[0, 1]])
    R = 0.1
    if state is None: return -1*np.inf
    elif not (min_temp <= float(state[T]) <= max_temp): return -1*np.inf
    elif not (min_conc <= float(state[Ca]) <= max_conc): return -1*np.inf
    d_action = action - prev_action
    x = state - np.array([Tref[iter_num], 0]).reshape(-1, 1)
    cost_val = x.T @ (Q @ x) + R * (d_action**2)
    return -1*np.float64(cost_val)

def greedy_policy(state, time_index):
    """ Calculates action based on Q matrix.
    Parameters: 
        state: a 2 x 1 numpy array
        time_index: discrete time index in time_list"""
    index = state_index(state)
    action_array = Q_matrix[time_index, index[T], index[Ca]]
    tj_index = np.argmax(action_array)
    tj_indexes = np.where(action_array == action_array[tj_index])[0]
    tj_index = np.random.choice(tj_indexes)
    return tj_index

def policy(state, time_index, e):
    """ Calculates action based on Q matrix.
    Parameters: 
        state: a 2 x 1 numpy array
        time_index: discrete time index in time_list
        e: epsilon - probability of choosing a random action"""
    rand_number = np.random.uniform()
    if rand_number < e: return np.random.randint(num_temp)
    else: return greedy_policy(state, time_index)
    
def train_rl_model(num_episodes):
    # time, state, state, action
    # time, Reactor temperature, Reactor concentration, Jacket temperature
    global Q_matrix
    episode_versus_reward = np.zeros((num_episodes, 2))
    for episode_index in range(num_episodes):
        reactor_temp = np.random.choice(temp_list)
        reactor_conc = np.random.choice(conc_list)
        episode = []
        state = np.array([reactor_temp, reactor_conc]).reshape(-1, 1)
        for i in range(0, n_tf):
            # find action wrt policy that uses Q matrix
            jacket_temp_index = policy(state, i, epsilon)
            # get new state from reactor
            next_state = new_state(state, tj_list[jacket_temp_index])
            # calculate reward for action
            if i==0: prev_action = tj_list[jacket_temp_index]
            else: prev_action = tj_list[episode[i-1][1]]
            reward = calc_reward(state, tj_list[jacket_temp_index], prev_action, i)
            # add state action pair to state_action list
            episode.append([state_index(state), jacket_temp_index, reward, i])
            # get new state from discrete T and Ca values
            temp_index, conc_index = state_index(next_state)
            curr_temp_index, curr_conc_index = state_index(state)
            if i == n_tf - 1: 
                Q_matrix[i, curr_temp_index, curr_conc_index, jacket_temp_index] += \
                learning_rate*(reward - Q_matrix[i, curr_temp_index, curr_conc_index, jacket_temp_index])
            else: 
                next_jacket_temp_index = greedy_policy(next_state, i+1)
                Q_matrix[i, curr_temp_index, curr_conc_index, jacket_temp_index] += \
                learning_rate*(reward + gamma*Q_matrix[i+1, temp_index, conc_index, next_jacket_temp_index] 
                               - Q_matrix[i, curr_temp_index, curr_conc_index, jacket_temp_index])
            state = np.array([temp_list[temp_index], conc_list[conc_index]]).reshape(-1, 1)
        episode_versus_reward[episode_index] = np.array([episode_index, sum([transition[2] for transition in episode])])
    return episode_versus_reward


if __name__ == "__main__":
    start_time = time.perf_counter()
    episodes_versus_reward = train_rl_model(1_000_000)
    cpu_time = time.perf_counter() - start_time
    curr_state = np.array([298, 0.6]).reshape(-1, 1)
    state_arr = np.zeros_like(time_list)
    action_arr = np.zeros_like(time_list)
    conc_arr = np.zeros_like(time_list)
    for i in range(n_tf):
        state_arr[i] = float(curr_state[T])
        conc_arr[i] = float(curr_state[Ca])
        tj_index = greedy_policy(curr_state, i)
        action_arr[i] = tj_list[tj_index]
        curr_state = new_state(curr_state, tj_list[tj_index])
    df = pd.DataFrame({"State": state_arr, "time": time_list, "Reference": Tref, "Concentration [A]": conc_arr, "Jacket Temperature": action_arr})
    sns.set_theme()
    sns.lineplot(
        data=df,
        x="time",
        y="State",
        label="Reactor temperature",
    )
    sns.lineplot(
        data=df,
        x="time",
        y="Reference",
        legend="full",
        label="Reference temperature",
    )
    df.to_excel(f"TD_{n_tf}.xlsx")
    MAE = np.sum(np.abs(state_arr-Tref))/len(state_arr)
    RMSE = (np.sum((state_arr-Tref)**2)/len(state_arr))**0.5
    episode_vs_reward_df = pd.DataFrame({"Episodes": episodes_versus_reward[:, 0], "Reward": episodes_versus_reward[:, 1]})
    # sns.lineplot(episode_vs_reward_df, x="Episodes", y="Reward")
    days = int(cpu_time // 86400)
    hrs = int((cpu_time -  86400 * days) // 3600)
    mins = int((cpu_time - 3600 * hrs - 86400 * days) // 60)
    seconds = int((cpu_time - 60 * mins - 3600 * hrs - 86400 * days) // 1)
    print(f"{days = }\n{hrs = }\n{mins = }\n{seconds = }")