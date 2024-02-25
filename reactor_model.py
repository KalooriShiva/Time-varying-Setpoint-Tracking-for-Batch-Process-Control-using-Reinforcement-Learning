# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:31:10 2022

@author: abuth
"""

import numpy as np
from temp_setpoint import dt, n_tf, time_list

C1 = -0.09 # litre/min
C2 = 4.1492 * (10 ** 19)
Ea_R = 13550.0
k0 = 2.53 * (10**19)
# Q = np.array([[1, 0],[0, 0]])

T = 0
Ca = 1
max_temp, min_temp, max_conc, min_conc, num_temp, num_conc = 308, 293, 1, 0, 10, 10
temp_list = np.linspace(min_temp, max_temp, num_temp)
conc_list = np.linspace(min_conc, max_conc, num_conc)
tj_list = np.linspace(273, 318, num_temp)


def state_index(state):
    """ Calculate the index of a state in the state arrays
    Parameters:
        state: a 2 x 1 numpy array"""
    # Gap between two successive temperatures in the temperature list
    temp_delta = 0.5*(max_temp - min_temp) / (num_temp - 1)
    # Gap between two successive concentrations in the concentration list
    conc_delta = 0.5*(max_conc - min_conc) / (num_conc - 1)
    # Finding temperature index and concentration index with builtin numpy function
    if min_temp > state[T]: temp_index = 0
    elif max_temp < state[T]: temp_index = num_temp - 1
    else: temp_index = np.random.choice(np.where(abs(temp_list - state[T]) <= temp_delta)[0])
    if min_conc > state[Ca]: conc_index = 0
    elif max_conc < state[Ca]: conc_index = num_conc - 1
    else: conc_index = np.random.choice(np.where(abs(conc_list - state[Ca]) <= conc_delta)[0])
    return np.array([temp_index, conc_index])


def action_index(action):
    """ Calculates the index of action in action array 
    Parameters:
        action: a scalar value"""
    temp_delta = 0.5*(318 - 273) / (num_temp - 1)
    # finding action index with built in numpy function
    if 273 > action: temp_index = 0
    elif 318 < action: temp_index = num_temp - 1
    else: temp_index = np.random.choice(np.where(abs(tj_list - action) <= temp_delta)[0])
    return temp_index

def new_state(state, action):
    """ Calculate next state given the current state and action
    Parameters:
        state: a 2 x 1 numpy array
        action: a scalar value"""
    Tj = action
    
    # Improvement for more accuracy of state action model
    # if dt >= 0.01:
    #     r_temp = state[T]
    #     r_ca = state[Ca]
    #     try:
    #         for i in np.arange(0, dt, 0.001):
    #             # Reference expression: T(k+1) = T(k) + dt*[C1*(T(k) - Tj) + C2*exp(-Ea/RT(k))*Ca(k)^2]
    #             r_temp = float(r_temp + dt * (C1 * (r_temp  - Tj) + C2 * np.exp(-Ea_R/r_temp ) * (r_ca**2)))
    #             # Reference expression: Ca(k+1) = Ca(k) - dt*k0*exp(-Ea/RT(k))*Ca(k)^2
    #             r_ca = float(r_ca - dt * k0*(r_ca**2)*np.exp(-1*Ea_R/r_temp ))
    #             if r_ca < 0: new_Ca = 0
    #         new_state_val = np.array([r_temp, r_ca], dtype=np.float64).reshape(-1, 1)
    #     except Exception as e:
    #         print(i)
    #         raise(e)
    #     return new_state_val
    # Jacket temperature
    
    # Reference expression: T(k+1) = T(k) + dt*[C1*(T(k) - Tj) + C2*exp(-Ea/RT(k))*Ca(k)^2]
    new_T = float(state[T] + dt * (C1 * (state[T] - Tj) + C2 * np.exp(-Ea_R/state[T]) * (state[Ca]**2)))
    # Reference expression: Ca(k+1) = Ca(k) - dt*k0*exp(-Ea/RT(k))*Ca(k)^2
    new_Ca = float(state[Ca] - dt * k0*(state[Ca]**2)*np.exp(-1*Ea_R/state[T]))
    if new_Ca < 0: new_Ca = 0
    new_state_val = np.array([new_T, new_Ca], dtype=np.float64).reshape(-1, 1)
    return new_state_val


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    
    curr_state = np.array([300, 0.6]).reshape(-1, 1)
    state_arr = np.zeros_like(time_list)
    for i in range(n_tf):
        state_arr[i] = float(curr_state[T])
        curr_state = new_state(curr_state, 350)
    df = pd.DataFrame({"State": state_arr, "time": time_list})
    sns.set_theme()
    sns.lineplot(
        data=df,
        x="time",
        y="State",
    )
