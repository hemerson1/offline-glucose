#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:44:34 2022

"""

import numpy as np 

from .general import calculate_risk

"""
Perform a grid search using the provided values to determine the optimal
PID parameters for a given patient.
"""
def optimal_pid_search(env, bas, cr, cf, k_ps, k_is, k_ds, num_days=10, target_bg=144): 
    
    # initialise the parameters
    listed_rewards = list() 
    max_timesteps = 480 * num_days
    counter, current_index = 0, 0
    current_max = -1000000000
    

    for k_p in k_ps:
        for k_i in k_is:
            for k_d in k_ds:
                
                max_val = 0
                
                # reset the seed
                env.seed(0)
                
                done, bg_val = False, env.reset()  
                rewards, timesteps, meal = 0, 0, 0
                
                # create the state
                meal_history = np.zeros(int((3 * 60) / 3))            
                integrated_state = 0
                previous_error = 0
                
                while not done and timesteps < max_timesteps:
                    
                    # proportional control
                    error = target_bg - bg_val[0] 
                    p_act = k_p * error
    
                    # integral control        
                    integrated_state += error
                    i_act = k_i * integrated_state
    
                    # derivative control
                    d_act = k_d * (error - previous_error)   
                    
                    # get the combined pid action
                    previous_error = error
                    action = (p_act + i_act + d_act + bas) / 3 
                    chosen_action = max(action, 0)
                    
                    # keep track of the max insulin dose
                    if action > max_val: max_val = action
                    
                    # get the bolus dose
                    bolus = 0
                    if meal > 0:                     
                        bolus = meal / cr
                        if np.sum(meal_history) == 0: 
                            bolus += (bg_val[0] - target_bg) / cf                        
                    chosen_action += max(bolus/3, 0)             
                    
                    # step the environment
                    next_bg_val, _, done, info = env.step(chosen_action)
                    reward = - calculate_risk(next_bg_val[0])
                    if done: reward -= 1e5
                        
                    # update the state
                    meal_history = np.append(meal_history, meal)
                    meal_history = np.delete(meal_history, 0)
                    
                    # update the state and memory
                    bg_val = next_bg_val
                    rewards += reward
                    timesteps += 1                
                    meal = info['meal']                
                
                counter += 1
                
                # keep track of the max reward
                if timesteps == max_timesteps:
                    
                    if reward > current_max:
                        current_max = rewards
                        current_index = counter
                    
                    data = {
                        "params" : "kp: {}, ki: {}, kd: {}".format(k_p, k_i, k_d),
                        "reward" : rewards, "max_val": max_val/bas
                        }
                    listed_rewards.append(data)
                
                # display the run results
                print('#{} kp:{} ki:{} kd:{} -- Reward: {} Timesteps: {}'.format(counter, str(k_p), str(k_i), str(k_d), rewards, timesteps))
                print('Max Action {}'.format(max_val/bas))
                print('--------------------------------')
    
    # display the best completed runs 
    print('Best run {}'.format(current_index))
    sorted_rewards = sorted(listed_rewards, key=lambda d: d['reward'], reverse=True)
    for idx, val in enumerate(sorted_rewards):
        print("Rank: {} | Reward {} | {} | {}".format(idx + 1, val["reward"], val["params"], val["max_val"])) 