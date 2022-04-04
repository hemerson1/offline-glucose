#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:28:57 2022
"""

"""
Functions for generating datasets from the glucose dynamics simulator.
"""

import numpy as np
import pickle
from collections import defaultdict

from .general import PID_action, calculate_bolus, calculate_risk


"""
Create a replay with a mixture of expert data and random data.
"""
def fill_replay_split(env, replay_name, data_split=0.5, replay_length=100_000, meal_announce=0,
                      noise=False, bolus_noise=None, bolus_overestimate=0.0, seed=0, params=None):
    
    # determine the split of the two datasets
    random_timesteps = int(data_split * replay_length) 
    expert_timesteps = int(replay_length - random_timesteps)     
    
    # Random data generation -------------------------------------------------
    
    new_replay = None  
    if random_timesteps > 0:
    
        new_replay = fill_replay(replay_length=random_timesteps,env=env, 
                                 replay_name=replay_name,
                                 player="random", bolus_noise=bolus_noise, 
                                 noise=False, meal_announce=meal_announce,
                                 bolus_overestimate=bolus_overestimate,
                                 seed=seed, params=params
                                 )
        print('Buffer Full with Random policy of size {}'.format(random_timesteps))  
    
    # Expert data generation -------------------------------------------------
    
    if expert_timesteps > 0:        
    
        full_replay = fill_replay(replay_length=expert_timesteps, 
                                  replay_name=replay_name,
                                  replay=new_replay,
                                  env=env, player="expert",
                                  bolus_noise=bolus_noise, seed=seed, 
                                  bolus_overestimate=bolus_overestimate,
                                  meal_announce=meal_announce,
                                  noise=noise,
                                  params=params
                                  )
        print('Buffer Full with Expert policy of size {}'.format(expert_timesteps))        
    else: 
        return new_replay
    
    # return the finished replay
    return full_replay


"""
Create a named replay of specified size with either a random or expert 
demonstrator. The replay produced is a list containing individual trajectories
stopping when the agent terminates or the max number of days is reached.
"""

def fill_replay(env, replay_name, replay=None, replay_length=100_000, player='random', meal_announce=0.0,
                bolus_noise=None, seed=0, params=None, noise=False, bolus_overestimate=0.0):
    
    # Unpack the additional parameters
    
    # Environment
    days = params.get("days", 10)  
    
    # Diabetes
    basal_default = params.get("basal_default")
    target_blood_glucose = params.get("target_blood_glucose")
    
    # PID
    kp, ki, kd = params.get("kp"), params.get("ki"), params.get("kd")
    
    # Bolus
    cr, cf = params.get("carbohydrate_ratio"), params.get("correction_factor")
        
    # OU Noise
    sigma = params.get("ou_sigma", 0.2) 
    theta = params.get("ou_theta", 0.0015) 
    dt = params.get("ou_dt", 0.9) 
    
    # seed numpy and the environment
    seed = seed
    np.random.seed(seed)
    env.seed(seed)
    
    # create the replay
    if replay is None: replay = []    
    buffer_not_full = True 
    replay_progress_freq = replay_length // 10    
    
    # Specify the counter for total timesteps
    counter = 0 
    episode_max = 480 * days
    
    while buffer_not_full:            

        # get the starting state
        insulin_dose = np.array([1/3 * basal_default], dtype=np.float32)
        meal, done, bg_val = 0, False, env.reset()
        time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
        state = np.array([bg_val[0], meal, insulin_dose[0], time], dtype=np.float32)
        
        # get the meal history for the last 3 hrs
        meal_history = np.zeros(60)
        
        # intiialise the PID and OU noise parameters
        integrated_state, previous_error = 0, 0
        prev_ou_noise = 0
        
        # record the trajectory and the current timestep
        trajectory = defaultdict(list)
        episode_timestep = 0

        while not done:  
            
            # select the basal dose ------------------------------------------
            
            # calculate the OU noise from the initial parameters 
            if noise:
                ou_noise = (prev_ou_noise + theta * (0 - prev_ou_noise) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(1,))[0])
                ou_noise = ou_noise * basal_default
                prev_ou_noise = ou_noise
            
            if player == "random":
                               
                # add the noise to a baseline
                action = np.array([1/3 * basal_default])
                agent_action = np.copy(action + ou_noise)
                
            elif player == "expert":
                
                # add the noise to a PID agent dose
                action, previous_error, integrated_state = PID_action(
                    blood_glucose=bg_val, 
                    previous_error=previous_error,
                    integrated_state=integrated_state, 
                    target_blood_glucose=target_blood_glucose, 
                    kp=kp, ki=ki, kd=kd, basal_default=basal_default
                )                
                agent_action = np.copy(action)
                
            # add on the noise    
            if noise: agent_action += ou_noise
            chosen_action = agent_action
                
            # select the bolus dose ------------------------------------------

            if meal > 0: 
                
                # save the adjusted meal
                adjusted_meal = meal                  
                
                # add some calculation error in bolus
                if bolus_noise:
                    adjusted_meal += bolus_noise * adjusted_meal * np.random.uniform(-1, 1, 1)[0]   
                
                # add a bias to the bolus estimation
                adjusted_meal += bolus_overestimate * meal  
                
                # calculate the bolus dose 
                chosen_action = calculate_bolus(
                    blood_glucose=bg_val, meal_history=meal_history, 
                    current_meal=adjusted_meal, carbohyrdate_ratio=cr,
                    correction_factor=cf, 
                    target_blood_glucose=target_blood_glucose
                    )
                
                # amend the agent action to the dose
                chosen_action += agent_action

            # take a step in the environment ----------------------------------
            
            # update the state and get the true reward
            next_bg_val, _, done, info = env.step(chosen_action)                
            reward = -calculate_risk(next_bg_val)    
            
            # announce a meal -------------------------------------------
            
            # meal announcement
            meal_input = meal
            if meal_announce != 0.0:
                
                # get times + meal schedule
                current_time = env.env.time.hour * 60 + env.env.time.minute
                future_time = current_time + meal_announce - 1
                meal_scenario = env.env.scenario.scenario["meal"]
                future_meal = 0
                
                # check for future meal                
                if future_time in meal_scenario["time"]:                    
                    index = meal_scenario["time"].index(future_time)
                    future_meal = meal_scenario["amount"][index] 
                    
                meal_input = future_meal / 3
                
            # configure the next state  ----------------------------------------
            
            time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479            
            next_state = np.array([next_bg_val[0], meal_input, chosen_action[0], time], dtype=np.float32)                    
            
            # add a termination penalty
            if done: reward = -1e5
            
            # update the replay ---------------------------------------------
                       
            # update the replay with trajectory
            sample = [('reward', reward), ('state', state), ('next_state', next_state), ('action', agent_action), ('done', done)]
                        
            for key, value in sample:
                trajectory[key].append(value)
            
            # add the new trajectory to the replay
            if done or episode_timestep == episode_max:
                replay.append(trajectory)
                break
            
            # update the variables -----------------------------------------
            
            # update the meal history
            meal_history = np.append(meal_history, meal)
            meal_history = np.delete(meal_history, 0)               
            
            # update the state
            bg_val, state, meal = next_bg_val, next_state, info['meal']
            counter += 1
            episode_timestep += 1              
            
            # save the replay ----------------------------------------------
                        
            # visualise replay size
            if counter % replay_progress_freq == 0:
                print('Replay size: {}'.format(counter))
                with open("./Replays/" + replay_name + ".txt", "wb") as file:
                    pickle.dump(replay, file)
                    
            # full termination condition -----------------------------------

            # add termination when full
            if counter == replay_length: 
                buffer_not_full = False
                replay.append(trajectory)
                return replay
