#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:29:00 2022
"""

"""
Functions for converting the replay output into the correct form for the 
chosen agent.
"""

import numpy as np
import random, torch

"""
Converts a list of trajectories gathered from the data_colleciton algorithm into
a replay with samples appropriate for model training along with state and action
means and stds.
"""
def unpackage_replay(trajectories, empty_replay, data_processing="condensed", sequence_length=80, params=None):
    
    # TODO: add functionality to change gamma if necessary
    gamma = 1.0   
    
    # initialise the data lists
    states, rewards, actions, dones = [], [], [], []
    
    for path in trajectories: 
        
        # states include blood glucose, meal carbs, insulin dose, time 
        states += path['state']
        rewards += path['reward']
        actions += path['action']  
        dones += path['done']
        
        # ensure that the last state is always a terminal state
        dones[-1] = True             
        
    # initialise the lists
    processed_states, processed_next_states, processed_rewards, processed_actions = [], [], [], []
    processed_dones, processed_timesteps, processed_reward_to_go, processed_last_actions  = [], [], [], []
    decay_state = np.arange(1 / (sequence_length + 2), 1, 1 / (sequence_length + 2))
    counter = 0 

    # Condense the state -------------------------------------------------

    # 4hr | 3.5hr | 3hr | 2.5hr | 2hr | 1.5hr | 1hr | 0.5hr | 0hr | meal_on_board | insulin_on_board
    if data_processing == "condensed":

        for idx, state in enumerate(states):
            
            # find the next done index
            if idx == 0 or dones[max(idx - 1, 0)]:
                done_index = idx + dones[idx:].index(True)

            # if there are 80 states previously
            if counter >= (sequence_length) and idx + 1 != len(states):  

                # add rewards, actions, dones and timestep label
                processed_rewards.append(rewards[idx])
                processed_actions.append(actions[idx])
                processed_last_actions.append(actions[idx - 1])
                processed_dones.append(dones[idx])
                processed_timesteps.append(counter)
                processed_reward_to_go.append(sum(rewards[idx: done_index]))

                # current state -----------------------------------------

                # unpackage the values
                related_states = states[idx - sequence_length: idx + 1]  
                related_bgs, related_meals, related_insulins, _ = zip(*related_states)

                # extract the correct metrics
                extracted_bg = related_bgs[::10]
                meals_on_board = [np.sum(np.array(related_meals) * decay_state)]
                insulin_on_board = [np.sum(np.array(related_insulins) * decay_state)]

                # append the state
                processed_state = list(extracted_bg) + meals_on_board + insulin_on_board
                processed_states.append(processed_state)   

                # next state -----------------------------------------

                # unpackage the values
                related_next_states = states[(idx - sequence_length) + 1: idx + 1 + 1]  
                related_next_bgs, related_next_meals, related_next_insulins, _ = zip(*related_next_states)

                # extract the correct metrics
                extracted_next_bg = related_next_bgs[::10]
                next_meals_on_board = [np.sum(np.array(related_next_meals) * decay_state)]
                next_insulin_on_board = [np.sum(np.array(related_next_insulins) * decay_state)]  

                # append the state
                processed_next_state = list(extracted_next_bg) + next_meals_on_board + next_insulin_on_board
                processed_next_states.append(processed_next_state) 

            # update the counter
            counter += 1
            if dones[idx]:
                counter = 0   


    # Create a sequence -------------------------------------------------

    elif data_processing == "sequence": 
        
        for idx, state in enumerate(states):
            
            # find the next done index            
            if idx == 0 or dones[max(idx - 1, 0)]:
                done_index = idx + dones[idx:].index(True) 

            # if there are 80 states previously
            if counter >= (sequence_length) and idx + 1 != len(states):                                           

                # add rewards, actions and dones
                processed_rewards.append(rewards[idx - sequence_length:idx])
                processed_actions.append(actions[idx - sequence_length:idx])
                processed_last_actions.append(actions[(idx - sequence_length) - 1:(idx - 1)])
                processed_dones.append(dones[idx - sequence_length:idx])
                processed_timesteps.append(list(range(counter - sequence_length, counter)))
                
                # get the reward_to_go
                rewards_to_go = [sum(rewards[(idx + 1) : done_index])]
                for i in range(sequence_length - 1):
                    rewards_to_go.append(rewards_to_go[-1] + rewards[idx -  i])                     
                processed_reward_to_go.append(rewards_to_go[::-1])                
                
                # add the state and next_state
                extracted_states = [state[:3] for state in states[idx - sequence_length:idx]]
                processed_states.append(extracted_states)
                processed_next_states.append(extracted_states[1:] + [[0, 0, 0]])                

            # update the counter
            counter += 1
            if dones[idx]:
                counter = 0  

    # Normalisation ------------------------------------------------------
    array_states = np.array(processed_states)
    array_actions = np.array(processed_actions)   
    array_rewards = np.array(processed_rewards)

    if data_processing == "condensed":

        # ensure the state mean and std are consistent across blood glucose
        state_mean, state_std = np.mean(array_states, axis=0), np.std(array_states, axis=0)
        action_mean, action_std = np.mean(array_actions, axis=0), np.std(array_actions, axis=0)   
        reward_mean, reward_std = np.mean(array_rewards, axis=0), np.std(array_rewards, axis=0) 
        state_mean[:-2], state_std[:-2]  = state_mean[0], state_std[0]    

    elif data_processing == "sequence":

        # reshape array and calculate mean and std
        state_size, action_size = array_states.shape[2], array_actions.shape[2] 
        state_mean = np.mean(array_states.reshape(-1, state_size), axis=0)
        state_std = np.std(array_states.reshape(-1, state_size), axis=0)
        action_mean = np.mean(array_actions.reshape(-1, action_size), axis=0)
        action_std = np.std(array_actions.reshape(-1, action_size), axis=0)      
        reward_mean = np.mean(array_rewards.reshape(-1, 1), axis=0)
        reward_std = np.std(array_rewards.reshape(-1, 1), axis=0) 

    # load in new replay ----------------------------------------------------
    
    # TODO: do hidden_in and hidden_out need to be explicitly added
           
    for idx, state in enumerate(processed_states):
        empty_replay.append((state, processed_actions[idx], processed_rewards[idx], processed_next_states[idx],
                             processed_dones[idx], processed_timesteps[idx], processed_reward_to_go[idx], 
                             processed_actions[idx], None, None
                            ))

    full_replay = empty_replay

    return full_replay, state_mean, state_std, action_mean, action_std, reward_mean, reward_std

"""
Extracts a batch of data from the full replay and puts it in an appropriate form
"""    
def get_batch(replay, batch_size, data_processing="condensed", sequence_length=80, device='cpu', online=True, params=None):
    
    # Environment
    state_size = params.get("state_size")  
    state_mean = params.get("state_mean")  
    state_std = params.get("state_std")  
    action_mean = params.get("action_mean")  
    action_std = params.get("action_std")  
    reward_mean = params.get("reward_mean")  
    reward_std = params.get("reward_std")  
    reward_scale = params.get("reward_scale", 1.0)
    
    # sample a minibatch
    minibatch = random.sample(replay, batch_size)
    
    if data_processing == "condensed":
        state = np.zeros((batch_size, state_size), dtype=np.float32)
        action = np.zeros(batch_size, dtype=np.float32)        
        reward = np.zeros(batch_size, dtype=np.float32)
        next_state = np.zeros((batch_size, state_size), dtype=np.float32)
        done = np.zeros(batch_size, dtype=np.uint8)
        timestep = np.zeros(batch_size, dtype=np.float32)
        reward_to_go = np.zeros(batch_size, dtype=np.float32)
        
        last_action = np.zeros(batch_size, dtype=np.float32) 
        hidden_in = [0] * batch_size
        hidden_out = [0] * batch_size
                
    elif data_processing == "sequence": 
        
        state = np.zeros((batch_size, sequence_length, state_size), dtype=np.float32)
        action = np.zeros((batch_size, sequence_length), dtype=np.float32)        
        reward = np.zeros((batch_size, sequence_length), dtype=np.float32)
        next_state = np.zeros((batch_size, sequence_length, state_size), dtype=np.float32)
        done = np.zeros((batch_size, sequence_length), dtype=np.uint8)   
        timestep = np.zeros((batch_size, sequence_length), dtype=np.float32)
        reward_to_go = np.zeros((batch_size, sequence_length), dtype=np.float32)        
        last_action = np.zeros((batch_size, sequence_length), dtype=np.float32)    
        hidden_in = [0] * batch_size
        hidden_out = [0] * batch_size
    
    # unpack the batch
    for i in range(len(minibatch)):
        state[i], action[i], reward[i], next_state[i], done[i], timestep[i], reward_to_go[i], last_action[i], hidden_in[i], hidden_out[i] = minibatch[i]      
    # convert to torch
    state = torch.FloatTensor((state - state_mean) / state_std).to(device)
    action = torch.FloatTensor((action - action_mean) / action_std).to(device)
    last_action = torch.FloatTensor((last_action - action_mean) / action_std).to(device)    
    next_state = torch.FloatTensor((next_state - state_mean) / state_std).to(device)
    done = torch.FloatTensor(1 - done).to(device)
    reward_to_go = torch.FloatTensor(reward_to_go / reward_scale).to(device)
    timestep = torch.tensor(timestep, dtype=torch.int32).to(device)
    
    # get norm of reward
    if reward_mean: reward = torch.FloatTensor(reward_scale * (reward - reward_mean) / reward_std).to(device)
    else: reward = torch.FloatTensor(reward).to(device)
    
    if hidden_in[0] is not None and online:
        
        # process lstm layers
        if len(hidden_in[0]) > 1:
            layer_in, cell_in = list(zip(*hidden_in))
            layer_out, cell_out = list(zip(*hidden_out))
            layer_in, cell_in = torch.cat(layer_in, 1).to(device).detach(), torch.cat(cell_in, 1).to(device).detach()
            layer_out, cell_out = torch.cat(layer_out, 1).to(device).detach(), torch.cat(cell_out, 1).to(device).detach()
            hidden_in, hidden_out = (layer_in, cell_in), (layer_out, cell_out)
        
        # process gru layers
        else:
            layer_in = torch.cat(hidden_in, 1).to(device).detach()
            layer_out = torch.cat(hidden_out, 1).to(device).detach()        
            hidden_in, hidden_out = layer_in, layer_out
                
    # Modify Dimensions
    action = action.unsqueeze(-1)
    last_action = last_action.unsqueeze(-1)
    reward = reward.unsqueeze(-1)
    reward_to_go = reward_to_go.unsqueeze(-1)
    done =  done.unsqueeze(-1)
    
    return state, action, reward, next_state, done, timestep, reward_to_go, last_action, hidden_in, hidden_out
            