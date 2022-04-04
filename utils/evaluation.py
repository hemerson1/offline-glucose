#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:28:19 2022
"""

"""
Functions for evaluating algorithmic performance and displaying it to the user.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch, random

from .general import PID_action, calculate_bolus, calculate_risk, is_in_range


"""
Test the learned policy of an agent against the PID algorithm over a 
specified length of time.
"""
def test_algorithm(env, agent_action, seed=0, max_timesteps=480, sequence_length=80,
                   data_processing="condensed", pid_run=False, lstm=False, params=None):
    
    # Unpack the params
    
    # Diabetes
    basal_default = params.get("basal_default") 
    target_blood_glucose = params.get("target_blood_glucose", 144)
    
    # Bolus
    cr = params.get("carbohydrate_ratio")
    cf = params.get("correction_factor")
    bolus_overestimate = params.get("bolus_overestimate", 0.0)
    meal_announce = params.get("meal_announce", 0.0)
    
    # PID
    kp = params.get("kp")
    ki = params.get("ki")
    kd = params.get("kd")
    
    # Means and Stds
    state_mean = params.get("state_mean")  
    state_std = params.get("state_std")  
    action_mean = params.get("action_mean")  
    action_std = params.get("action_std")
    
    # Device
    device = params.get("device")
    
    # Network
    model_dim = params.get("model_dim", 256)
    
    
    # initialise the arrays for data collection    
    rl_reward, rl_blood_glucose, rl_action = 0, [], []
    pid_reward, pid_blood_glucose, pid_action = 0, [], []
    rl_insulin, rl_meals = [], []

    # select the number of iterations    
    if not pid_run: runs = 2
    else: runs = 1
    
    for ep in range(runs):
        
        # set the seed for the environment
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        random.seed(seed)    
        
        # Initialise the environment --------------------------------------
        
        # get the state
        insulin_dose = 1/3 * basal_default
        meal, done, bg_val = 0, False, env.reset()
        time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
        state = np.array([bg_val[0], meal, insulin_dose, time], dtype=np.float32)
        last_action = insulin_dose

        # get a suitable input
        state_stack = np.tile(state, (sequence_length + 1, 1))
        state_stack[:, 3] = (state_stack[:, 3] - np.arange(((sequence_length + 1) / 479), 0, -(1 / 479))[:sequence_length + 1]) * 479            
        state_stack[:, 3] = (np.around(state_stack[:, 3], 0) % 480) / 479         

        # get the action and reward stack
        action_stack = np.tile(np.array([insulin_dose], dtype=np.float32), (sequence_length + 1, 1))        
        reward_stack = np.tile(-calculate_risk(bg_val), (sequence_length + 1, 1))

        # get the meal history
        meal_history = np.zeros(int((3 * 60) / 3), dtype=np.float32)
        
        # intiialise pid parameters
        integrated_state = 0
        previous_error = 0
        timesteps = 0 
        reward = 0
        
        # init the hidden_layer
        if params["rnn"] == "gru":
            hidden_in = torch.zeros([1, 1, model_dim], dtype=torch.float).to(device) 
            
        else:                    
            hidden_in = (torch.zeros([1, 1, model_dim], dtype=torch.float).to(device),
                             torch.zeros([1, 1, model_dim], dtype=torch.float).to(device)) 
        
        while not done and timesteps < max_timesteps:
            
            # Run the RL algorithm ------------------------------------------------------
            if ep == 0:
                
                # condense the state
                if data_processing == "condensed":
                                    
                    # Unpack the state
                    bg_vals, meal_vals, insulin_vals = state_stack[:, 0][::10], state_stack[:, 1], state_stack[:, 2]
                    
                    # calculate insulin and meals on board
                    decay_factor = np.arange(1 / (sequence_length + 2), 1, 1 / (sequence_length + 2))
                    meals_on_board, insulin_on_board = np.sum(meal_vals * decay_factor), np.sum(insulin_vals * decay_factor) 
                    
                    # create the state
                    state = np.concatenate([bg_vals, meals_on_board.reshape(1), insulin_on_board.reshape(1)])  
                    prev_action = last_action
                    
                # TOOD: replace with explicity state and action size
                
                # get the state a sequence of specified length
                elif data_processing == "sequence":
                    state = state_stack[1:, :3].reshape(1, sequence_length, 3) 
                    prev_action = action_stack[1:, :].reshape(1, sequence_length)
                                            
                # Normalise the current state
                state = (state - state_mean) / state_std
                prev_action = (prev_action - action_mean) / action_std
                
                # get the action prediction from the model
                if lstm:
                    action, hidden_in = agent_action(state, prev_action, timestep=timesteps, hidden_in=hidden_in, prev_reward=reward)                    
                else:
                    action = agent_action(state, prev_action, timestep=timesteps, prev_reward=reward)                    
                                        
                # Unnormalise action output  
                action_pred = (action * action_std + action_mean)[0]
                
                # to stop subtracting from bolus when -ve
                action_pred = max(0, action_pred)
                player_action = action_pred
                

            # Run the pid algorithm ------------------------------------------------------
            else:                            
                player_action, previous_error, integrated_state = PID_action(
                    blood_glucose=bg_val, previous_error=previous_error, 
                    integrated_state=integrated_state, 
                    target_blood_glucose=target_blood_glucose, 
                    kp=kp, ki=ki, kd=kd, basal_default=basal_default
                    )
                
    
            # update the chosen action
            chosen_action = np.copy(player_action)
            
            # Get the meal bolus --------------------------------------------

            # take meal bolus
            if meal > 0:   
                
                # add a bias to the bolus estimation
                adjusted_meal = meal
                adjusted_meal += bolus_overestimate * meal 

                bolus_action = calculate_bolus(
                    blood_glucose=bg_val, meal_history=meal_history,
                    current_meal=adjusted_meal, carbohyrdate_ratio=cr, 
                    correction_factor=cf, 
                    target_blood_glucose=target_blood_glucose
                    ) 
                                       
                chosen_action = float(chosen_action) + bolus_action
                
            # Step the environment ------------------------------------------                

            # append the basal and bolus action
            action_stack = np.delete(action_stack, 0, 0)
            action_stack = np.vstack([action_stack, player_action])

            # step the simulator
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

            # get the rnn array format for state
            time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
            next_state = np.array([float(next_bg_val[0]), float(meal_input), float(chosen_action), time], dtype=np.float32)   

            # update the state stacks
            next_state_stack = np.delete(state_stack, 0, 0)
            next_state_stack = np.vstack([next_state_stack, next_state]) 
            reward_stack = np.delete(reward_stack, 0, 0)
            reward_stack = np.vstack([reward_stack, np.array([reward], dtype=np.float32)])

            # add a termination penalty
            if done: 
                reward = -1e5
                break
            
            # Save the testing results --------------------------------------

            # for RL agent
            if ep == 0:
                rl_blood_glucose.append(next_bg_val[0])
                rl_action.append(player_action)
                rl_insulin.append(chosen_action)
                rl_reward += reward
                rl_meals.append(info['meal'])

            # for pid agent
            else:
                pid_blood_glucose.append(next_bg_val[0])
                pid_action.append(player_action)
                pid_reward += reward
                                
            # Update the state ---------------------------------------------

            # update the meal history
            meal_history = np.append(meal_history, meal)
            meal_history = np.delete(meal_history, 0)   

            # update the state stacks
            state_stack = next_state_stack

            # update the state
            bg_val, state, meal = next_bg_val, next_state, info['meal']
            last_action = player_action
            timesteps += 1 
            
    return rl_reward, rl_blood_glucose, rl_action, rl_insulin, rl_meals, pid_reward, pid_blood_glucose, pid_action


"""
Plot a four-tiered graph comparing the blood glucose control of a PID 
and RL algorithm, showing the blood glucose, insulin doses and meal 
carbohyrdates.
"""
def create_graph(rl_reward, rl_blood_glucose, rl_action, rl_insulin, rl_meals,
                 pid_reward, pid_blood_glucose, pid_action, params):
    
    # Unpack the params
    
    # Diabetes
    basal_default = params.get("basal_default")    
    hyper_threshold = params.get("hyper_threshold", 180) 
    sig_hyper_threshold = params.get("sig_hyper_threshold", 250)
    hypo_threshold = params.get("hypo_threshold ", 70)
    sig_hypo_threshold = params.get("sig_hypo_threshold ", 54)
    
    # Display the evaluation metrics
    
    # TIR Metrics ----------------------------------------------
    
    # PID algorithm
    pid_in_range, pid_above_range, pid_below_range, pid_total = 0, 0, 0, len(pid_blood_glucose)
    pid_sig_above_range, pid_sig_below_range = 0, 0 
    for pid_bg in pid_blood_glucose:
        
        # classify the blood glucose_value
        classification = is_in_range(pid_bg, hypo_threshold, hyper_threshold, sig_hypo_threshold, sig_hyper_threshold)   
        
        # in range
        if classification == 0: 
            pid_in_range += 1
            
        # hyperglycaemia
        elif classification > 0:
            pid_above_range += 1
            if classification > 1:
                pid_sig_above_range += 1            
        
        # hypoglycaemia 
        else: 
            pid_below_range += 1
            if classification > -1:
                pid_sig_below_range += 1  
    
    # RL algorithm
    rl_in_range, rl_above_range, rl_below_range, rl_total = 0, 0, 0, len(rl_blood_glucose)
    rl_sig_above_range, rl_sig_below_range = 0, 0    
    for rl_bg in rl_blood_glucose:
        
        # classify the blood glucose_value        
        classification = is_in_range(rl_bg, hypo_threshold, hyper_threshold, sig_hypo_threshold, sig_hyper_threshold)  
        
        # in range
        if classification == 0: 
            rl_in_range += 1
        
        # hyperglycaemia
        elif classification > 0: 
            rl_above_range += 1
            if classification > 1:
                rl_sig_above_range += 1
                
        # hypoglycaemia    
        else: 
            rl_below_range += 1
            if classification < -1:
                rl_sig_below_range += 1
        
    # Statistical Metrics -----------------------------------------
    
    pid_mean, pid_std = np.mean(pid_blood_glucose), np.std(pid_blood_glucose)
    rl_mean, rl_std = np.mean(rl_blood_glucose), np.std(rl_blood_glucose)
    pid_cv, rl_cv = (pid_std / pid_mean), (rl_std / rl_mean)
    
    # Diabetes Metrics ---------------------------------------------
    
    # get the average hypo/hyper length for PID 
    pid_hypo_length, pid_hyper_length = [], []
    prev_classification, hypo_count, hyper_count = 0, 0, 0
    
    for pid_bg in pid_blood_glucose:
        
        # classify the blood glucose_value
        classification = is_in_range(pid_bg, hypo_threshold, hyper_threshold, sig_hypo_threshold, sig_hyper_threshold)  
        
        # if continued hyper
        if classification > 0:
            
            # add to the hyper count
            if prev_classification > 0:
                hyper_count += 1
                
            # reset the count
            else:
                pid_hyper_length.append(hyper_count * 3)
                hyper_count = 0
                
        # if continued hypo
        if classification < 0:
            
            # add to the hypo count
            if prev_classification < 0:
                hypo_count += 1
                
            # reset the count
            else:
                pid_hypo_length.append(hypo_count * 3)
                hypo_count = 0
                
        prev_classification = classification
        
    # get the average hypo/hyper length for RL
    rl_hypo_length, rl_hyper_length = [], []
    prev_classification, hypo_count, hyper_count = 0, 0, 0
    
    for rl_bg in rl_blood_glucose:
        
        # classify the blood glucose_value
        classification = is_in_range(rl_bg, hypo_threshold, hyper_threshold, sig_hypo_threshold, sig_hyper_threshold)  
        
        # if continued hyper
        if classification > 0:
            
            # add to the hyper count
            if prev_classification > 0:
                hyper_count += 1
                
            # reset the count
            else:
                rl_hyper_length.append(hyper_count * 3)
                hyper_count = 0
                
        # if continued hypo
        if classification < 0:
            
            # add to the hypo count
            if prev_classification < 0:
                hypo_count += 1
                
            # reset the count
            else:
                rl_hypo_length.append(hypo_count * 3)
                hypo_count = 0
                
        prev_classification = classification
        
    mean_pid_hypo_length = sum(pid_hypo_length) / max(len(pid_hypo_length), 1)
    mean_pid_hyper_length = sum(pid_hyper_length) / max(len(pid_hyper_length), 1)
    mean_rl_hypo_length = sum(rl_hypo_length) / max(len(rl_hypo_length), 1)
    mean_rl_hyper_length = sum(rl_hyper_length) / max(len(rl_hyper_length), 1)    
    
    print('\n-----------------------------------------------------------')
    print('                    | {: ^016} | {: ^016} |'.format("PID", "RL"))
    print('-----------------------------------------------------------')
    print('Reward              | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_reward, rl_reward))
    print('TIR (%)             | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_in_range / pid_total * 100, rl_in_range / rl_total * 100))
    print('TAR (%)             | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_above_range / pid_total * 100, rl_above_range / rl_total * 100))
    print('TBR (%)             | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_below_range / pid_total * 100, rl_below_range / rl_total * 100))    
    print('Mean (mg/dl)        | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_mean, rl_mean))
    print('STD (mg/dl)         | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_std, rl_std))
    print('CoV                 | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_cv, rl_cv))
    print('Hyper Length (mins) | {: ^#016.2f} | {: ^#016.2f} |'.format(mean_pid_hyper_length, mean_rl_hyper_length))
    print('Hypo Length (mins)  | {: ^#016.2f} | {: ^#016.2f} |'.format(mean_pid_hypo_length, mean_rl_hypo_length))
    print('TMBR (%)            | {: ^#016.2f} | {: ^#016.2f} |'.format((pid_below_range - pid_sig_below_range) / pid_total * 100, (rl_below_range - rl_sig_below_range) / rl_total * 100))
    print('TSBR (%)            | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_sig_below_range / pid_total * 100, rl_sig_below_range / rl_total * 100)) 
    print('TMAR (%)            | {: ^#016.2f} | {: ^#016.2f} |'.format((pid_above_range - pid_sig_above_range) / pid_total * 100, (rl_above_range - rl_sig_above_range) / rl_total * 100))
    print('TSAR (%)            | {: ^#016.2f} | {: ^#016.2f} |'.format(pid_sig_above_range / pid_total * 100, rl_sig_above_range / rl_total * 100))    
    print('-----------------------------------------------------------')
    
    # Produce the glucose display graph -----------------------------------------------
    
    # Check that the rl algorithm completed the full episode
    if len(pid_blood_glucose) == len(rl_blood_glucose):        
        
        # Plot insulin actions alongside blood glucose ------------------------------
                
        # get the x-axis 
        x = list(range(len(pid_blood_glucose)))
        
        # Initialise the plot and specify the title
        fig = plt.figure(dpi=160)
        gs = fig.add_gridspec(4, hspace=0.0)
        axs = gs.subplots(sharex=True, sharey=False)        
        fig.suptitle('Blood Glucose Control Algorithm Comparison')
        
        # define the hypo, eu and hyper regions
        axs[0].axhspan(180, 500, color='lightcoral', alpha=0.6, lw=0)
        axs[0].axhspan(70, 180, color='#c1efc1', alpha=1.0, lw=0)
        axs[0].axhspan(0, 70, color='lightcoral', alpha=0.6, lw=0)
        
        # plot the blood glucose values
        axs[0].plot(x, pid_blood_glucose, label='pid', color='orange')
        axs[0].plot(x, rl_blood_glucose, label='rl', color='dodgerblue')
        axs[0].legend(bbox_to_anchor=(1.0, 1.0))
        
        # specify the limits and the axis lables
        axs[0].axis(ymin=50, ymax=500)
        axs[0].axis(xmin=0.0, xmax=len(pid_blood_glucose))
        axs[0].set_ylabel("BG \n(mg/dL)")
        axs[0].set_xlabel("Time \n(mins)")
        
        # show the basal doses
        axs[1].plot(x, pid_action, label='pid', color='orange')
        axs[1].plot(x, rl_action, label='rl', color='dodgerblue')
        axs[1].axis(ymin=0.0, ymax=(basal_default * 1.4))
        axs[1].set_ylabel("Basal \n(U/min)")

        # show the bolus doses
        axs[2].plot(x, rl_insulin)
        axs[2].axis(ymin=0.01, ymax=0.99)
        axs[2].set_ylabel("Bolus \n(U/min)")

        # show the scheduled meals
        axs[3].plot(x, rl_meals)
        axs[3].axis(ymin=0, ymax=29.9)
        axs[3].set_ylabel("CHO \n(g/min)")

        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
            
        plt.show()
        
        # Plot the distribution of states ------------------------------
        
        fig2 = plt.figure(dpi=160)
        
        bins = np.linspace(10, 1000, 100)
        
        # plot the bins and the legend
        plt.hist(pid_blood_glucose, bins, alpha=0.5, label='pid', color='orange')
        plt.hist(rl_blood_glucose, bins, alpha=0.5, label='rl', color='dodgerblue')
        plt.legend(loc='upper right')
        
        # mark the target range
        plt.axvline(hyper_threshold, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(hypo_threshold, color='k', linestyle='dashed', linewidth=1)
        
        # set the axis labels
        plt.xlabel("Blood glucose (mg/dl)")
        plt.ylabel("Frequency")
        plt.title("Blood glucose distribution")
        
        plt.show()
    
    # specify the timesteps before termination
    else: print('Terminated after: {} timesteps.'.format(len(rl_blood_glucose)))