#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:35:53 2022

"""

import numpy as np 
import copy, random, torch, gym
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

from utils import get_batch, test_algorithm, create_graph, calculate_bolus, calculate_risk

"""
Recurrent neural network for the Actor.
"""
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, min_log_std=-20, max_log_std=2):
        super(Actor, self).__init__()
        hidden_dim = hidden_dim        
        
        # linear branch
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # lstm branch
        self.lstm1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)  
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state, last_action, hidden_in):
        
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        
        # branch 1
        x1 = F.relu(self.fc1(state))  
        
        # branch 2
        x2 = torch.cat([state, last_action], -1)
        x2 = F.relu(self.lstm1(x2))        
        x2, hidden_out = self.lstm2(x2, hidden_in) 
        
        # merging
        x = torch.cat([x1, x2], -1) 
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x))
        x = x.permute(1, 0, 2)
        
        # mean and std
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        
        return mu, log_std_head, hidden_out

"""
Recurrent neural network for the Critic.
"""
class Q(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Q, self).__init__()
        hidden_dim = hidden_dim 
        
        self.state_dim, self.action_dim = state_dim, action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, last_action, hidden_in):
        
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        
        # branch 1
        x1 = torch.cat((state, action), -1)         
        x1 = F.relu(self.fc1(x1))
        
        # branch 2
        x2 = torch.cat((state, last_action), -1)
        x2 = F.relu(self.fc2(x2))
        x2, hidden_out = self.lstm1(x2, hidden_in)
                
        # merged
        x = torch.cat([x1, x2], -1)         
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.permute(1,0,2)
        
        return x, hidden_out
    

class sac_rnn(object):
    
    def __init__(self, init_seed, patient_params, params):

        # ENVIRONMENT
        self.params = params
        self.env_name = patient_params["env_name"]
        self.folder_name = patient_params["folder_name"]  
        self.bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
        self.env = gym.make(self.env_name)
        self.action_size, self.state_size = 1, 3
        self.params["state_size"] = self.state_size
        self.sequence_length = 80  
        self.data_processing = "sequence" 
        self.device = params["device"]    
        
        # HYPERPARAMETERS
        self.tau = 0.01
        self.gamma = 0.99
        self.ac_learning_rate = 3e-4
        self.ct_learning_rate = 3e-4
        self.ap_learning_rate = 3e-4
        self.batch_size = 3
        self.target_entropy = -1.0
        self.starting_timesteps = 80 * (self.batch_size + 1) # 4801
        self.entropy = True  
        self.hidden_dim = 128
                
        # DISPLAY
        self.training_timesteps = params["training_timesteps"]
        self.training_progress_freq = int(self.training_timesteps // 10)
        self.max_timesteps = 480 * 10    
        
        # SEEDING        
        self.train_seed = init_seed        
        self.env.seed(self.train_seed) 
        np.random.seed(self.train_seed)
        torch.manual_seed(self.train_seed)  
        random.seed(self.train_seed) 
        
        # MEMORY
        self.memory_size = self.training_timesteps 
        self.memory = deque(maxlen=self.memory_size)
        
        # NORMALISATION         
        self.state_mean = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        self.state_std = np.array([990.0, 35, 0.5], dtype=np.float32)        
        self.action_mean, self.action_std  = np.ones(1) * patient_params["max_dose"] * self.bas, np.ones(1) * patient_params["max_dose"] * self.bas
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std        
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std  
        self.unnormed_max_action = self.action_mean * 2
        self.action_range = 1
    
    """
    Initalise the neural networks.
    """
    def init_model(self):
        
        # policy network
        self.policy_net = Actor(self.state_size, self.action_size, self.hidden_dim).to(self.device)        
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.ac_learning_rate)
        
        # Q network
        self.target_soft_q_net1 = Q(self.state_size, self.action_size, self.hidden_dim).to(self.device)
        self.soft_q_net1 = Q(self.state_size, self.action_size, self.hidden_dim).to(self.device)        
        self.soft_q_net2 = Q(self.state_size, self.action_size, self.hidden_dim).to(self.device)
        self.target_soft_q_net2 = Q(self.state_size, self.action_size, self.hidden_dim).to(self.device)
                  
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)
        
        self.soft_q_optimizer1 = torch.optim.Adam(self.soft_q_net1.parameters(), lr=self.ct_learning_rate) 
        self.soft_q_optimizer2 = torch.optim.Adam(self.soft_q_net2.parameters(), lr=self.ct_learning_rate) 
        
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.ap_learning_rate)
    
    """
    Save the learned models.
    """
    def save_model(self):  
        
        torch.save(self.policy_net.state_dict(), './Models/'+ str(self.env_name) + "_" + str(self.train_seed) + "_" +'SAC_RNN_online_weights_actor')
        torch.save(self.soft_q_net1.state_dict(), './Models/' + str(self.env_name) + "_" + str(self.train_seed) + "_" +'SAC_RNN_online_weights_q1')
        torch.save(self.soft_q_net2.state_dict(), './Models/'+ str(self.env_name) + "_" + str(self.train_seed) + "_" + 'SAC_RNN_online_weights_q2')
    
    """
    Load pre-trained weights for testing.
    """
    def load_model(self, name):
        
        # load actor
        self.policy_net.load_state_dict(torch.load(name + '_actor'))
        self.policy_net.eval()   
        
        # load q1
        self.soft_q_net1.load_state_dict(torch.load(name + '_q1'))
        self.soft_q_net1_target = copy.deepcopy(self.soft_q_net1)
        self.soft_q_net1.eval()      
         
        # load q2
        self.soft_q_net2.load_state_dict(torch.load(name + '_q2'))
        self.soft_q_net2_target = copy.deepcopy(self.soft_q_net2)
        self.soft_q_net2.eval()     
    
    """
    Determine the action based on the state.
    """
    def select_action(self, state, last_action, hidden_in, timestep, prev_reward, deterministic=True):
        
        with torch.no_grad():        
            state = torch.FloatTensor(state[:, -1].reshape(1, 1, -1)).to(self.device)
            last_action = torch.FloatTensor(last_action[:, -1].reshape(1, 1, -1)).to(self.device)
            mean, log_std, hidden_out = self.policy_net(state, last_action, hidden_in)
            std = log_std.exp()
            
            normal = Normal(0, 1)
            z = normal.sample(mean.shape).to(self.device)
            action = self.action_range * torch.tanh(mean + std * z)
            action = self.action_range * torch.tanh(mean).detach() if deterministic else action
            
        return action[0][0].detach().cpu().numpy(), hidden_out
    
    """
    Get the action, log proabilities, ect. from a from state.    
    """
    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        
        mean, log_std, hidden_out = self.policy_net(state, last_action, hidden_in)
        std = log_std.exp() 
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(self.device)) 
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(self.device)) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out 
    
    """
    Train the model on a pre-collected sample of training data.
    """    
    def train_model(self):
        
        # initialise the environment and set max timesteps
        env = gym.make(self.env_name) 
        
        env.seed(self.train_seed)
        total_timesteps = 0
        
        # initialise the model
        self.init_model()
        
        while total_timesteps < self.training_timesteps:
            
            # Reset all the parameters ----------------------------------------------------------
            total_rewards = 0

            # get the state
            insulin_dose = 1/3 * self.bas
            meal, done, bg_val = 0, False, env.reset()
            time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479   
            state = np.array([bg_val[0], meal, insulin_dose, time] , dtype=np.float32)

            # get a suitable input
            state_stack = np.tile(state, (self.sequence_length + 1, 1))
            
            # ensure that the time is correct
            state_stack[:, 3] = (state_stack[:, 3] - np.arange(((self.sequence_length + 1) / 479), 0, -(1 / 479))[:self.sequence_length + 1]) * 479            
            state_stack[:, 3] = (np.around(state_stack[:, 3], 0) % 480) / 479  

            # get the action and reward stack
            action_stack = np.tile(np.array([insulin_dose], dtype=np.float32), (self.sequence_length + 1, 1))        
            reward_stack = np.tile(-calculate_risk(bg_val), (self.sequence_length + 1, 1))
            done_stack = np.tile(np.array([False]), (self.sequence_length + 1, 1))

            # get the meal history
            meal_history = np.zeros(int((3 * 60) / 3), dtype=np.float32)
            
            # initialise the hidden layer
            hidden_in = (torch.zeros([1, 1, 128], dtype=torch.float).to(self.device),
                         torch.zeros([1, 1, 128], dtype=torch.float).to(self.device))    
            hidden_layers = [hidden_in]   
            
            # initialise data and time tracking
            timesteps, counter = 0, 0       

            while not done and timesteps < self.max_timesteps:
                                
                # Get the player action ----------------------------------------------------  
                
                state = state_stack[1:, :3].reshape(1, self.sequence_length, 3) 
                prev_action = action_stack[1:, :].reshape(1, self.sequence_length, 1)              

                # Feed state into model
                state = (state - self.state_mean) / self.state_std      
                prev_action = (prev_action - self.action_mean) / self.action_std   
                
                action, hidden_out = self.select_action(state, prev_action, timestep=None, prev_reward=None, hidden_in=hidden_in, deterministic=False)
                output_action = np.maximum(np.minimum(action, np.ones(1)), -np.ones(1))  * self.action_std + self.action_mean
                
                # add the hidden layer
                hidden_layers.append(hidden_out)

                # Unnormalise action output and add gaussian noise                    
                action_pred  = (output_action).clip(0, self.unnormed_max_action)[0]
                player_action = action_pred                
                    
                # Step the environment ----------------------------------------------------

                # update the chosen action
                chosen_action = np.copy(player_action)

                # take meal bolus
                if meal > 0:                             
                    chosen_action += calculate_bolus(
                        bg_val, meal_history, meal, self.params['carbohydrate_ratio'],
                        self.params['correction_factor'], self.params['target_blood_glucose']
                    )

                # append the basal and bolus action
                action_stack = np.delete(action_stack, 0, 0)
                action_stack = np.vstack([action_stack, player_action])

                # step the simulator
                next_bg_val, _, done, info = env.step(chosen_action)
                reward = -calculate_risk(next_bg_val) 

                # get the rnn array format for state
                time = ((env.env.time.hour * 60) / 3 + env.env.time.minute / 3) / 479
                next_state = np.array([float(next_bg_val[0]), float(info['meal']), float(chosen_action), time], dtype=np.float32)   

                # update the state stacks
                next_state_stack = np.delete(state_stack, 0, 0)
                next_state_stack = np.vstack([next_state_stack, next_state]) 
                reward_stack = np.delete(reward_stack, 0, 0)
                reward_stack = np.vstack([reward_stack, np.array([reward], dtype=np.float32)])
                done_stack = np.delete(done_stack, 0, 0)
                done_stack = np.vstack([done_stack, np.array([done], dtype=np.float32)])

                # add a termination penalty
                if done: reward = -1e5

                # update the memory ---------------------------------------------------
                
                counter += 1
                if counter % self.sequence_length == 0 or done or timesteps == self.max_timesteps - 1:
                    
                    # get the states in the correct form
                    state_inp = next_state_stack[:-1, :3].reshape(1, self.sequence_length, 3)
                    next_state_inp  = next_state_stack[1:, :3].reshape(1, self.sequence_length, 3) 
                    reward_inp  = reward_stack[1:, :].reshape(1, self.sequence_length)
                    last_action_inp  = action_stack[:-1, :].reshape(1, self.sequence_length) 
                    action_inp  = action_stack[1:, :].reshape(1, self.sequence_length)
                    done_inp  = done_stack[:-1, :].reshape(1, self.sequence_length) 
                                       
                    # reset the counter and upload the data
                    counter = 0  
                    self.memory.append((
                        state_inp, action_inp, reward_inp, 
                        next_state_inp, done_inp, None, None,
                        last_action_inp,
                        hidden_layers[-self.sequence_length],
                        hidden_layers[-self.sequence_length + 1])
                    )
                    
                # update the states ---------------------------------------------------
                    
                # update the meal history
                meal_history = np.append(meal_history, meal)
                meal_history = np.delete(meal_history, 0)   

                # update the state stacks
                state_stack = next_state_stack

                # update the state
                bg_val = next_bg_val
                state = next_state     
                meal = info['meal']
                timesteps += 1
                total_timesteps += 1
                hidden_in = hidden_out
                total_rewards += reward
                
                # break the loop if terminated
                if done: break
                
                # Sample a batch of data ------------------------------------------------
                
                if total_timesteps  >= self.starting_timesteps:
                    
                    # unpackage the samples and split
                    state_array, action_array, reward_array, next_state_array, done_array, _, _, last_action_array, hidden_in_array, hidden_out_array = get_batch(
                        replay=self.memory, batch_size=self.batch_size, 
                        data_processing=self.data_processing, 
                        sequence_length=self.sequence_length, device=self.device, 
                        params=self.params
                    )     

                    # Training ---------------------------------------------------------
                                        
                    reward_array = (reward_array - torch.mean(reward_array).to(self.device)) / (torch.std(reward_array) + 1e-6).to(self.device)
                    
                    # get q values
                    predicted_q_value1, _ = self.soft_q_net1(state_array, action_array, last_action_array, hidden_in_array)
                    predicted_q_value2, _ = self.soft_q_net2(state_array, action_array, last_action_array, hidden_in_array)
                    
                    # predict actions
                    new_action, log_prob, z, mean, log_std, _ = self.evaluate(state_array, last_action_array, hidden_in_array)
                    new_next_action, next_log_prob, _, _, _, _ = self.evaluate(next_state_array, action_array, hidden_out_array)
                    
                    
                    if self.entropy:
                        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        self.alpha = self.log_alpha.exp()
                    else:
                        self.alpha = 1.
                        alpha_loss = 0

                    # calculate the q function loss
                    predict_target_q1, _ = self.target_soft_q_net1(next_state_array, new_next_action, action_array, hidden_out_array)
                    predict_target_q2, _ = self.target_soft_q_net2(next_state_array, new_next_action, action_array, hidden_out_array)
                    target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob    
                    target_q_value = reward_array + done_array * self.gamma * target_q_min 
                    
                    q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
                    q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
                    
                    # step the optimisers
                    self.soft_q_optimizer1.zero_grad()
                    q_value_loss1.backward()
                    self.soft_q_optimizer1.step()
                    self.soft_q_optimizer2.zero_grad()
                    q_value_loss2.backward()
                    self.soft_q_optimizer2.step()  
                    
                    # calculate the policy loss
                    predict_q1, _= self.soft_q_net1(state_array, new_action, last_action_array, hidden_in_array)
                    predict_q2, _ = self.soft_q_net2(state_array, new_action, last_action_array, hidden_in_array)
                    predicted_new_q_value = torch.min(predict_q1, predict_q2)
                    policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
                    
                    # step the policy optimiser
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    # update the target networks
                    for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                    for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)   
                        
                    # Testing ----------------------------------------------------------
                            
                    # show the progress
                    if total_timesteps % self.training_progress_freq == 0:
                        
                        # show the updated loss
                        print('Timesteps {} - Actor Loss {} - Q1 Loss {} - Q2 Loss {}'.format(total_timesteps, policy_loss, q_value_loss1, q_value_loss2))
                        self.save_model()
                        
            print('Episode score {} - Episode Timesteps {}'.format(total_rewards, timesteps))
            
    """
    Test the learned weights against the PID controller.
    """   
    def test_model(self, input_seed=0, input_max_timesteps=4800):  
        
        # initialise the environment
        env = gym.make(self.env_name)        
            
        # initialise the model
        self.init_model()
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'SAC_RNN_online_weights')
        test_seed, max_timesteps = input_seed, input_max_timesteps
        
        # test the algorithm's performance vs pid algorithm
        rl_reward, rl_bg, rl_action, rl_insulin, rl_meals, pid_reward, pid_bg, pid_action = test_algorithm(
            env=env, agent_action=self.select_action, seed=test_seed, max_timesteps=max_timesteps,
            sequence_length=self.sequence_length, data_processing=self.data_processing, 
            pid_run=False, lstm=True,  params=self.params
        )
         
        # display the results
        create_graph(
            rl_reward=rl_reward, rl_blood_glucose=rl_bg, rl_action=rl_action, rl_insulin=rl_insulin,
            rl_meals=rl_meals, pid_reward=pid_reward, pid_blood_glucose=pid_bg, 
            pid_action=pid_action, params=self.params
        )        
