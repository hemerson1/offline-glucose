#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:35:30 2022

"""

import numpy as np 
import copy, random, torch, gym, pickle
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from utils import unpackage_replay, get_batch, test_algorithm, create_graph


"""
Simple feedforward neural network for the Actor.
"""
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


"""
Simple feedforward neural network for the Critic.
"""
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1        
        
        
class td3_bc: 
    
    def __init__(self, init_seed, patient_params, params):           

        # ENVIRONMENT
        self.params = params
        self.env_name = patient_params["env_name"]       
        self.folder_name = patient_params["folder_name"]   
        self.replay_name = patient_params["replay_name"]  
        self.bas = patient_params["u2ss"] * (patient_params["BW"] / 6000) * 3
        self.env = gym.make(self.env_name)
        self.action_size, self.state_size = 1, 11
        self.params["state_size"] = self.state_size
        self.sequence_length = 80   
        self.data_processing = "condensed"
        
        # HYPERPARAMETERS
        self.device = params["device"]
        self.batch_size = 256
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.alpha = 2.5
        
        # DISPLAY
        self.pid_bg, self.pid_insulin, self.pid_action, self.pid_reward  = [], [], [], 0
        self.training_timesteps = params["training_timesteps"]
        self.training_progress_freq = int(self.training_timesteps // 10)
        
        # SEEDING
        self.train_seed = init_seed
        self.env.seed(self.train_seed) 
        np.random.seed(self.train_seed)
        torch.manual_seed(self.train_seed)  
        random.seed(self.train_seed)      
        
        # MEMORY
        self.memory_size = self.training_timesteps 
        self.memory = deque(maxlen=self.memory_size)
        
    """
    Initialise the Actor and the Critic.
    """        
    def init_model(self):
        
        # actor
        self.actor = Actor(self.state_size, self.action_size, self.max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        
        # critic
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    
    """
    Save the learned models.
    """
    def save_model(self):
        
        torch.save(self.actor.state_dict(), './Models/'+ str(self.env_name) + str(self.train_seed) +'TD3_offline_BC_weights_actor' + self.replay_name.split("-")[-1])
        torch.save(self.critic.state_dict(), './Models/'+ str(self.env_name) + str(self.train_seed) +'TD3_offline_BC_weights_critic' + self.replay_name.split("-")[-1])
            
    """
    Load pre-trained weights for testing.
    """
    def load_model(self, name):
        
        # load actor
        self.actor.load_state_dict(torch.load(name + '_actor'))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.eval()   
        
        # load critic
        self.critic.load_state_dict(torch.load(name + '_critic'))
        self.critic_target = copy.deepcopy(self.critic)
        self.critic.eval()   
        
    """
    Determine the action based on the state.
    """        
    def select_action(self, state, action, timestep, prev_reward):
        
        # Feed state into model
        with torch.no_grad():
            tensor_state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            tensor_action = self.actor(tensor_state)
            
        return tensor_action.cpu().data.numpy().flatten()
    
    """
    Train the model on a pre-collected sample of training data.
    """
    def train_model(self):
        
        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file) 
            
        # Process the replay --------------------------------------------------
        
        # unpackage the replay
        self.memory, self.state_mean, self.state_std, self.action_mean, self.action_std, _, _ = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )
        
        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std    
        self.max_action = float(((self.bas * 3.0) - self.action_mean) / self.action_std)
        
        # initialise the networks
        self.init_model()
        
        print('Processing Complete.')             
        
        for t in range(1, self.training_timesteps + 1):
            
            # Get the batch ------------------------------------------------
            
            # unpackage the samples and split
            state, action, reward, next_state, done, _, _, _, _, _ = get_batch(
                replay=self.memory, batch_size=self.batch_size, 
                data_processing=self.data_processing, 
                sequence_length=self.sequence_length, device=self.device, 
                params=self.params
            )

            # Training -----------------------------------------------
            
            with torch.no_grad():
                
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + done * self.gamma * target_Q
                
            # Update the critic -------------------------------------------

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Perform the actor update ---------------------------------------------------

            # Delayed policy updates
            if t % self.policy_freq == 0:

                # Compute the modfied actor loss
                pi = self.actor(state)
                Q = self.critic.Q1(state, pi)
                lmbda = self.alpha / Q.abs().mean().detach()
                actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Show progress
            if t % self.training_progress_freq == 0:
                
                # show the updated loss
                print('Timesteps {} - Actor Loss {} - Critic Loss {}'.format(t, actor_loss, critic_loss))
                self.save_model()
                
    """
    Test the learned weights against the PID controller.
    """
    def test_model(self, input_seed=0, input_max_timesteps=4800): 
        
        # initialise the environment
        env = gym.make(self.env_name)           
            
        # load the replay buffer
        with open("./Replays/" + self.replay_name + ".txt", "rb") as file:
            trajectories = pickle.load(file)  
        
        # Process the replay --------------------------------------------------
        
        # unpackage the replay
        self.memory, self.state_mean, self.state_std, self.action_mean, self.action_std, _, _ = unpackage_replay(
            trajectories=trajectories, empty_replay=self.memory, data_processing=self.data_processing, sequence_length=self.sequence_length
        )

        # update the parameters
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas) 
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        self.max_action = float(((self.bas * 3) - self.action_mean) / self.action_std)
        self.init_model()          
              
        # load the learned model
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'TD3_offline_BC_weights')
        test_seed, max_timesteps = input_seed, input_max_timesteps
            
        # TESTING -------------------------------------------------------------------------------------------
        
        # test the algorithm's performance vs pid algorithm
        rl_reward, rl_bg, rl_action, rl_insulin, rl_meals, pid_reward, pid_bg, pid_action = test_algorithm(
            env=env, agent_action=self.select_action, seed=test_seed, max_timesteps=max_timesteps,
            sequence_length=self.sequence_length, data_processing=self.data_processing, 
            pid_run=False, params=self.params
        )
         
        # display the results
        create_graph(
            rl_reward=rl_reward, rl_blood_glucose=rl_bg, rl_action=rl_action, rl_insulin=rl_insulin,
            rl_meals=rl_meals, pid_reward=pid_reward, pid_blood_glucose=pid_bg, 
            pid_action=pid_action, params=self.params
        )