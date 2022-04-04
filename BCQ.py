#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:36:19 2022

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
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

"""
Simple feedforward neural network for the Critic.
"""
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

"""
Vanilla Variational Auto-Encoder 
"""
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a)) 
     
        
class bcq: 
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
        self.latent_size = self.action_size * 2
        self.sequence_length = 80   
        self.data_processing = "condensed" 
        self.device = params["device"]    
        
        # HYPERPARAMETERS
        self.batch_size = 100
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.gamma = 0.99  
        self.tau = 0.005
        self.phi = 0.05
        self.lmbda = 0.75 
        
        # DISPLAY
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
    Initalise the neural networks.
    """
    def init_model(self):
        
        # Actor
        self.actor = Actor(self.state_size, self.action_size, self.max_action, self.phi).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critic
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # VAE
        self.vae = VAE(self.state_size, self.action_size, self.latent_size, self.max_action, self.device).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 
        
    """
    Save the learned models.
    """
    def save_model(self):
        torch.save(self.actor.state_dict(), './Models/' + str(self.env_name) + str(self.train_seed) + 'BCQ_weights_actor')
        torch.save(self.critic.state_dict(), './Models/' + str(self.env_name) + str(self.train_seed) +  'BCQ_weights_critic')
        torch.save(self.vae.state_dict(), './Models/' + str(self.env_name) + str(self.train_seed) + 'BCQ_weights_vae')
    
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
        
        # load vae
        self.vae.load_state_dict(torch.load(name + '_vae'))
        self.vae_target = copy.deepcopy(self.vae)
        self.vae.eval()        
    
    """
    Determine the action based on the state.
    """
    def select_action(self, state, action, timestep, prev_reward):
        
        # Feed state into model
        with torch.no_grad():
            tensor_state = torch.FloatTensor(state.reshape(1, -1)).repeat(self.batch_size, 1).to(self.device)                       
            tensor_action = self.actor(tensor_state, self.vae.decode(tensor_state))                        
            q1 = self.critic.q1(tensor_state, tensor_action)
            ind = q1.argmax(0)  
            
        return tensor_action[ind].cpu().data.numpy().flatten()
    
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
        self.max_action = float(((self.bas * 3) - self.action_mean) / self.action_std)
        
        # initialise the networks
        self.init_model()
        
        print('Processing Complete.')
        # ------------------------------------------------------------------------        
        
        for t in range(1, self.training_timesteps + 1):
            
            # Get the batch ------------------------------------------------
            
            # unpackage the samples and split
            state, action, reward, next_state, done, _, _, _, _, _ = get_batch(
                replay=self.memory, batch_size=self.batch_size, 
                data_processing=self.data_processing, 
                sequence_length=self.sequence_length, device=self.device, 
                params=self.params
            )

            # Variational Auto-Encoder Training --------------------------------------------------------
            
            recon, mean, std = self.vae(state, action)            
            
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()  
            vae_loss = recon_loss + 0.5 * KL_loss 

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training --------------------------------------------------------
            
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, 0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

                # Soft Clipped Double Q-learning 
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(self.batch_size, -1).max(1)[0].reshape(-1, 1)
                target_Q = reward + done * self.gamma * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Pertubation Model / Action Training --------------------------------------------------------
            
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()
                
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Show progress
            if t % self.training_progress_freq == 0:
                
                # show the updated loss
                print('Timesteps {} - Actor Loss {} - Critic Loss {} - Encoder Loss {}'.format(t, actor_loss, critic_loss, vae_loss))
                self.save_model()
    
    """
    Test the learned weights against the PID controller.
    """
    def test_model(self, training=False, input_seed=0, input_max_timesteps=4800):
        
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

        # adding this allows better results?
        self.action_std = 1.75 * self.bas * 0.25 / (self.action_std / self.bas)
        self.params["state_mean"], self.params["state_std"]  = self.state_mean, self.state_std
        self.params["action_mean"], self.params["action_std"] = self.action_mean, self.action_std
        self.max_action = float(((self.bas * 3) - self.action_mean) / self.action_std)
        self.init_model()            
        
        # load the learned model
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'BCQ_weights')
        test_seed, max_timesteps = input_seed, input_max_timesteps
        
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


