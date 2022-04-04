#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:36:04 2022

"""

import numpy as np 
import copy, random, torch, gym, pickle
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from utils import unpackage_replay, get_batch, test_algorithm, create_graph


"""
Create a scalar constant
"""
class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant

"""
 Extend and repast the tensor along axis and repeat it
"""
def extend_and_repeat(tensor, dim, repeat):
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)

"""
Forward the q function with multiple actions on each state, to be used as a decorator
"""
def multiple_action_q_function(forward):
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


"""
Fully connected feedforward neural network.
"""
class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256'):
        super().__init__()
        
        # get the parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]
        
        # add linear layers to the network
        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size
        
        # add the output layer
        last_fc = nn.Linear(d, output_dim)
        modules.append(last_fc)
        
        # construct the network
        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)

"""
Fully connected Q function approximator.
"""
class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256'):
        super().__init__()
        
        # get the parameters
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        
        # initialise the network
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1
        )
    
    @multiple_action_q_function
    def forward(self, observations, actions):  
        
        # concatentate the tensors and feed unto network
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)
 
"""
Reparamterised Policy 
"""    
class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        
        # get the parameters
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def log_prob(self, mean, log_std, sample):
        
        # restrict log probability and then calculate exponential
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # construct the action distribution
        action_distribution = torch.distributions.transformed_distribution.TransformedDistribution(
            torch.distributions.Normal(mean, std), torch.distributions.transforms.TanhTransform(cache_size=1)
        )
        
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        
        # restrict log probability and then calculate exponential
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # construct the action distribution
        action_distribution = torch.distributions.transformed_distribution.TransformedDistribution(
            torch.distributions.Normal(mean, std), torch.distributions.transforms.TanhTransform(cache_size=1)
        )
        
        # sample from the action distribution
        if deterministic: action_sample = torch.tanh(mean)
        else: action_sample = action_distribution.rsample()
        
        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob

"""
Tanh Gaussian Policy
"""
class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0):
        
        super().__init__()
        
        # get the parameters
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        
        # initialise the base network
        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch
        )
        
        # initiailse the reparameterized tanh gaussian
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian()

    def log_prob(self, observations, actions):
        
        # change the dimensions of the observation to match the action
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        
        # prepare the parameters
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        
        # get the log probability
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        
        # change the dimensions of the observation to match the action
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        
        # prepare the parameters
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        
        # get the action sample and log prob
        return self.tanh_gaussian(mean, log_std, deterministic)
    
    
class cql:

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
        self.device = params["device"]         
        
        # HYPERPARAMETERS
        self.batch_size = 256
        self.policy_arch = '256-256'
        self.qf_arch = '256-256'
        self.policy_log_std_multiplier = 1.0
        self.policy_log_std_offset = -1.0
        self.discount = 0.99
        self.alpha_multiplier = 1.0
        self.target_entropy = 0.0
        self.policy_lr = 3e-4
        self.qf_lr = 3e-4
        self.soft_target_update_rate = 5e-3
        self.target_update_period = 1
        self.cql_n_actions = 10
        self.cql_temp = 1.0
        self.cql_min_q_weight = 5.0
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf         
            
        # DISPLAY
        self.training_timesteps = params["training_timesteps"]
        self.training_progress_freq = int(self.training_timesteps // 10)
        
        # SEEDING
        self.train_seed = init_seed # use seeds 1, 2, 3        
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
        
        # policy network
        self.policy = TanhGaussianPolicy(self.state_size, self.action_size, arch=self.policy_arch,
                                         log_std_multiplier=self.policy_log_std_multiplier, 
                                         log_std_offset=self.policy_log_std_offset).to(self.device)
        self.log_alpha = Scalar(0.0).to(self.device)
        
        # Q networks and Target networks
        self.qf1 = FullyConnectedQFunction(self.state_size, self.action_size, arch=self.qf_arch).to(self.device)
        self.qf2 = FullyConnectedQFunction(self.state_size, self.action_size, arch=self.qf_arch).to(self.device)
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        
        # Initialise the optimisers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.qf_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.qf_lr)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.policy_lr)           
    
    """
    Save the learned models.
    """
    def save_model(self):
        torch.save(self.policy.state_dict(), './Models/'  + str(self.env_name) + str(self.train_seed) + 'CQL_weights_policy')
        torch.save(self.qf1.state_dict(), './Models/' + str(self.env_name) + str(self.train_seed) + 'CQL_weights_qf1')
        torch.save(self.qf2.state_dict(), './Models/' + str(self.env_name) + str(self.train_seed) + 'CQL_weights_qf2')
    
    """
    Load pre-trained weights for testing.
    """
    def load_model(self, name):
        
        # load he policy
        self.policy.load_state_dict(torch.load(name + '_policy'))
        self.policy.eval()   
        
        # load qf1 and target
        self.qf1.load_state_dict(torch.load(name + '_qf1'))
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.qf1.eval()   
        
        # load qf2 and target
        self.qf2.load_state_dict(torch.load(name + '_qf2'))
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.qf2.eval()            
    
    """
    Determine the action based on the state.
    """ 
    def select_action(self, state, action, timestep, prev_reward):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action, _ = self.policy(state, deterministic=True)
        
        return action.cpu().data.numpy().flatten()
    
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

            # Perform the training update -----------------------------------------

            #  get the action predictions
            new_actions, log_pi = self.policy(state)

            # update the alpha loss
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier

            # Compute the policy loss --------------------------------
            q_new_actions = torch.min(
                self.qf1(state, new_actions),
                self.qf2(state, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()

            # Compute the Q function loss --------------------------------
            q1_pred = self.qf1(state, action)
            q2_pred = self.qf2(state, action)

            new_next_actions, next_log_pi = self.policy(next_state)
            target_q_values = torch.min(
                self.target_qf1(next_state, new_next_actions),
                self.target_qf2(next_state, new_next_actions),
            )

            td_target = reward.reshape(-1) + (1. - done).reshape(-1) * self.discount * target_q_values            
            qf1_loss = F.mse_loss(q1_pred, td_target.detach())
            qf2_loss = F.mse_loss(q2_pred, td_target.detach())

            # CQL -> incorporate conservativism into Q function loss --------------------------------

            # create an array of unitiialised values of size below between -1 and 1
            cql_random_actions = action.new_empty((self.batch_size, self.cql_n_actions, self.action_size), requires_grad=False).uniform_(-1, 1)

            # get the current policy predictions
            cql_current_actions, cql_current_log_pis = self.policy(state, repeat=self.cql_n_actions)
            cql_next_actions, cql_next_log_pis = self.policy(next_state, repeat=self.cql_n_actions)

            # detach the values from the graph
            cql_current_actions, cql_current_log_pis = cql_current_actions.detach(), cql_current_log_pis.detach()
            cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()          

            # get the network predictions for random, current and next actions
            cql_q1_rand = self.qf1(state, cql_random_actions)
            cql_q2_rand = self.qf2(state, cql_random_actions)
            cql_q1_current_actions = self.qf1(state, cql_current_actions)
            cql_q2_current_actions = self.qf2(state, cql_current_actions)
            cql_q1_next_actions = self.qf1(state, cql_next_actions)
            cql_q2_next_actions = self.qf2(state, cql_next_actions)

            # concatenate the results and calculate the standard deviation
            cql_cat_q1 = torch.cat([cql_q1_rand, torch.unsqueeze(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], dim=1)
            cql_cat_q2 = torch.cat([cql_q2_rand, torch.unsqueeze(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], dim=1)
            cql_std_q1 = torch.std(cql_cat_q1, dim=1)
            cql_std_q2 = torch.std(cql_cat_q2, dim=1)

            # Subtract density function from the Q function predictions
            random_density = np.log(0.5 ** self.action_size)
            cql_cat_q1 = torch.cat(
                [cql_q1_rand - random_density,
                 cql_q1_next_actions - cql_next_log_pis.detach(),
                 cql_q1_current_actions - cql_current_log_pis.detach()],
                dim=1
            )
            cql_cat_q2 = torch.cat(
                [cql_q2_rand - random_density,
                 cql_q2_next_actions - cql_next_log_pis.detach(),
                 cql_q2_current_actions - cql_current_log_pis.detach()],
                dim=1
            )

            # Check if the predictions are out of the distribution (OOD)
            cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
            cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

            # Subtract the log likelihood of data
            cql_qf1_diff = torch.clamp(cql_qf1_ood - q1_pred, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
            cql_qf2_diff = torch.clamp(cql_qf2_ood - q2_pred, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

            # calculate the conservative loss
            cql_min_qf1_loss = cql_qf1_diff * self.cql_min_q_weight
            cql_min_qf2_loss = cql_qf2_diff * self.cql_min_q_weight

            # Returns a new tensor of the dimensions of the state
            alpha_prime_loss = state.new_tensor(0.0).to(self.device)
            alpha_prime = state.new_tensor(0.0).to(self.device)

            # get the combined conservative loss
            qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss
            
            # Backpropagation --------------------------------------------------

            # backpropagate and gradient step
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()
            
            # Target Update -----------------------------------------------------

            # update the target networks
            if t % self.target_update_period == 0:

                for param, target_param in zip(self.qf1.parameters(), self.target_qf1.parameters()):
                    target_param.data.copy_(self.soft_target_update_rate * param.data + (1 - self.soft_target_update_rate) * target_param.data)

                for param, target_param in zip(self.qf2.parameters(), self.qf2.parameters()):
                    target_param.data.copy_(self.soft_target_update_rate * param.data + (1 - self.soft_target_update_rate) * target_param.data)                    
            
            # Show progress
            if t % self.training_progress_freq == 0:
                
                # show the updated loss
                print('Timesteps {} - Policy Loss {} - Q function Loss {}'.format(t, policy_loss, qf_loss))
                self.save_model()
                
    
    """
    Test the learned weights against the PID controller.
    """
    def test_model(self, input_seed=0, input_max_timesteps=4800):        
            
        # TESTING -------------------------------------------------------------------------------------------- 
        
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
        self.init_model()            
        
        # load the learned model
        self.load_model('./Models/' + self.folder_name + "/" + "Seed" + str(self.train_seed) + "/" + 'CQL_weights')
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