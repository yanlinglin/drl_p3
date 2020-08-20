import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR_ACTOR = 1e-4         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 4        # how often to update the network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MAagent():
        def __init__(self, state_size, action_size, random_seed, num_agents, noise_scalar_init=2.0, noise_reduction_factor=0.999, update_every=1, actor_fc1_units=400, actor_fc2_units=300,critic_fcs1_units=400, critic_fc2_units=300,gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,mu=0., theta=0.15, sigma=0.2):
            self.t_step = 0
            self.update_every=update_every

            self.actor_fc1_units=actor_fc1_units
            self.actor_fc2_units=actor_fc2_units
            self.critic_fcs1_units=critic_fcs1_units
            self.critic_fc2_units=critic_fc2_units            
            self.gamma=gamma
            self.tau=tau
            self.lr_actor=lr_actor
            self.lr_critic=lr_critic
            self.weight_decay=weight_decay
            self.mu=mu 
            self.theta=theta 
            self.sigma=sigma

            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(random_seed)
            self.random_seed=random_seed
            self.num_agents=num_agents
            self.noise_scalar_init = noise_scalar_init
            self.noise_reduction_factor = noise_reduction_factor
            
            self.agents = [Agent(state_size=self.state_size, action_size=self.action_size,\
                                 noise_scalar_init=self.noise_scalar_init,\
                                 noise_reduction_factor=self.noise_reduction_factor,\
                                 random_seed= self.random_seed,num_agents=self.num_agents,\
                                 update_every=self.update_every, actor_fc1_units=self.actor_fc1_units, \
                                 actor_fc2_units=self.actor_fc2_units,critic_fcs1_units=self.critic_fcs1_units, \
                                 critic_fc2_units=self.critic_fc2_units,gamma=self.gamma, tau=self.tau, \
                                 lr_actor=self.lr_actor, lr_critic=self.lr_critic,weight_decay=self.weight_decay,\
                                 mu=self.mu, theta=self.theta, sigma=self.sigma) for i in range(self.num_agents)]
            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed,num_agents)
            
        def act(self, states, add_noise=True):    
            actions = [self.agents[i].act(states[i, np.newaxis]).flatten() for i in range(self.num_agents)]
            actions = np.array(actions)
            return actions 


        def step(self, state, action, reward, next_state, done):
            """Save experience in replay memory, and use random sample from buffer to learn."""
            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)

            # Learn, if enough samples are available in memory
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)


        
        def learn(self, experiences, gamma):
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            obs, states, actions, rewards, next_obs, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next=self.agents[0].actor_target(next_obs[0])
            actions_pred = self.agents[0].actor_local(obs[0])
            for i in range(1, self.num_agents):
                actions_next=torch.cat((actions_next, self.agents[i].actor_target(next_obs[i])),1)
                actions_pred=torch.cat((actions_pred, self.agents[i].actor_local(obs[i])),1)
#             actions_next = self.actor_target(next_states)
            for i in range(self.num_agents):
                Q_targets_next = self.agents[i].critic_target(next_states, actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = rewards[i] + (gamma * Q_targets_next * (1 - dones[i]))
                # Compute critic loss
                Q_expected = self.agents[i].critic_local(states, actions)
                critic_loss = F.mse_loss(Q_expected, Q_targets)
                # Minimize the loss
                self.agents[i].critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.agents[i].critic_local.parameters(), 1)
                self.agents[i].critic_optimizer.step()

                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actor_loss = -self.agents[i].critic_local(states, actions_pred).mean()
                # Minimize the loss
                self.agents[i].actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.agents[i].actor_local.parameters(), 1)
                self.agents[i].actor_optimizer.step()

                # ----------------------- update target networks ----------------------- #
                self.agents[i].soft_update(self.agents[i].critic_local, self.agents[i].critic_target, self.tau)
                self.agents[i].soft_update(self.agents[i].actor_local, self.agents[i].actor_target, self.tau) 



class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents, noise_scalar_init, noise_reduction_factor, update_every=1, actor_fc1_units=400, actor_fc2_units=300,critic_fcs1_units=400, critic_fc2_units=300,gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0,mu=0., theta=0.15, sigma=0.2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.t_step = 0
        self.update_every=update_every
        
        self.actor_fc1_units=actor_fc1_units
        self.actor_fc2_units=actor_fc2_units
        self.critic_fcs1_units=critic_fcs1_units
        self.critic_fc2_units=critic_fc2_units            
        self.gamma=gamma
        self.tau=tau
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.weight_decay=weight_decay
        self.mu=mu 
        self.theta=theta 
        self.sigma=sigma
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.noise_scalar = noise_scalar_init
        self.noise_reduction_factor = noise_reduction_factor

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size,action_size,random_seed,\
                                 actor_fc1_units=self.actor_fc1_units,\
                                 actor_fc2_units=self.actor_fc2_units).to(device)
        self.actor_target = Actor(state_size,action_size,random_seed,\
                                 actor_fc1_units=self.actor_fc1_units,\
                                 actor_fc2_units=self.actor_fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),\
                                lr=self.lr_actor,weight_decay=self.weight_decay)
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.num_agents*state_size, self.num_agents*action_size, random_seed,\
                                   critic_fcs1_units=self.critic_fcs1_units,\
                                   critic_fc2_units=self.critic_fc2_units).to(device)
        self.critic_target =Critic(self.num_agents *state_size, self.num_agents*action_size, random_seed,\
                                   critic_fcs1_units=self.critic_fcs1_units,\
                                   critic_fc2_units=self.critic_fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),\
                                     lr=self.lr_critic,weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed,mu=self.mu,theta=self.theta, sigma=self.sigma)
        
#         # initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * self.noise_scalar
            self.noise_scalar *= self.noise_reduction_factor
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

                       

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
          
    def hard_update(self, target, source):   # couldn't you just do a soft update with Tau = 1.0????
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)   


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))]) #using normal distri
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.num_agents = num_agents
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        obs = [torch.from_numpy(np.vstack([e.state[i] for e in experiences if e is not None])).float().to(device) for i in range(self.num_agents)]
        states  = torch.from_numpy(np.vstack([e.state.flatten() for e in experiences if e is not None])).float().to(device) 
        
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action.flatten() for e in experiences if e is not None])).float().to(device)
        rewards = [torch.from_numpy(np.vstack([e.reward[i] for e in experiences if e is not None])).float().to(device) \
                   for i in range(self.num_agents)]
        next_obs =  [torch.from_numpy(np.vstack([e.next_state[i] for e in experiences if e is not None])).float().to(device) \
                     for i in range(self.num_agents)]
        next_states = torch.from_numpy(np.vstack([e.next_state.flatten() for e in experiences if e is not None])).float().to(device)
        dones = [torch.from_numpy(np.vstack([e.done[i] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for i in range(self.num_agents)]

        return (obs, states, actions, rewards, next_obs, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)