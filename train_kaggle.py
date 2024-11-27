import json

with open('label_dict.json') as json_file:
    label_dict = json.load(json_file)

label_dict
import json

with open('class_weights.json') as json_file:
    class_weights = json.load(json_file)

class_weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
from gym import  spaces
import gym
import json
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import platform
import torch.cuda as cuda
import cpuinfo

# Add system info printing
def print_system_info():
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {cpuinfo.get_cpu_info()['brand_raw']}")
    print(f"RAM: {psutil.virtual_memory().total / (1024.0 ** 3):.1f} GB")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # GPU information
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        print(f"GPU Available: Yes")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU Available: No")
    
    print("\n=== Current Device ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=====================\n")

# Call system info print function before main execution
print_system_info()

use_subset = True  # Set to False to use the full dataset
suffix = "_subset.csv" if use_subset else ".csv"
if use_subset:
    train_df = pd.read_csv("processed_data/train_df_subset.csv")
    test_df = train_df  # Use the same data for testing
else:
    train_df = pd.read_csv("processed_data/train_df.csv")
    test_df = pd.read_csv("processed_data/test_df.csv")

with open('processed_data/class_weights.json', 'r') as f:
    class_weights = json.load(f)
with open('processed_data/label_dict.json', 'r') as f:
    label_dict = json.load(f)
class data_cls:
    def __init__(self, train_test, attack_map, **kwargs):
        self.train_test = train_test
        if use_subset and self.train_test == 'test':
            self.path = "processed_data/train_df_subset.csv"
        else:
            self.path = f"processed_data/{self.train_test}_df.csv"
        self.attack_map =   attack_map 
        self.attack_types = list(attack_map.keys())
        
        self.loaded = False
    
    def get_batch(self, batch_size=100):
        if not self.loaded:
            self._load_df()
        
        # Ensure batch_size does not exceed the DataFrame size
        if batch_size > self.data_shape[0]:
            raise ValueError(f"batch_size ({batch_size}) cannot be larger than the dataset size ({self.data_shape[0]}).")
        
        # Calculate wrapped indices using modulo
        indexes = [(self.index + i) % self.data_shape[0] for i in range(batch_size)]
        
        # Update the index for the next batch
        self.index = (self.index + batch_size) % self.data_shape[0]
        
        # Select the batch using iloc with valid indices
        batch = self.df.iloc[indexes]
        
        map_type = pd.Series(index=self.attack_types, data=np.arange(len(self.attack_types))).to_dict()
        labels = batch[label_col].map(self.attack_map).map(map_type).values
        del batch[label_col]
        
        return np.array(batch), labels
    
    def get_full(self):

        self._load_df()
        
        batch = self.df
        map_type = pd.Series(index=self.attack_types,data=np.arange(len(self.attack_types))).to_dict()
        labels = batch[label_col].map(self.attack_map).map(map_type).values
        
        del(batch[label_col])
        
        return np.array(batch), labels
    
    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        return self.data_shape
    
    def _load_df(self):
        self.df = pd.read_csv(self.path)
            
        self.index=np.random.randint(0,self.df.shape[0]-1,dtype=np.int32)
        self.loaded = True
class NetworkClassificationEnv(gym.Env, data_cls):
    def __init__(self,train_test, attack_map, **kwargs):
        data_cls.__init__(self,train_test, attack_map,**kwargs)
        self.data_shape = self.get_shape()
        self.batch_size = kwargs.get('batch_size', 1) 
        self.fails_episode = kwargs.get('fails_episode', 10) 
        
        # Gym spaces
        self.action_space = spaces.Discrete(len(self.attack_types))
        self.observation_space = spaces.Discrete(self.data_shape[0])
        
        self.observation_len = self.data_shape[1]-1
        
        self.counter = 0

    def _update_state(self):
        self.states,self.labels = self.get_batch(self.batch_size)
        

    def reset(self):
        self.states,self.labels = self.get_batch(self.batch_size)
        self.counter = 0
        
        return self.states
    
    def _get_rewards(self,actions):
        self.reward = 0
        if actions == self.labels:
            self.reward = 1
        else: 
            self.counter += 1

    def step(self,actions):
        self._get_rewards(actions)
            
        self._update_state()

        if self.counter >= self.fails_episode:
            self.done = True
        else:
            self.done = False
            
        return self.states, self.reward, self.done

# Modify the QNetwork class to use PyTorch
class QNetwork(nn.Module):
    def __init__(self, obs_size, num_actions, hidden_dense_layer_dict={"Dense_1": {"Size": 100}}, learning_rate=0.001):
        super(QNetwork, self).__init__()
        layers = []
        input_size = obs_size
        for key, value in hidden_dense_layer_dict.items():
            layers.append(nn.Linear(input_size, value["Size"]))
            layers.append(nn.ReLU())
            input_size = value["Size"]
        layers.append(nn.Linear(input_size, num_actions))
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)

# Update the Agent class to use PyTorch optimizers and loss functions
class Agent(object):
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size
        
        self.epsilon = kwargs.get('epsilon', 1)
        self.gamma = kwargs.get('gamma', 0.001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate',0.99)
        self.exp_rep = kwargs.get('exp_rep',True)
        
        if self.exp_rep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))
        
        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time

        self.model_network = QNetwork(
            self.obs_size, 
            self.num_actions,
            hidden_dense_layer_dict=kwargs.get('hidden_dense_layer_dict', {"Dense_1": {"Size": 100}})
        )
        
        self.target_model_network = QNetwork(
            self.obs_size, 
            self.num_actions,
            hidden_dense_layer_dict=kwargs.get('hidden_dense_layer_dict', {"Dense_1": {"Size": 100}})
        )
        
        self.optimizer = optim.Adam(self.model_network.parameters(), lr=kwargs.get('learning_rate', 0.001))
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        # Copy weights from the model network to the target network
        self.target_model_network.load_state_dict(self.model_network.state_dict())
        
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network,
                                         len(actions),
                                         self.epsilon,
                                         self.decay_rate,
                                         self.epoch_length)
        
    def act(self,states):
        actions = self.policy.get_actions(states)
        return actions
    
    def learn(self, states, actions,next_states, rewards, done):
        if self.exp_rep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done


    def update_model(self):
        if self.exp_rep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done
            
        # Convert everything to PyTorch tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        done = torch.FloatTensor(done)
        actions = torch.LongTensor(actions)
            
        # Get Q values for next states
        next_Q = self.model_network(next_states)
        
        # Find best actions for next states
        next_actions = torch.argmax(next_Q, dim=1)
        
        # Get current Q values
        current_Q = self.target_model_network(states)
        
        # Create target Q values
        targets = current_Q.clone()
        for i in range(len(actions)):
            targets[i, actions[i]] = rewards[i] + \
                                   self.gamma * next_Q[i, next_actions[i]] * \
                                   (1 - done[i])
        
        # Calculate loss and update
        loss = self.criterion(current_Q, targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
            self.target_model_network.load_state_dict(self.model_network.state_dict()) 
        
        return loss.item()

class Policy:
    def __init__(self, num_actions, estimator):
        self.num_actions = num_actions
        self.estimator = estimator
    
class Epsilon_greedy(Policy):
    def __init__(self,estimator ,num_actions,epsilon,decay_rate, epoch_length):
        Policy.__init__(self, num_actions, estimator)
        self.name = "Epsilon Greedy"
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush = True)
            sys.exit(0)
        self.epsilon = epsilon
        self.step_counter = 0
        self.epoch_length = epoch_length
        self.decay_rate = decay_rate
        self.epsilon_decay = True
        
    def get_actions(self,states):
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions,states.shape[0])
        else:
            # Convert numpy array to PyTorch tensor
            states_tensor = torch.FloatTensor(states)
            self.Q = self.estimator(states_tensor)
            # Convert back to numpy for processing
            self.Q = self.Q.detach().numpy()

            actions = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row]))
                actions.append(best_actions[np.random.choice(len(best_actions))].item())
            
        self.step_counter += 1 

        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                self.epsilon = max(.01, self.epsilon * self.decay_rate**self.step_counter)
            
        return actions

        
class ReplayMemory(object):
    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros(self.max_size * 1 * self.observation_size,
                                       dtype=np.float32).reshape(self.max_size, self.observation_size),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
                 'reward'   : np.zeros(self.max_size * 1).reshape(self.max_size, 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
               }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1, :], dtype=np.float32)

        a      = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r      = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done   = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)
import shutil

if os.path.isdir("models"):
    shutil.rmtree("models", ignore_errors=False, onerror=None)
label_col = 'Label'
model_path = "models"

epsilon = 1  

batch_size = 1

minibatch_size = 100
exp_rep = True

iterations_episode = 100  # Changed back from 10 to 100

decay_rate = 0.99
gamma = 0.001

learning_rate = 0.001

hidden_dense_layer_dict = {"Dense_2": {"Size": 64},
                           "Dense_3": {"Size": 32}
                           }

env = NetworkClassificationEnv('train',
                                label_dict,
                                batch_size = batch_size,
                                iterations_episode = iterations_episode)

# num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
num_episodes = 300
valid_actions = list(range(len(env.attack_types)))
num_actions = len(valid_actions)

obs_size = env.observation_len

agent = Agent(valid_actions,
              obs_size,
              "EpsilonGreedy",
              learning_rate = learning_rate,
              epoch_length = iterations_episode,
              epsilon = epsilon,
              decay_rate = decay_rate,
              gamma = gamma,
              hidden_dense_layer_dict = hidden_dense_layer_dict,
              minibatch_size=minibatch_size,
              mem_size = 10000,
              exp_rep=exp_rep)    


# Statistics
reward_chain = []
loss_chain = []

# Main loop
for epoch in range(num_episodes):
    start_time = time.time()
    loss = 0.
    total_reward_by_episode = 0

    states = env.reset()

    done = False

    true_labels = np.zeros(len(env.attack_types))
    estimated_labels = np.zeros(len(env.attack_types))

    for i_iteration in range(iterations_episode):
        actions = agent.act(states)

        estimated_labels[actions] += 1
        true_labels[env.labels] += 1

        next_states, reward, done = env.step(actions)
        agent.learn(states, actions, next_states, reward, done)

        if exp_rep and epoch*iterations_episode + i_iteration >= minibatch_size:
            loss += agent.update_model()
        elif not exp_rep:
            loss += agent.update_model()

        update_end_time = time.time()

        states = next_states

        total_reward_by_episode += np.sum(reward, dtype=np.int32)

    reward_chain.append(total_reward_by_episode)    
    loss_chain.append(loss) 


    end_time = time.time()
    if epoch % 10 == 0 or epoch == num_episodes - 1:
        print("\r|Epoch {:03d}/{:03d} | Loss {:4.4f} |" 
              "Tot reward in ep {:03d}| time: {:2.2f}|"
              .format(epoch, num_episodes 
              ,loss, total_reward_by_episode,(end_time-start_time)))
        print("\r|Estimated: {}|Labels: {}".format(estimated_labels,true_labels))

if not os.path.exists('models'):
    os.makedirs('models')
torch.save(agent.model_network.state_dict(), "models/DDQN_model.pth")
plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(len(reward_chain)),reward_chain)
plt.title('Total reward by episode')
plt.xlabel('n Episode')
plt.ylabel('Total reward')

plt.subplot(212)
plt.plot(np.arange(len(loss_chain)),loss_chain)
plt.title('Loss by episode')
plt.xlabel('n Episode')
plt.ylabel('loss')
plt.tight_layout()
plt.show()


if not os.path.exists('results'):
    os.makedirs('results')
    
plt.savefig('results/train_type_improved.png', format='png', dpi=300)