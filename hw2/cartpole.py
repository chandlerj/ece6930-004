from collections import namedtuple, deque
from itertools import count
import math
import random
import time

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

#----------------------------------------------------------------ENVIRONMENT PARAMETERS

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1500
TAU = 0.005
LR = 3e-4

# PER Hyperparams
PER_ALPHA = 0.6
PER_BETA0 = 0.4
PER_EPS = 1e-5


# choose if want to use the priority replay buffer
USE_PRIORITY_BUFFER=False

#----------------------------------------------------------------PLOT BUFFERS

dqn_rewards = []
ddqn_rewards = []
dqn_episode_durations = []
ddqn_episode_durations = []

#----------------------------------------------------------------GLOBALS (sorry)

steps_done = 0
per_beta = PER_BETA0

#----------------------------------------------------------------ENVIRONMENT SETUP

env = gym.make("CartPole-v1", render_mode='rgb_array')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# To ensure reproducibility during training, you can fix the random seeds
# by uncommenting the lines below. This makes the results consistent across
# runs, which is helpful for debugging or comparing different approaches.
#
# That said, allowing randomness can be beneficial in practice, as it lets
# the model explore different training trajectories.
seed = 42
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


#----------------------------------------------------------------MODEL SETUP

# This tuple is used to batch experiences together into a single tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        """
        Initialize the memory with a deque object. We 
        use a deque instead of a traditional python list
        as it is efficient for both FIFO and LIFO structures,
        alongside many operations having faster complexities
        compared to vanilla python implementations.

        Another secondary benefit is that the deque will only
        have at most `capacity` elements, and once the capacity is
        met, the left/right-most item will be pushed out of the queue.
        This means our replay buffer will only have the `capacity` newest
        experiences.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        randomly return an experience from the replay buffer.
        """
        return random.sample(self.memory, batch_size)
    
    def clear(self):
        """
        clear memory - helpful for resetting environment
        """
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

# You will see this flag throughout the implementation as it allows
# the PER implementation and vanilla dqn/ddqn implementations coexist
if USE_PRIORITY_BUFFER:
    # Use the torch replay buffer which natively supports the PER sampler.
    # uses a sum-tree to store priorities which allows for O(log n) retreival
    # of samples.
    memory = ReplayBuffer(
            storage=ListStorage(10000),
            sampler=PrioritizedSampler(max_capacity=10000, alpha=PER_ALPHA, beta=PER_BETA0),
            collate_fn=lambda x: x,)
    
else:
    # The ReplayMemory will hold 10_000 previous experiences (steps)
    memory = ReplayMemory(10000)
 

class DQN(nn.Module):
    """
    DQN inherits the nn.Module class, allowing
    for construction of the neural network which
    will be used in place of the Q-table.
    """
    def __init__(self, n_observations, n_actions):
        """
        Construct the neural network as a fully 
        connected NN with a hidden layer of 128x128
        neurons
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """
        function called to feed an input through the network
        with a ReLU activation function

          6       /
                 / 
          0 -----  
            0     6
        fig: ReLU in ASCII
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




def select_action(state):
    """
    Select the action to perform given the current state.
    Uses epsilon greedy sampling to allow for exploration
    in early phases of training, with a decay towards 
    picking optimal actions as the number of steps completed
    increases. As the policy gets to a closer approximation of
    q*, the number of actions chosen by the model and not by
    random sampling increases.
    """
    global steps_done
    
    # this is the random value used to determine if explore/exploit
    # will be chosen
    sample = random.random()

    # this decays the probability of selecting a random action as
    # the number of steps increases.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    # print(f"epsilon threshold: {eps_threshold}")
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(show_result=False):
    """
    Plotter that plots the reward, since the reward for this environment
    is just the duration the pole is upright. I failed to read this in the
    documentation, so I have redundent plotting code
    """
    plt.figure(1)
    dqn_durations_t = torch.tensor(dqn_episode_durations, dtype=torch.float)
    ddqn_durations_t = torch.tensor(ddqn_episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(dqn_durations_t.numpy())
    plt.plot(ddqn_durations_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model(ddqn=False):

    global per_beta
    global memory

    if len(memory) < BATCH_SIZE:
        """
        only optimize if enough experiences are in replay buffer
        to train on the desired batch size.
        """
        return

    # this sampled the replay buffer either from the PER buffer
    # or randomly from the vanilla buffer
    if USE_PRIORITY_BUFFER:
        transitions, info = memory.sample(BATCH_SIZE, return_info=True)
        # We need this to get the indicies of the experiences so we
        # can update the sum tree with the obtained priorities
        probs, index = info.values()
        index.reshape(-1)
    else:
        transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # filter out non-terminal states
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).long().view(-1,1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if ddqn:
            # find actions which the policy network considers optimal
            optimal_actions = policy_net(non_final_next_states).max(1).indices.long().unsqueeze(1)
            # access their value using the target net and feed that as the next state values
            target_q_vals = target_net(non_final_next_states).gather(1, optimal_actions).squeeze(1)
            next_state_values[non_final_mask] = target_q_vals
        else:    
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    td_error = (expected_state_action_values - state_action_values).detach()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss(reduction="none")
    loss = criterion(state_action_values, expected_state_action_values).mean()


    if USE_PRIORITY_BUFFER:
        # this is the juice of PER. We compute the importance of each sample 
        # and floor it to a near-zero but not zero value
        probs_tensor = torch.as_tensor(probs, dtype=torch.float32, device=device)
        N = len(memory)
        per_beta = min(1.0, per_beta + 1e-5)
        importance_sampling = (N * probs_tensor).clamp_min(1e-12).pow(-per_beta)
        importance_sampling = importance_sampling / importance_sampling.max().clamp_min(1e-12)
        loss = (importance_sampling * loss).mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    if USE_PRIORITY_BUFFER:
        # this is where the TD-error is actually computed and stored 
        # in the sum-tree
        new_prios = (td_error.abs() + PER_EPS).pow(PER_ALPHA)
        memory.update_priority(index.reshape(-1), new_prios.reshape(-1))


def plot_rewards(show_result=False):
    """
    This is basically the same exact thing as the
    plot_durations function
    """
    plt.figure(2)
    dqn_plot = torch.tensor(dqn_rewards, dtype=torch.float)
    ddqn_plot = torch.tensor(ddqn_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.plot(dqn_plot.numpy())
    plt.plot(ddqn_plot.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#----------------------------------------------------------------TRAINING LOOP


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50


#----------------------------------------------------------------DDQN RUN - RESETTING ENVIRONMENT
steps_done=0


per_beta = PER_BETA0
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# this the policy being used to perform actions
policy_net = DQN(n_observations, n_actions).to(device)
# this is a copy of the policy net which will be updated
# at less frequent rates to increase stability of training
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# this sets up the AdamW optimizer, which adapts the learning
# rate for each parameter as the model is trained. This is done
# applying 'weight decay' to the loss function which improves
# the generalization of the trained model. This causes the weights
# to converge to 0 producing a simpler (and more generalized) model.
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)




start_time = time.time()
print(f"DQN RUN")
for i_episode in range(num_episodes):
    # initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    ep_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        if USE_PRIORITY_BUFFER:
            indicies = memory.extend([(state, action, next_state, reward)])
        else:
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

        # move to the next state
        state = next_state

        # perform one step of the optimization (on the policy network)
        optimize_model()

        # soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            dqn_rewards.append(ep_reward)
            plot_rewards()
            break
end_time = time.time()
total_time = end_time - start_time
print(f'Complete in {total_time} seconds')

env.close()


env = gym.make("CartPole-v1", render_mode='rgb_array')
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

steps_done=0


per_beta = PER_BETA0
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# this the policy being used to perform actions
policy_net = DQN(n_observations, n_actions).to(device)
# this is a copy of the policy net which will be updated
# at less frequent rates to increase stability of training
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# this sets up the AdamW optimizer, which adapts the learning
# rate for each parameter as the model is trained. This is done
# applying 'weight decay' to the loss function which improves
# the generalization of the trained model. This causes the weights
# to converge to 0 producing a simpler (and more generalized) model.
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

if USE_PRIORITY_BUFFER:
    memory = ReplayBuffer(
        storage=ListStorage(10000),
        sampler=PrioritizedSampler(max_capacity=10000, alpha=PER_ALPHA, beta=PER_BETA0),
        collate_fn=lambda x: x,
    )
else:
    memory = ReplayMemory(10000)


start_time = time.time()
print(f"DDQN RUN")
for i_episode in range(num_episodes):
    # initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    ep_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        if USE_PRIORITY_BUFFER:
            indicies = memory.extend([(state, action, next_state, reward)])
        else:
            # store the transition in memory
            memory.push(state, action, next_state, reward)

        # move to the next state
        state = next_state

        # perform one step of the optimization (on the policy network)
        optimize_model(True)

        # soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            ddqn_rewards.append(ep_reward)
            plot_rewards()
            break
end_time = time.time()
total_time = end_time - start_time
print(f'Complete in {total_time} seconds')



plot_rewards(show_result=True)
plt.ioff()
plt.show()
