import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time

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
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


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

# The ReplayMemory will hold 10_000 previous experiences (steps)
memory = ReplayMemory(10000)


steps_done = 0


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


dqn_episode_durations = []
ddqn_episode_durations = []


def plot_durations(show_result=False):
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
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if ddqn:
            # find actions which the policy network considers optimal
            optimal_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
            # access their value using the target net and feed that as the next state values
            target_q_vals = target_net(non_final_next_states).gather(1, optimal_actions).squeeze(1)
            next_state_values[non_final_mask] = target_q_vals
        else:    
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

dqn_rewards = []
ddqn_rewards = []
def plot_rewards(show_result=False):
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

start_time = time.time()
print("DQN RUN")
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
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

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            dqn_episode_durations.append(t + 1)
            dqn_rewards.append(ep_reward)
            plot_durations()
            plot_rewards()
            break
end_time = time.time()

total_time = end_time - start_time
print(f'Complete in {total_time} seconds')
print("RESETTING MEMORY")
memory.clear()

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

# The ReplayMemory will hold 10_000 previous experiences (steps)
memory = ReplayMemory(10000)

print("DDQN RUN")
start_time = time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
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

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(ddqn=True)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            ddqn_episode_durations.append(t + 1)
            ddqn_rewards.append(ep_reward)
            plot_durations()
            plot_rewards()
            break
end_time = time.time()

total_time = end_time - start_time
print(f'Complete in {total_time} seconds')

plot_rewards(show_result=True)
plot_durations(show_result=True)
plt.ioff
plt.show()
