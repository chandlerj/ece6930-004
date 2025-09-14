import random
from collections import defaultdict, deque
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import csv

random.seed(42)
np.random.seed(42)

X, O, E = 'X', 'O', ' '  # players and empty

def empty_board():
    return tuple([E]*9)  # immutable state

def reset():
    return empty_board()

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),  # rows
    (0,3,6),(1,4,7),(2,5,8),  # cols
    (0,4,8),(2,4,6)           # diagonals
]

def check_winner(state):
    for a,b,c in WIN_LINES:
        line = (state[a], state[b], state[c])
        if line == (X,X,X): return X
        if line == (O,O,O): return O
    if E not in state: return 'DRAW'
    return None

def legal_actions(state):
    return [i for i,v in enumerate(state) if v == E]

def place(state, idx, p):
    """
    state - state of the board. a list of 9 elements
    idx   - idx of where to place p
    p     - the player's designation (X or O)
    """
    s = list(state)
    s[idx] = p
    return tuple(s)

def opponent_move(state):
    """Opponent O plays uniformly random legal move. If terminal, returns state."""
    if check_winner(state): return state
    acts = legal_actions(state)
    if not acts: return state
    return place(state, random.choice(acts), O)

def check_terminal(s1):
    w = check_winner(s1)
    if w == X:  return +1.0, True
    if w == O:  return -1.0, True
    if w == 'DRAW': return 0.0, True
    else: return 0.0, False


def step(state, action):
    """Agent X acts; environment responds with O's random move.
       Returns (next_state, reward, done) from X's perspective.
    """
    assert action in legal_actions(state), "Illegal action"
    s1 = place(state, action, X)
    w = check_winner(s1)
    if w == X:  return s1, +1.0, True
    if w == O:  return s1, -1.0, True
    if w == 'DRAW': return s1, 0.0, True

    # Opponent acts
    s2 = opponent_move(s1)
    w2 = check_winner(s2)
    if w2 == X:  return s2, +1.0, True
    if w2 == O:  return s2, -1.0, True
    if w2 == 'DRAW': return s2, 0.0, True
    return s2, 0.0, False  # ongoing

def random_policy(state):
    acts = legal_actions(state)
    return random.choice(acts)

# ---------- Monte Carlo Policy Evaluation (First-Visit) ----------
def generate_episode(policy):
    state = reset()

    states = []
    actions = []
    done = False
    reward = 0

    while not done:
        action = policy(state)

        states.append(state)
        actions.append(action)

        state, reward, done = step(state, action)

    return states, actions, reward

def mc_first_visit_V(num_episodes=50000):
    returns = {}
    V = {}
    for episode in tqdm(range(num_episodes)):
        states, _, reward = generate_episode(random_policy)
        visited = []
        for state in states:

            if state in visited:
                continue

            visited.append(state)
            if state not in returns:
                returns[state] = []
            returns[state].append(reward)
            V[state] = sum(returns[state]) / len(returns[state])
    return V

# ---------- Iterative policy evaluation ----------
def enumerate_reachable_X_states():
    """
    Returns a set of all boards (tuples) where it's X to move,
    reachable from the empty board by legal alternating play.
    """
    X_states = set([empty_board()])
    q = deque([empty_board()])

    while q:
        s = q.popleft()
        _, done = check_terminal(s)
        if done:
            continue

        # X moves
        for a in legal_actions(s):
            s1 = place(s, a, 'X')
            _, done1= check_terminal(s1)
            if done1:
                continue  # terminal after X's move; no O reply

            # O replies uniformly; enumerate all replies
            for o in legal_actions(s1):
                s2 = place(s1, o, 'O')  # back to X-to-move
                # s2 is an X-to-move state
                if s2 not in X_states:
                    X_states.add(s2)
                    q.append(s2)

    return X_states

def bellman_backup(state, V):
    legal = legal_actions(state)
    num_legal_actions = len(legal)

    if not num_legal_actions:
        winner = check_winner(state)
        match winner:
            case "X":
                return 1
            case "O":
                return -1
            case "DRAW":
                return 0
    v_new = 0
    for action in legal:
        s1 = place(state, action, 'X')
        reward, done = check_terminal(s1)
        
        if done:
            v_new += reward / num_legal_actions
        else:
            op_actions = legal_actions(s1)
            num_op_actions = len(op_actions)
            op_expectation = 0
            for op_action in op_actions:
                s2 = place(s1, op_action,'O')
                op_reward, op_done = check_terminal(s2)
                op_expectation += (op_reward if op_done else V[s2]) / num_op_actions
            v_new += op_expectation / num_legal_actions
    return v_new
            
def iterative_policy_eval(x_states, tol=1e-4):
    V = {s: 0.0 for s in x_states}

    while True:
        delta = 0
        for state in x_states:
            reward, done = check_terminal(state)
            v = reward if done else bellman_backup(state, V)
            
            delta = max(delta, abs(v - V[state]))

            V[state] = v
        if delta < tol:
            break

    return V

# ---------- One-step Improvement using V ----------
def greedy_one_step_with_V(state, V):
    """
    for any state, pick the action that maximizes the one-step lookahead using
    the V table
    
    formulated as:
    \pi = \argmax_a \sum_{s',r} p(s',r|s,a)\[r + \gamma V(s')\]
    """
    best = -10000000000000.0
    choices = []

    for action in legal_actions(state):
        s1 = place(state, action, 'X')
        reward, done = check_terminal(s1)
        if done:
            q = reward
        else:
            op_expectation = 0
            op_actions = legal_actions(s1)
            num_op_actions = len(op_actions)
            for op_action in op_actions:
                s2 = place(s1, op_action, 'O')
                op_done, op_reward = check_terminal(s2)
                op_expectation += (op_reward if op_done else V[s2]) / num_op_actions
            q = op_expectation
        if q > best:
            best = q
            choices = [action]
        elif q == best:
            choices.append(action)
    return random.choice(choices)


def improved_policy(V):
    def pi(s):
        return greedy_one_step_with_V(s, V)
    return pi

# ---------- Evaluation ----------
def play_many(policy, n=10000):
    wins=draws=losses=0
    for _ in tqdm(range(n)):
        s = empty_board()
        done=False
        while not done:
            a = policy(s)
            s, r, done = step(s, a)
        if r > 0: wins += 1
        elif r < 0: losses += 1
        else: draws += 1
    return wins/n, draws/n, losses/n

if __name__ == "__main__":
    print("Estimating V under random policy...")
    V = mc_first_visit_V(50000)
    v_empty = V.get(empty_board(), 0.0)
    print("V_random(empty) ≈", round(v_empty, 4))
    
    # UNCOMMENT FOR PLOTTING OF VALUE DISTRIBUTIONS
    # sns.histplot(V.values())
    # plt.title('distribution of values under first visit monte-carlo')
    # plt.show()

    print("Estimating V under random policy (iterative policy evaluation)")
    x_states = enumerate_reachable_X_states()
    V = iterative_policy_eval(x_states)
    v_empty = V.get(empty_board(), 0.0)
    print("V_random(empty) ≈", round(v_empty, 4))

    # UNCOMMENT FOR PLOTTING OF VALUE DISTRIBUTIONS
    # sns.histplot(V.values())
    # plt.title('distribution of values under iterative policy improvement')
    # plt.show()

    # Dump the Value table to a CSV
    fieldnames = ['State', 'Value']
    with open('value-table.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, value in V.items():
            writer.writerow([key, value])

    print("Evaluating random vs random:")
    w,d,l = play_many(random_policy, 10000)
    print(f"Random policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}")

    pi1 = improved_policy(V)
    print("Evaluating improved policy:")
    w,d,l = play_many(pi1, 10000)
    print(f"Improved policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}")
