import random
from collections import defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

X, O, E = 'X', 'O', ' '  # players and empty

def empty_board():
    return tuple([E]*9)  # immutable state

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
    s = list(state)
    s[idx] = p
    return tuple(s)

def opponent_move(state):
    """Opponent O plays uniformly random legal move. If terminal, returns state."""
    if check_winner(state): return state
    acts = legal_actions(state)
    if not acts: return state
    return place(state, random.choice(acts), O)

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
    # TODO: Make this work
    pass

def mc_first_visit_V(num_episodes=50000):
    # TODO: Make this work
    pass

# ---------- One-step Improvement using V ----------
def greedy_one_step_with_V(state, V):
    # TODO: This one too if you are a graduate student
    pass

def improved_policy(V):
    def pi(s):
        return greedy_one_step_with_V(s, V)
    return pi

# ---------- Evaluation ----------
def play_many(policy, n=10000):
    wins=draws=losses=0
    for _ in range(n):
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

    print("Evaluating random vs random:")
    w,d,l = play_many(random_policy, 10000)
    print(f"Random policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}")

    pi1 = improved_policy(V)
    print("Evaluating improved policy:")
    w,d,l = play_many(pi1, 10000)
    print(f"Improved policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}")
