import random
from collections import defaultdict

X, O, E = 'X','O',' '
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def empty_board(): return (E,)*9
def legal_actions(s): return [i for i,v in enumerate(s) if v==E]
def place(s,i,p): 
    lst=list(s); lst[i]=p; return tuple(lst)
def winner(s):
    for a,b,c in WIN_LINES:
        line=(s[a],s[b],s[c])
        if line==(X,X,X): return X
        if line==(O,O,O): return O
    return 'DRAW' if E not in s else None

def step(s, a, opponent='random'):
    """X plays action a; O replies (random). Returns s', r, done (X view)."""
    s1 = place(s, a, X)
    w = winner(s1)
    if w==X:  return s1, +1, True
    if w==O:  return s1, -1, True
    if w=='DRAW': return s1, 0, True
    # O move
    acts = legal_actions(s1)
    if not acts: return s1, 0, True
    if opponent=='random':
        o = random.choice(acts)
    else:
        o = random.choice(acts)  # could plug in smarter O
    s2 = place(s1, o, O)
    w2 = winner(s2)
    if w2==X: return s2, +1, True
    if w2==O: return s2, -1, True
    if w2=='DRAW': return s2, 0, True
    return s2, 0, False

# --- Q-learning ---
def q_learning(num_episodes=5000, alpha=0.5, gamma=1.0, eps=0.1):
    Q = defaultdict(float)
    def eps_greedy(s):
        acts = legal_actions(s)
        if random.random() < eps: return random.choice(acts)
        return max(acts, key=lambda a: Q[(s,a)] if (s,a) in Q else 0.0)
    def maxQ(s):
        acts = legal_actions(s)
        return max((Q[(s,a)] for a in acts), default=0.0)

    for _ in range(num_episodes):
        s = empty_board()
        done = False
        while not done:
            a = eps_greedy(s)
            s2, r, done = step(s, a, opponent='random')
            target = r + (0 if done else gamma*maxQ(s2))
            Q[(s,a)] += alpha * (target - Q[(s,a)])
            s = s2
    return Q

# --- Choose a test board state ---
test_state = (X, O, E,
              E, E, E,
              E, E, E)   # X in corner, O in opposite corner

def print_q_for_state(Q, state):
    acts = legal_actions(state)
    print("Board:")
    for r in range(0,9,3):
        print(state[r:r+3])
    print("Legal moves & Q-values:")
    for a in acts:
        print(f"  Move {a}: Q={Q[(state,a)]:.3f}")
    print()

# --- Train early vs late ---
Q_early = q_learning(500)    # short run
Q_final = q_learning(20000)  # longer run

print("=== Early Q-table values ===")
print_q_for_state(Q_early, test_state)

print("=== Final Q-table values ===")
print_q_for_state(Q_final, test_state)
