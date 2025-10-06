import random

ALPHA = 0.5

A={'S0':['Up', 'Down', 'Left', 'Right'],
   'S1':['Up', 'Down', 'Left', 'Right'],
   'S2':['Up', 'Down', 'Left', 'Right'],
   'S3':['Up', 'Down', 'Left', 'Right'],}

Q={s:{a:0.0 for a in A[s]} for s in ['S0','S1','S2','S3']}

rewards = { 
           'S0':    0,
           'S1':    0,
           'S2':    0,
           'S3':    0,
           'Bomb': -1,
           'Goal':  1,
          }

rewards_action_penalty = {
           'S0':    -0.25,
           'S1':    -0.25,
           'S2':    -0.25,
           'S3':    -0.25,
           'Bomb':  -1.25,
           'Goal':   0.75,
          }
inbound_transitions = {
    ('S0', 'Right'): ['S1', 0],
    ('S0', 'Down'):  ['S3', 0],
    ('S1', 'Down'):  ['Bomb', -1],
    ('S1', 'Right'): ['S2', 0],
    ('S1', 'Left'):  ['S0', 0],
    ('S2', 'Down'):  ['Goal', 1],
    ('S2', 'Left'):  ['S1', 0],
    ('S3', 'Right'): ['Bomb', -1],
    ('S3', 'Up'):    ['S0', 0]
}


inbound_transitions_penalty = {
    ('S0', 'Right'): ['S1', -0.25],
    ('S0', 'Down'):  ['S3', -0.25],
    ('S1', 'Down'):  ['Bomb', -1.25],
    ('S1', 'Right'): ['S2', -0.25],
    ('S1', 'Left'):  ['S0', -0.25],
    ('S2', 'Down'):  ['Goal', 0.75],
    ('S2', 'Left'):  ['S1', -0.25],
    ('S3', 'Right'): ['Bomb', -1.25],
    ('S3', 'Up'):    ['S0', -0.25]
}

def wall_hit(s, teleport=False, action_cost=False):
    if teleport:
        new_cell = random.choice(list(rewards.keys()))
        return new_cell, rewards[new_cell] if not action_cost else rewards_action_penalty[new_cell]
    else:
        return s, 0


def step(state, action, teleport=False, action_cost=False):
    if (state, action) in inbound_transitions:
        sprime, r = inbound_transitions[(state, action)] if not action_cost else inbound_transitions_penalty[(state, action)]
        return sprime, r

    return wall_hit(state, teleport, action_cost)


def maxQ(s):
    if s in ('Bomb','Goal'):
        return 0.0
    return max(Q[s].values())

def update(s, a, teleport=False, random_action=False, action_cost=False, random_p=0.2):
    if random_action:
        coin = 1 if random.random() < random_p else 0
        if coin:
            a = random.choice(['Up', 'Down', 'Left', 'Right'])

    sprime, r = step(s,a, teleport, action_cost)
    if sprime == s and not teleport:
        return # we are trying to transition into a wall
    target = r + (0 if sprime in ('Bomb','Goal') else maxQ(sprime))
    Q[s][a] = Q[s][a] + ALPHA * (target - Q[s][a])

def print_Q(Q):
    for state, values in Q.items():
        print(state, end='\t')
        for value in values.values():
            print(f'{value:8.5f}', end=' ')
        print()

if __name__ == "__main__":
    random.seed(42)
    ITERATIONS = 100
    TELEPORT = True
    RANDOM_MOTION = True
    ACTION_COST = True
    for iter in range(ITERATIONS):
        for s in A.keys():
            for a in A[s]:
                update(s,a, TELEPORT, RANDOM_MOTION, ACTION_COST)
        print(f"Q-table at iteration {iter}")
        print_Q(Q)
