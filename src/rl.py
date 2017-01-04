import numpy as np
from numpy.linalg import inv

def trajectory_return(traj, gamma):
    """
    Compute the return of a Monte-Carlo trajectory
    """
    ret = 0
    for x, a, r in traj[::-1]:
        ret = ret * gamma + r
    return ret


def Bellman(policy):
    def T(W):
        s1 = ((P * R).sum(axis=1) * policy).sum(axis=0)
        s2 = (P.swapaxes(1, 2) * policy.reshape(k, n, 1)).sum(axis=0).dot(W)
        return s1 + gamma * s2
    return T


def mat_value(P, R, policy, gamma):
    k, n, _ = P.shape
    In = np.eye(n)
    A = (P.swapaxes(1, 2) * policy.reshape(k, n, 1)).sum(axis=0)
    B = ((P * R).sum(axis=1) * policy).sum(axis=0)
    return inv(In - gamma * A).dot(B)


def value_iteration(P, R, max_K=100):
    V = np.random.random(n)
    log = []
    for i in range(max_K):
        log.append(V)
        next_V = T(P, R, V)
        # fixpoint stop
        if abs(next_V - V).max() < almost_zero:
            break
        V = next_V
    return greedy_policy(P, R, V), np.array(log)



def policy_iteration(P, R, max_K=100):
    k, n, _ = P.shape
    policy = np.random.random((k, n))
    log = []
    for i in range(max_K):
        V = mat_value(P, R, policy)
        log.append(V)
        next_policy = greedy_policy(P, R, V)
        # fixpoint stop
        if abs(next_policy - policy).max() < almost_zero:
            break
        policy = next_policy
    return policy, np.array(log)


# RL procedures for discrete gym environments with attributes `transition` and
# `reward` as matrices.

def gym_mat_value(env, policy, gamma):
    return mat_value(env.transition.swapaxes(1, 2), env.reward.swapaxes(1, 2), policy, gamma)

def gym_value_iteration(env, max_K=100):
    return value_iteration(env.transition.swapaxes(1, 2), env.reward.swapaxes(1, 2), max_K=max_K)

def gym_policy_iteration(env, max_K=100):
    return policy_iteration(env.transition.swapaxes(1, 2), env.reward.swapaxes(1, 2), max_K=max_K)


def exploration_policy(q, epsilon=0.1):
    """Exploration policy is a relaxed greedy policy where there is
    a probability epsilon of not choosing the argmax of q."""
    k = q.shape[0]
    policy_template = np.eye(k) * (1 - epsilon) + np.ones((k, k)) * epsilon / k
    return policy_template[q.argmax(axis=0)].T


def q_learning(P, R, x0,
               nb_episodes=100, episode_length=1000,
               epsilon=0.1, alpha0=1.0, alpha_decay = 1.0):
    
    # Initial $\hat Q$ matrix
    q = np.zeros((k, n))
    # Visit counter for alpha()
    visits = np.zeros(n)
    # Log of performance
    log = []
    
    for ep in range(nb_episodes):
        traj = []
        x = x0
        for i in range(episode_length):
            policy = exploration_policy(q, epsilon)
            
            a = np.random.choice(action_space, p=policy[:, x])
            next_x = np.random.choice(state_space, p=P[a, :, x])
            r = R[a, next_x, x]
            
            delta = r + gamma * q[:, next_x].max() - q[a, x]
            alpha = min(1.0, alpha0 / (visits[x] + 1) ** alpha_decay)
            
            q[a, x] += alpha * delta
            visits[x] += 1
            
            traj.append((x, a, r))
            x = next_x
        
        # Performance evaluation
        greedy = exploration_policy(q, epsilon=0)
        V = mat_value(P, R, greedy)
        ret = trajectory_return(traj)
        log.append((V, ret, q))
    
    return exploration_policy(q, epsilon), log


def gym_q_learning(env, gamma=0.99,
                   nb_episodes=100, episode_length=1000,
                   epsilon=0.1, alpha0=1.0, alpha_decay = 1.0):
    """
    env: discrete gym env with special attribute state_space supporting "len()"
    """
    observation = env.reset()
    n = len(env.state_space)
    k = env.action_space.n

    # Initial $\hat Q$ matrix
    q = np.zeros((k, n))
    # Visit counter for alpha()
    visits = np.zeros(n)
    # Log of performance
    log = []

    for ep in range(nb_episodes):
        traj = []
        state = env.reset()
        for i in range(episode_length):
            policy = exploration_policy(q, epsilon)
            
            action = np.random.choice(range(k), p=policy[:, state])
            next_state, reward, done, info = env.step(action)

            delta = reward + gamma * q[:, next_state].max() - q[action, state]
            alpha = min(1.0, alpha0 / (visits[state] + 1) ** alpha_decay)
            
            q[action, state] += alpha * delta
            visits[state] += 1
            
            traj.append((state, action, reward))
            state = next_state
        
        # Performance evaluation
        greedy = exploration_policy(q, epsilon=0)
        if hasattr(env, 'transition') and hasattr(env, 'reward'):
            V = gym_mat_value(env, greedy, gamma)
        else:
            V = -1
        ret = trajectory_return(traj, gamma)
        log.append((V, ret, q))
    
    return exploration_policy(q, epsilon), log


def gym_mc_trajectory(env, policy, T):
    """
    env: gym environment
    policy: policy to evaluate
    T: number of simulation steps
    """
    k = env.action_space.n
    trajectory = []
    state = env.reset()
    for i in range(T):
        action = np.random.choice(range(k), p=policy[:, state])
        next_state, reward, done, info = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    return trajectory

