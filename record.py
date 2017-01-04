import numpy as np
import pickle
import rl
import student

env = student.StudentDilemmaEnv()


# == q learn strategy == #

policy, log = rl.gym_q_learning(env)
# np.save("student-policy.npy", policy)
# pickle.dump(log, open("student-log.pkl", 'wb'))


# == record positive trajectories == #

nb_traj = 100
traj_length = 20
trajectories = np.ndarray((nb_traj, traj_length, 3))
for i in range(nb_traj):
    trajectories[i] = gym_mc_trajectory(env, policy, traj_length)

# np.save("student-100traj20-positive.npy", trajectories)
