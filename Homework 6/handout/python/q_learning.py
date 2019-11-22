from environment import MountainCar
import sys
import numpy as np

def write_weights_to_file(output_file, weights, bias):
	with open(output_file, 'w') as f:
		f.write(str(bias))
		f.write('\n')
		for i in range(weights.shape[0]):
			for j in range(weights.shape[1]):
				f.write(str(weights[i, j]))
				f.write('\n')

def write_rewards_to_file(output_file, rewards):
	with open(output_file, 'w') as f:
		for item in rewards:
			f.write(str(item))
			f.write('\n')

#calculates and returns q for a given state, action pair 
def compute_q(action, state, weights, bias, ep, count):
	q = np.float64(0)
	for ind, val in state.items():
		q += np.float64(val) * weights[ind, action]
	q += bias
	return q

#calculates maximum q value and returns the corresponding action index
def calc_max_q(state, weights, bias, ep, count):
	q_values = []
	for i in range(num_actions):
		q_values.append(compute_q(i, state, weights, bias, ep, count))
	#np.argmax returns the smaller index of the max value, which satisfies the breaking-ties requirement
	# if ep == 0 and count <= 2:
	# 	print(q_values)
	return np.argmax(q_values)

#chooses action based on epsilon greedy policy
def choose_action(state, weights, bias, epsilon, i, count):
	#generate random number
	prob = np.random.random()
	if prob <= epsilon:
		chosen_action = np.random.choice(num_actions)
	else:
		chosen_action = calc_max_q(state, weights, bias, i, count)
	return chosen_action

#args
mode = str(sys.argv[1])
weight_out = sys.argv[2]
returns_out = sys.argv[3]
num_episodes = int(sys.argv[4])
max_iter = int(sys.argv[5])
epsilon = np.float64(sys.argv[6])
gamma = np.float64(sys.argv[7])
alpha = np.float64(sys.argv[8])

num_actions = 3

car = MountainCar(mode)

#initialize parameters
weights = np.zeros([car.state_space, num_actions], dtype=np.float64)

bias = np.float64(0)
rewards = []

for i in range(num_episodes):
	count = 1
	state = car.reset()
	done = False
	rewards.append(0)
	while done == False and count <= max_iter:
		action = choose_action(state, weights, bias, epsilon, i, count)
		q_current = compute_q(action, state, weights, bias, i, count)
		new_state, reward, done = car.step(action)
		rewards[i] += reward
		max_q = calc_max_q(new_state, weights, bias, i, count)
		q_new = compute_q(max_q, new_state, weights, bias, i, count) #changed 1st argument from 
		#folding alpha into td_error
		td_error = alpha * (q_current - (reward + gamma * q_new))
		for ind, val in state.items():
			weights[ind, action] -= td_error * np.float64(val)
		bias -= td_error
		# if i == 0 and count <= 4:
		# 	print('action = ', action)
		# 	print('q_current = ', q_current)
		# 	print('q_new = ', q_new)
		state = new_state
		count += 1

write_weights_to_file(weight_out, weights, bias)
write_rewards_to_file(returns_out, rewards)