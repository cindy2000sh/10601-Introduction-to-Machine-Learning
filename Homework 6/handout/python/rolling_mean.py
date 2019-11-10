import sys
import numpy as np
import matplotlib.pyplot as plt

def read_txt(input_file):
	rewards = []
	with open(input_file, 'r') as f:
		rewards = f.read().splitlines() 
	return rewards 

def roll_mean(rewards, subset):
	n = len(rewards)
	mean = []
	for i in range(n-subset):
		mean.append(np.sum(rewards[i:i+subset])/subset)
	return mean

def pad(rewards, subset):
	threshold = np.floor_divide(subset, 2)
	front_val = rewards[0]
	back_val = rewards[-1]
	if 2 * threshold < subset:
		front_n = threshold + 1
		back_n = threshold
	for i in range(front_n):
		rewards.insert(0, front_val)
		if i < back_n:
			rewards.append(back_val)
	return rewards

raw_data = sys.argv[1]
tile_data = sys.argv[2]

subset_length = 25

raw_rew = read_txt(raw_data)
tile_rew = read_txt(tile_data)

raw_rewards = np.asarray(raw_rew, dtype=np.float64)
tile_rewards = np.asarray(tile_rew, dtype=np.float64)

padded_raw = raw_rew
padded_tile = tile_rew

padded_raw = np.asarray(pad(padded_raw, subset_length), dtype=np.float64)
padded_tile = np.asarray(pad(padded_tile, subset_length), dtype=np.float64)


raw_n = len(raw_rewards)
tile_n = len(tile_rewards)

raw_ep_length = np.arange(raw_n+1)[1:]
tile_ep_length = np.arange(tile_n+1)[1:]

raw_mean = np.asarray(roll_mean(padded_raw, subset_length), dtype=np.float64)
tile_mean = np.asarray(roll_mean(padded_tile, subset_length), dtype=np.float64)

plt.plot(raw_ep_length, raw_rewards, 'r', label = 'Total Rewards')
plt.plot(raw_ep_length, raw_mean, 'b', label = 'Rolling Average')
plt.xlabel('Number of Episodes')
plt.ylabel('Rewards')
plt.title('Rewards - Number of episodes for Raw Features')
plt.legend(loc='best')
plt.show()

plt.plot(tile_ep_length, tile_rewards, 'r', label = 'Total Rewards')
plt.plot(tile_ep_length, tile_mean, 'b', label = 'Rolling Average')
plt.xlabel('Number of Episodes')
plt.ylabel('Rewards')
plt.title('Rewards - Number of Episodes for Tile Features')
plt.legend(loc='best')
plt.show()