import sys
import csv
import numpy as np 

train_data = sys.argv[1]
index_to_word = sys.argv[2]
index_to_tag = sys.argv[3]
hmm_prior = sys.argv[4]
hmm_emit = sys.argv[5]
hmm_trans = sys.argv[6]

def readWordsnTags(input_file):
	empty_dict = {}
	i = 0
	with open(input_file) as f:
		csvreader = csv.reader(f)
		for line in f:
			empty_dict[line.strip('\n')] = i
			i += 1
	return empty_dict

def readSequences(input_file, word_map, tag_map):
	words = []
	tags = []
	empty_list = []
	with open(input_file) as f:
		csvreader = csv.reader(f)
		i = 0
		for line in f:
			words.append([])
			tags.append([])
			temp = line.strip(' ')
			temp = temp.strip('\n')
			temp = temp.split(' ')
			for val in temp:
				w, t = val.split('_')
				words[i].append(word_map[w])
				tags[i].append(tag_map[t])
			i += 1
	return words, tags

words = readWordsnTags(index_to_word)
tags = readWordsnTags(index_to_tag)
num_words = len(words)
num_tags = len(tags)

word_seq, tag_seq = readSequences(train_data, words, tags)

#Get initial probability matrix
p_count = np.zeros([num_tags])
init_p = np.zeros([num_tags, 1])

a_count = np.zeros([num_tags, num_tags])
b_count = np.zeros([num_tags, num_words])

for w_seq, t_seq in zip(word_seq, tag_seq):
	p_count[t_seq[0]] += 1
	for i in range(len(w_seq)):
		b_count[t_seq[i]][w_seq[i]] += 1
		if i < len(w_seq) - 1:
			a_count[t_seq[i]][t_seq[i+1]] += 1

p_sum = np.sum(p_count) + num_tags
init_p = np.divide(p_count + 1, p_sum)

#Get transition probability matrix
trans_p = np.zeros([num_tags, num_tags])
a_sum = np.sum(a_count, axis=1) + num_tags
a_sum = a_sum[np.newaxis]
a_sum = a_sum.T
trans_p = np.divide(a_count + 1, a_sum)

#Get emission probability matrix
emis_p = np.zeros([num_tags, num_words])
b_sum = np.sum(b_count, axis=1) + num_words
b_sum = b_sum[np.newaxis]
b_sum = b_sum.T
emis_p = np.divide(b_count + 1, b_sum)

np.savetxt(hmm_prior, init_p)
np.savetxt(hmm_emit, emis_p)
np.savetxt(hmm_trans, trans_p)