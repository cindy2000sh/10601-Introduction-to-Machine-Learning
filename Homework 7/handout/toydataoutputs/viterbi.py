import sys
import csv
import numpy as np 

test_data = sys.argv[1]
index_to_word = sys.argv[2]
index_to_tag = sys.argv[3]
hmm_prior = sys.argv[4]
hmm_emit = sys.argv[5]
hmm_trans = sys.argv[6]
predicted_file = sys.argv[7]
metric_file = sys.argv[8]

def readWordsnTags(input_file):
	empty_dict = {}
	empty_dict2 = {}
	i = 0
	with open(input_file) as f:
		csvreader = csv.reader(f)
		for line in f:
			empty_dict[line.strip('\n')] = i
			empty_dict2[i] = line.strip('\n')
			i += 1
	return empty_dict, empty_dict2

def readSequences(input_file, word_map, tag_map):
	words = []
	tags = []
	empty_list = []
	total_words = 0
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
				total_words += 1
			i += 1
	return words, tags, total_words

def init_w(obs, pointers, p, a, b, num_tags):
	w = []
	for i in range(num_tags):
		w.append(p[i] * b[i][obs])
		pointers[0][i] = i
	return w

def recurse(seq, pos, c, a, b, w, pointers, num_tags):
	pos += 1
	if pos == len(seq):
		return np.argmax(w)
	else:
		obs = seq[pos]
		w_next = []
		for i in range(num_tags):
			w_temp = []
			for j in range(num_tags):
				w_temp.append(b[i][obs] * a[j][i] * w[j])
			w_next.append(np.max(w_temp))
			pointers[pos][i] = np.argmax(w_temp)
		return recurse(seq, pos, c, a, b, w_next, pointers, num_tags)

def predict(last_pred, backp, seq_length):
	pred = []
	pred.append(last_pred)
	prev_pred = last_pred
	for i in range(seq_length - 1, 0, -1):
		pred.append(backp[i][int(prev_pred)])
		prev_pred = backp[i][int(prev_pred)]
	pred = list(reversed(pred))
	return pred

def eval(ground_truth, pred, total_words):
	num_corr = 0
	for true_seq, pred_seq in zip(ground_truth, pred):
		for true_val, pred_val in zip(true_seq, pred_seq):
			if true_val == pred_val:
				num_corr += 1
	return num_corr / total_words

def writePred(pred, word_seq, ind2word, ind2tag):
	with open(predicted_file, 'w') as f:
		for sent, seq in zip(word_seq, pred):
			for word, ind in zip(sent, seq):
				f.write(ind2word[word] + '_' + ind2tag[ind] + ' ')
			f.write('\n')

word2ind, ind2word = readWordsnTags(index_to_word)
tag2ind, ind2tag = readWordsnTags(index_to_tag)
num_words = len(word2ind)
num_tags = len(tag2ind)

word_seq, tag_seq, total_words = readSequences(test_data, word2ind, tag2ind)
p = np.loadtxt(hmm_prior)
a = np.loadtxt(hmm_trans)
b = np.loadtxt(hmm_emit)

preds = []
c = 0

for seq in word_seq:
	pointers = np.zeros([len(seq), num_tags])
	pos = 0
	w = init_w(seq[0], pointers, p, a, b, num_tags)
	last_pred = recurse(seq, pos, c, a, b, w, pointers, num_tags)
	preds.append(predict(last_pred, pointers, len(seq)))
	c += 1

accuracy = eval(tag_seq, preds, total_words)
writePred(preds, word_seq, ind2word, ind2tag)

with open(metric_file, 'w') as f:
	f.write('Accuracy: ' + str(accuracy))