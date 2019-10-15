import sys
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()

#input arguments
train_input = sys.argv[1]
valid_input = sys.argv[2]
test_input = sys.argv[3]
dict_input = sys.argv[4]
train_out = sys.argv[5]
test_out = sys.argv[6]
metrics_out = sys.argv[7]
num_epoch = int(sys.argv[8])


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def predict(thetas, features_i):
	return sigmoid(sparse_mult(thetas, features_i))

def eval_labels(labels_pred, thetas, features, n):
	for i in range(n):
		if predict(thetas, features[i]) > 0.5:
			labels_pred.append(1)
		else:
			labels_pred.append(0)

def calc_nll(thetas, features, labels):
	loss = 0
	for i in range(len(labels)):
		dot_prod = sparse_mult(thetas, features[i])
		loss = loss - int(labels[i]) * dot_prod + np.log(1 + np.exp(dot_prod))
	return loss/len(labels)

def calc_error(labels, labels_pred):
	num_mistakes = 0
	for i in range(len(labels)):
		if labels[i] != labels_pred[i]:
			num_mistakes += 1
	return num_mistakes / len(labels)
	

def preprocess(input_file):
	words = []
	labels = []
	features = []
	with open(input_file) as f:
		tsvreader = csv.reader(f, delimiter = '\t')
		for line in f:
			labels.append(int(line[0])) #changed line[0] to int(line[0])
			words.append(line[1:].split())


	for i in range(len(words)):
		d = {-1:1} #changed '-1' to -1
		for j in range(len(words[i])):
			(key, val) = words[i][j].split(':')
			d[int(key)] = int(val) #changed float to int
		features.append(d)
	return labels, features

def sparse_mult(arr, dic):
	product = 0
	for key, val in dic.items():
		product += arr[key] #changed int(key to key)
	return product

def sgd(thetas, learning_rate, features_i, label_i):
	dot_prod = sparse_mult(thetas, features_i)
	for i in features_i.keys():
		thetas[i] += learning_rate * (label_i - sigmoid(dot_prod)) #changed int(i) to i and int(label_i) to label_i
	return thetas

def write_pred_to_file(output_file, labels):
	with open(output_file, 'w') as f:
		for item in labels:
			f.write(str(item))
			f.write('\n')

m = 0
l_r = 0.1

with open(dict_input) as f:
	for i, l in enumerate(f):
		pass
	m = i + 1

theta = np.zeros(m+1)
nll_train_values = np.zeros(200)
nll_valid_values = np.zeros(200)
epoch_values = np.arange(200)

train_labels, train_features = preprocess(train_input)
valid_labels, valid_features = preprocess(valid_input)
test_labels, test_features = preprocess(test_input)
train_pred_labels = []
valid_pred_labels = []
test_pred_labels = []

time_1 = time.time() - start_time
print('Preprocessing time: ', time_1)
#training phase
def train(thetas, features, learning_rate, labels):
	for i in range(num_epoch):
		for j in range(len(train_features)):
			theta = sgd(thetas, learning_rate, features[j], labels[j])
		#nll_train_values[i] = calc_nll(theta, train_features, train_labels)
		#nll_valid_values[i] = calc_nll(theta, valid_features, valid_labels)

train(theta, train_features, l_r, train_labels)
time_2 = time.time() - time_1 - start_time
print('Training time: ', time_2)

eval_labels(train_pred_labels, theta, train_features, len(train_labels))
train_error =  calc_error(train_labels, train_pred_labels)

eval_labels(test_pred_labels, theta, test_features, len(test_labels))
test_error = calc_error(test_labels, test_pred_labels)

#eval_labels(valid_pred_labels, theta, valid_features, len(valid_labels))

time_3 = time.time() - time_2 - start_time
print('Evaluation time: ', time_3)

elapsed_time = time.time() - start_time
print('Total elapsed time: ', elapsed_time)

print('error(train): ', train_error)
print('error(test): ', test_error)
#plot average negative log-likelihood
# plt.plot(epoch_values, nll_train_values, 'r', label = 'Training')
# plt.plot(epoch_values, nll_valid_values, 'b', label = 'Validation')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Average NLL')
# plt.title('Average NLL - Number of Epochs')
# plt.legend(loc='best')
# plt.show()

write_pred_to_file(train_out, train_pred_labels)
write_pred_to_file(test_out, test_pred_labels)

#write metrics to output file
with open(metrics_out, 'w') as f:
	f.write(f'error(train): {train_error}\nerror(test): {test_error}')