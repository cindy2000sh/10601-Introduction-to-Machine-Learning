import sys
import numpy as np 
import csv
import time
import matplotlib.pyplot as plt


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def onehot(x, n, num_classes):
	onehot_encoding = np.zeros([n, num_classes])
	onehot_encoding[np.arange(n), x] = 1
	return onehot_encoding.T

def read(input_file):
	labels = []
	features = []
	with open(input_file) as f:
		csvreader = csv.reader(f, delimiter = '\t')
		for line in f:
			line = line.strip('\n')
			labels.append(int(line[0]))
			features.append(line[2:].split(','))
		num_examples = len(labels)
		num_features = len(features[0])
		return features, labels, num_examples, num_features

def calcError(ground_truth, prediction):
	wrong_pred = 0
	for i in range(len(ground_truth)):
		if ground_truth[i] != prediction[i]:
			wrong_pred += 1
	return wrong_pred / len(ground_truth)

def write_pred_to_file(output_file, labels):
	with open(output_file, 'w') as f:
		for item in labels:
			f.write(str(item))
			f.write('\n')

train_input = sys.argv[1]
test_input = sys.argv[2]
train_out = sys.argv[3]
test_out = sys.argv[4]
metrics_out = sys.argv[5]
num_epochs = int(sys.argv[6])
hidden_units = int(sys.argv[7])
init_flag = int(sys.argv[8])
learning_rate = float(sys.argv[9])

print(num_epochs, hidden_units, init_flag, learning_rate)

num_classes = 10

x_train, y_train, n_train, m_train = read(train_input)
x_test, y_test, n_test, m_test = read(test_input)

num_features = len(x_train[0])

x_train = np.asarray(x_train, dtype=float)
y_train = np.asarray(y_train, dtype=int)

x_test = np.asarray(x_test, dtype=float)
y_test = np.asarray(y_test, dtype=int)

#Initialize parameters
alpha = np.zeros([hidden_units, num_features+1])
beta = np.zeros([num_classes, hidden_units+1])
if init_flag == 1:
	alpha[:, 1:] = np.random.uniform(-0.1, 0.1, size=(hidden_units, num_features))
	beta[:, 1:] = np.random.uniform(-0.1, 0.1, size=(num_classes, hidden_units))



class NeuralNetwork:
	def __init__(self, x, y, alpha, beta, d, k, lr, flag):
		self.y = y #labels
		self.n = y.size #number of examples
		self.x = np.hstack((np.reshape(np.ones(self.n), (self.n, 1)), x)) #features, dimension: [n x m+1]
		self.alpha = alpha
		self.beta = beta
		self.m = x.shape[1] #number of features
		self.d = d #number of hidden units
		self.k = k #number of classes
		self.lr = lr #learning rate
		self.flag = flag #weight initialization identifier

	def forward(self):
		self.a = np.matmul(self.alpha, self.x.T) #a matrix, linear layer, dimension: [d x n]
		self.z_nobias = sigmoid(self.a) #z matrix without bias, sigmoid layer, dimension: [d x n]
		self.z_bias = np.vstack((np.ones(self.n), self.z_nobias)) #z matrix with bias, appending ones to the first row, dimension: [d+1 x n]
		self.b = np.matmul(self.beta, self.z_bias) #b matrix, linear layer, dimension: [k x n]
		self.exp_b = np.exp(self.b) #elementwise exponentiation of b, needed for softmax, dimension: [k x n]
		self.softmax = self.exp_b / self.exp_b.sum(axis=0) #softmax matrix/prob. distribution for k classes, dimension: [k x n]
		self.loss = - np.sum(np.sum(np.multiply(onehot(self.y, self.n, self.k), np.log(self.softmax)), axis=0))/self.n
		#self.loss = -np.sum(self.softmax.sum(axis=0))/self.n #real number


	def backward(self, i):
		self.dldb = self.softmax[:, i] - onehot(self.y, self.n, self.k)[:, i] #dimension: [k x 1], removed onehot from softmax
		self.dldb = np.reshape(self.dldb, (self.k, 1)) #dimension: [k x 1]
		self.dldbeta = np.matmul(self.dldb, np.reshape(self.z_bias[:, i].T, (1, self.d+1))) #dimension: [k x d+1]
		self.dldz = np.matmul(self.beta[:, 1:].T, self.dldb) #dimension: [d x n]
		self.dlda = np.multiply(self.dldz, np.multiply(np.reshape(self.z_nobias[:, i], (self.d, 1)), np.reshape(1 - self.z_nobias[:, i], (self.d, 1)))) #elementwise multiplication, dimension: [d x n]
		self.dldalpha = np.matmul(self.dlda, np.reshape(self.x[i, :], (1, self.m+1)))
		self.beta -= self.lr * self.dldbeta
		self.alpha -= self.lr * self.dldalpha

	def predict(self):
		self.forward()
		self.y_pred = np.argmax(self.softmax, axis=0)
		return self.alpha, self.beta, self.loss

net_train = NeuralNetwork(x_train, y_train, alpha, beta, hidden_units, num_classes, learning_rate, init_flag)

#training
loss_train = []
loss_test = []

for i in range(num_epochs):
	for j in range(n_train):
		net_train.forward()
		net_train.backward(j)
	alpha_trained, beta_trained, loss = net_train.predict()
	loss_train.append(loss)
	print('epoch ', i ' : Train Loss: ', loss)
	net_test = NeuralNetwork(x_test, y_test, alpha_trained, beta_trained, hidden_units, num_classes, learning_rate, init_flag)
	_, _, loss = net_test.predict()
	print('epoch ', i ': Test Loss: ', loss)
	loss_test.append(loss)

error_train = calcError(y_train, net_train.y_pred)
error_test = calcError(y_test, net_test.y_pred)

# write_pred_to_file(train_out, net_train.y_pred)
# write_pred_to_file(test_out, net_test.y_pred)

# #write metrics to file
# with open(metrics_out, 'w') as f:
# 	for i in range(num_epochs):
# 		f.write(f'epoch={i+1} crossentropy(train): {loss_train[i]}\n')
# 		f.write(f'epoch={i+1} crossentropy(test): {loss_test[i]}\n')
# 	f.write(f'error(train): {error_train}\n')
# 	f.write(f'error(test): {error_test}')

#plot average training and testing cross-entropy vs the number of hidden units