import sys
import csv
import math

class Node:
	def __init__(self, key, depth):
		self.left = None
		self.right = None
		self.val = key
		self.max_ig_ind = None
		self.depth = depth
		self.label = [0, 0]
		self.pred = None

#read tsv file and arrange data into a list of lists
def readTsv(input):
	with open(input) as tsv_file:
		tsvreader = csv.reader(tsv_file, delimiter = '\t')
		count = 0
		data = []
		for line in tsvreader:
			if count == 0:
				for i in range(len(line)):
					data.append([])
			count += 1
			for i in range(len(line)):
				data[i].append(line[i])
		return data

def calcEntropy(numb_of_label_1, numb_of_label_2):
	total_examples = numb_of_label_1 + numb_of_label_2
	p1 = numb_of_label_1 / total_examples
	p2 = numb_of_label_2 / total_examples

	#avoiding log2(0) since log2(1) = 0 as well
	if p1 == 0:
		p1 = 1
	elif p2 == 0:
		p2 = 1

	entropy = - p1 * math.log2(p1) - p2 * math.log2(p2)

	return entropy

#p_attr: possibility of an attribute, p_cond_lab: conditional probability
def calcInfoGain(entropy, n_val1, n_val2, n_cond1, n_cond2, n_cond3, n_cond4):
	pA = n_val1 / (n_val1 + n_val2)
	pB = n_val2 / (n_val1 + n_val2)
	if n_cond1 == 0 and n_cond2 == 0:
		p1 = 1
		p2 = 1
	else:
		p1 = n_cond1 / (n_cond1 + n_cond2)
		p2 = n_cond2 / (n_cond1 + n_cond2)
	if n_cond3 == 0 and n_cond4 == 0:
		p3 = 1
		p4 = 1
	else:
		p3 = n_cond3 / (n_cond3 + n_cond4)
		p4 = n_cond4 / (n_cond3 + n_cond4)
	if p1 == 0:
		p1 = 1
	if p2 == 0:
		p2 = 1
	if p3 == 0:
		p3 = 1
	if p4 == 0:
		p4 = 1
	cond_entropy = pA * (- p1 * math.log2(p1) - p2 * math.log2(p2)) + pB * (- p3 * math.log2(p3) - p4 * math.log2(p4))
	info_gain = entropy - cond_entropy

	return info_gain



def preorder(root):
	if root:
		symb = '| '
		if root.depth == 0:
			print('[',root.label[0],lab1,'/',root.label[1],lab2,']')
		if root.max_ig_ind != None:
			print(root.depth*symb,root.val[root.max_ig_ind][0],'=',root.val[root.max_ig_ind][1],': [',root.label[0],lab1,'/',root.label[1],lab2,']')
		preorder(root.left)
		preorder(root.right)


def findMaxInfoGain(data):
	numb_of_label_1 = 0
	numb_of_label_2 = 0
	n_val1 = []
	n_val2 = []
	n_cond1 = []
	n_cond2 = []
	n_cond3 = []
	n_cond4 = []
	info_gain = []
	count = 0
	for i in range(1, len(data[0])):
		if data[-1][i] == lab1:
			numb_of_label_1 += 1
		else:
			numb_of_label_2 += 1
		for j in range(len(data) - 1):
			if count == 0:
				n_val1.append(0)
				n_val2.append(0)
				n_cond1.append(0)
				n_cond2.append(0)
				n_cond3.append(0)
				n_cond4.append(0)
				info_gain.append(0)
		count += 1
		for j in range(len(data) - 1):
			if data[j][i] == val1:
				n_val1[j] += 1 #N(A = 0)
				if data[-1][i] == lab1:
					n_cond1[j] += 1 #N(Y = 0|A = 0)
				else:
					n_cond2[j] += 1 #N(Y = 1|A = 0)
			else:
				n_val2[j] += 1 #N(A = 1)
				if data[-1][i] == lab1:
					n_cond3[j] += 1 #N(Y = 0|A = 1)
				else:
					n_cond4[j] += 1 #N(Y = 1|A = 1)

	entropy = calcEntropy(numb_of_label_1, numb_of_label_2)
	flag = False
	for j in range(len(data) - 1):
		info_gain[j] = calcInfoGain(entropy, n_val1[j], n_val2[j], n_cond1[j], n_cond2[j], n_cond3[j], n_cond4[j])
		if info_gain[j] > 0:
			flag = True
	if flag:
		max_ig = info_gain.index(max(info_gain))
	else:
		max_ig = -1
	return max_ig

def splitData(data, max_info_gain_ind, val):
	split_list = []
	for i in range(len(data)):
		split_list.append([])
		split_list[i].append(data[i][0])
	for i in range(1, len(data[0])):
		for j in range(len(data)):
			if data[max_info_gain_ind][i] == val:
				split_list[j].append(data[j][i])
	return split_list

def train(root, max_depth):
	max_info_gain = findMaxInfoGain(root.val)
	if max_info_gain < 0 or root.depth > max_depth:
		return
	if root.depth == 0:
		na = 0
		nb = 0
		for i in range(1, len(root.val[0])):
			if root.val[-1][i] == lab1:
				na += 1
			else:
				nb += 1
		root.label[0] = na
		root.label[1] = nb
		if na >= nb:
			root.pred = lab1
		else:
			root.pred = lab2
	n1 = n2 = n3 = n4 = 0
	root.left = Node(splitData(root.val, max_info_gain, val1), root.depth + 1)
	root.right = Node(splitData(root.val, max_info_gain, val2), root.depth + 1)
	root.left.max_ig_ind = max_info_gain
	root.right.max_ig_ind = max_info_gain
	for i in range(1, len(root.left.val[0])):
		if root.left.val[-1][i] == lab1:
			n1 += 1
		else:
			n2 += 1
	for i in range(1, len(root.right.val[0])):
		if root.right.val[-1][i] == lab1:
			n3 += 1
		else:
			n4 += 1
	if n1 >= n2:
		root.left.pred = lab1
	else:
		root.left.pred = lab2
	if n3 >= n4:
		root.right.pred = lab1
	else:
		root.right.pred = lab2
	root.left.label[0] = n1
	root.left.label[1] = n2
	root.right.label[0] = n3
	root.right.label[1] = n4
	train(root.left, max_depth)
	train(root.right, max_depth)

def traverse(root, dataset, ind, r):
	if root.left and dataset[root.left.max_ig_ind][ind] == val1:
			traverse(root.left, dataset, ind, r)
	elif root.right and dataset[root.right.max_ig_ind][ind] == val2:
			traverse(root.right, dataset, ind, r)
	else:
		r.append(root.pred)
	return


def predict(dataset, root, r):
	for i in range(1, len(dataset[0])): 
		if dataset[root.left.max_ig_ind][i] == val1:
			traverse(root.left, dataset, i, r)
		else:
			traverse(root.right, dataset, i, r)
	return

def calcError(ground_truth, prediction):
	ground_truth = ground_truth[-1][1:]
	wrong_pred = 0
	for i in range(len(prediction)):
		if ground_truth[i] != prediction[i]:
			wrong_pred += 1
	error = wrong_pred / len(prediction)
	return error


#main program
data = readTsv(sys.argv[1])
test = readTsv(sys.argv[2])
max_depth = int(sys.argv[3]) - 1

#set reference attribute value
for i in range(1, len(data[0])):
	if i == 1:
		val1 = data[0][i]
	if data[0][i] != val1:
		val2 = data[0][i]
		break

#set reference label value
for i in range(1, len(data[0])):
	if i == 1:
		lab1 = data[-1][i]
	if data[-1][i] != lab1:
		lab2 = data[-1][i]
		break

train_results = []
test_results = []

#if handling the possibility of having 0 depth, just return majority vote of the root node
if max_depth == - 1:
	numb_of_label_1 = 0
	numb_of_label_2 = 0
	for i in range(1, len(data[0])):
		if data[-1][i] == lab1:
			numb_of_label_1 += 1
		else:
			numb_of_label_2 += 1
	if numb_of_label_1 >= numb_of_label_2:
		train_results = [lab1 for i in range(len(data[0]) - 1)]
		test_results = [lab1 for i in range(len(test[0]) - 1)]
	else:
		train_results = [lab2 for i in range(len(data[0]) - 1)]
		test_results = [lab2 for i in range(len(test[0]) - 1)]
else:
	root = Node(data, 0)

	train(root, max_depth)
	preorder(root)

	predict(data, root, train_results)
	predict(test, root, test_results)

train_error = calcError(data, train_results)
test_error = calcError(test, test_results)

with open(sys.argv[4], 'w') as train_predictions:
    for item in train_results:
        train_predictions.write(item)
        train_predictions.write('\n')

with open(sys.argv[5], 'w') as test_predictions:
    for item in test_results:
        test_predictions.write(item)
        test_predictions.write('\n')

with open(sys.argv[6], 'w') as metrics_file:
	metrics_file.write(f'error(train): {train_error}\nerror(test): {test_error}')