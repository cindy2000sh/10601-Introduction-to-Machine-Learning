from __future__ import print_function
import sys
import csv
import time
from math import exp, log

def read_dict(dict_file):
    res = {}
    try:
        with open(dict_file, 'r') as file:
            for line in file:
                (word, index) = line.strip().split(' ')
                index = int(index)
                res[word] = index
    except IOError as ioerror:
        print(dict_file + " open error")
    #print res
    return res

def read_data(inputfile):
    data = []
    labels = []
    res = []
    try:
        with open(inputfile, 'r') as infile:
            tsv_reader = csv.reader(infile, delimiter = '\t')
            for row in tsv_reader:
                data.append(row)
    except IOError as ioerror:
        print(inputfile + "open error")
    for row in data:
        labels.append(int(row[0]))
        x_dict = {};
        x = row[1:]
        #print (x[2]); break
        for word in x:
            sp = word.split(':')
            #print sp;print(len(sp));break
            if len(sp) < 2:
                break
            x_dict[int(sp[0])] = int(sp[1])
        res.append(x_dict)
    return res, labels

def sigmoid(x):
    return 1.0 /(1.0 + exp(-x))

def multiply(theta, x):
    u = 0.0
    for i in x.keys():
        u += theta[i]
    u += theta[-1];
    return u

def SGD(theta, rate, x, y):
    u = multiply(theta, x)
    h = sigmoid(u)
    for i in x.keys():
        theta[i] += rate * (y - h)
    theta[-1] += rate * (y - h)
    return theta
    
def prediction(theta, x):
    u = multiply(theta, x)
    if sigmoid(u) > 0.5:
        return 1
    else:
        return 0

def cal_error(labels, predict):
    n = float(len(labels))
    error = 0.0
    for i in range(len(labels)):
        if labels[i] != predict[i]:
            error += 1.0
    return error / n

def avg_neg_log_likelihood(data, labels, theta):
    n = len(labels)
    s = 0.0
    for i in range(n):
        u = multiply(theta, data[i])
        p = sigmoid(u)
        if labels[i] == 0:
            p = 1 - p
        s -= log(p)
    return s / n


#Main
start = time.time()
if __name__ == '__main__':
  train_in = sys.argv[1]
  val_in = sys.argv[2]
  test_in = sys.argv[3]
  dict_in = sys.argv[4]
  # train_out = sys.argv[5]
  # test_out = sys.argv[6]
  # metrics_out = sys.argv[7]
  # num_epoch = int(sys.argv[8])

num_epoch = 60

_dict = read_dict(dict_in)
train_data, train_labels = read_data(train_in)
test_data, test_labels = read_data(test_in)
#with open("temp1.txt", 'w') as temp:
#    print(test_labels, file = temp)


print(len(train_data))
# train model
theta = [0.0] * (len(_dict) + 1)
for i in range(num_epoch):
    for j in range(len(train_data)):
        theta = SGD(theta, 0.1, train_data[j], train_labels[j])
    #print(i, end = '\t')
    #print(avg_neg_log_likelihood(data, labels, theta), end = '\t')
    #print(avg_neg_log_likelihood(val_data, val_labels, theta))

# predict training data
train_out_labels = []
for j in range(len(train_data)):
    predict = prediction(theta, train_data[j])
    train_out_labels.append(predict)
# try:
#     with open(train_out, 'w') as outfile:
#         for i in train_out_labels:
#             print(i, file = outfile)
# except IOError as ioerror:
#     print(train_out + "open error")

# predict test data
test_out_labels = []
for j in range(len(test_data)):
    predict = prediction(theta, test_data[j])
    test_out_labels.append(predict)
# try:
#     with open(test_out, 'w') as outfile:
#         for i in test_out_labels:
#             print(i, file = outfile)
# except IOError as ioerror:
#     print(test_out + "open error")

# write metrics
train_error = cal_error(train_labels, train_out_labels)
test_error = cal_error(test_labels, test_out_labels)
# try:
#     with open(metrics_out, 'w') as outfile:
#         print("error(train): ", end = '', file = outfile)
#         print('%.6f' % train_error, file = outfile)
#         print("error(test): ", end = '', file = outfile)
#         print('%.6f' % test_error, file = outfile)
# except IOError as ioerror:
#     print(metrics_out + "open error")

end = time.time() - start
print(train_error)
print(test_error)
print(end)
#print("\nRuntime:"+str(end-start)+'s')