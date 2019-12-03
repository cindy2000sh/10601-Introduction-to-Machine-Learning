import matplotlib.pyplot as plt 
import numpy as np 

num_seq = [10, 100, 1000, 10000]
train_accuracy = [0.8328, 0.8337, 0.8608, 0.9379]
test_accuracy = [0.8325, 0.8336, 0.8565, 0.9226]

plt.plot(num_seq, train_accuracy, 'r', label = 'Train')
plt.plot(num_seq, test_accuracy, 'b', label = 'Test')
plt.xlabel('Number of Training Sequences')
plt.ylabel('Accuracy')
plt.title('Accuracy - Number of Training Sequences')
plt.legend(loc='best')
plt.show()