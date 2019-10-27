import numpy as np 
import matplotlib.pyplot as plt

#plot 1
hidden_units = [5, 20, 50, 100, 200]
Jtrain = [0.392888, 0.12521, 0.09811, 0.08791, 0.07626]
Jtest = [1.20876, 1.37414, 1.40446, 1.52422, 1.59825]

plt.plot(hidden_units, Jtrain, 'r', label = 'Training')
plt.plot(hidden_units, Jtest, 'b', label = 'Testing')
plt.xlabel('Number of hidden units')
plt.ylabel('Average Cross-Entropy Loss')
plt.title('Average Cross-Entropy - Number of hidden units')
plt.legend(loc='best')
plt.show()