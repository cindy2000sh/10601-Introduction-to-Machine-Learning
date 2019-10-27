import numpy as np 
import matplotlib.pyplot as plt

#plot 1
hidden_units = [5, 20, 50, 100, 200]
Jtrain = [0.52026, 0.12705, 0.05388, 0.04658, 0.04671]
Jtest = [0.71833, 0.59215, 0.45576, 0.44054, 0.43169]

plt.plot(hidden_units, Jtrain, 'r', label = 'Training')
plt.plot(hidden_units, Jtest, 'b', label = 'Testing')
plt.xlabel('Number of hidden units')
plt.ylabel('Average Cross-Entropy Loss')
plt.title('Average Cross-Entropy - Number of hidden units')
plt.legend(loc='best')
plt.show()