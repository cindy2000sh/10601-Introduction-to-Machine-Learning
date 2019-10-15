import matplotlib.pyplot as plt

a = [1, 2, 3, 4]
b = [2,9,10,15]
c = [4,18,25, 40]

plt.plot(a, b, 'r', label='curve1')
plt.plot(a, c, 'g', label='curve2')
plt.ylabel('some numbers')
plt.title('the plot')
plt.legend(loc='best')
plt.show()