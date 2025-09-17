import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-5, 5, 100)
y = sigmoid(x)

fig, ax = plt.subplots(figsize=(4, 2))
ax.set_xticks([])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.set_yticks([0.5, 1.0])
ax.set_yticklabels(["0.5", "1"])
ax.annotate("0", xy=(0, 0), xytext=(-0.5, -0.1), textcoords='data', ha='center')

ax.plot(x, y, color="black", linewidth=0.8)

plt.savefig('sigmoid_function.png', dpi=300)

plt.show()
