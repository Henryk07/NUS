from asyncio import FastChildWatcher
from cgi import test
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm


y = [1, 10, 18, 19, 20, 67]
plt.figure()
plt.plot(y)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")

y = [-11, -19, 177, -1000]
plt.plot(y)
plt.show()
