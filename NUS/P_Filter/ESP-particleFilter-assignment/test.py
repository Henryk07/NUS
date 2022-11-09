import numpy as np
landmark = (1, 2, 3, 4)
a = np.sqrt((2)**2 + (4)**2) + \
    (5+9*12)
a_list = list(range(5000))
print(range(len(a_list)))
indices = np.random.choice(range(len(a_list)), replace=False, size=50)

print(np.array(a_list)[indices.astype(int)])
