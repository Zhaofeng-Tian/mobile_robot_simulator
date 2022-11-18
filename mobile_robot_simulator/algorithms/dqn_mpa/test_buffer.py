from collections import deque
import random
import numpy as np
a = [i for i in range(0,20)]
d = deque(a)
print(d)
s = np.random.choice(20, 3)
print(s)
ds = random.sample(d, 10)
print(ds)

d = deque(maxlen=10)
d.append(np.zeros((3,3)))
d.append(np.ones((3,3)))
d.append(np.ones((3,3)))
p = random.sample(d,2)
p = np.array(p)
print(p)