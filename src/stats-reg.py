import os, sys
os.chdir(sys.path[0])

import json
import numpy as np
from scipy.stats import t

with open("ann_out/nn.json") as f:
  o1 = json.load(f)
with open("ann_out/lr.json") as f:
  o2 = json.load(f)

alpha = 0.05

zi1 = o1["all_losses"]
zi2 = o2["all_losses"]
egen1 = o1["E_gen"]
egen2 = o2["E_gen"]

n = len(zi1)

sigma2 = 1 / (n*(n-1)) * sum((zi1[i] - zi2[i])**2 for i in range(n))
sigma = np.sqrt(sigma2)
zhat = egen1 - egen2
zL = t.ppf(alpha/2, n-1, zhat, sigma)
zU = t.ppf(1-alpha/2, n-1, zhat, sigma)
p = 2 * t.cdf(-np.abs(zhat), n-1, 0, sigma)

print("sigma =", sigma)
print("zL =", zL)
print("zU =", zU)
print("p  =", p)

