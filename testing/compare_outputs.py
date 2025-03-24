import math
import numpy as np
import matplotlib.pyplot as plt

ben_outputs = np.loadtxt("../../mpark-julia/vanderpol/impmid/output.txt", dtype=np.float64, delimiter='\t')
my_outputs = np.loadtxt("../vanderpol/implicit_midpoint/output.txt", dtype=np.float64, delimiter='\t')

diffs = np.subtract(ben_outputs, my_outputs)
errs = np.zeros(len(diffs))
for i in range(len(diffs)):
    errs[i] = math.sqrt(math.pow(diffs[i][0], 2) + math.pow(diffs[i][1], 2))

max_errs = np.zeros(len(diffs))
for i in range(len(diffs)):
    max_errs[i] = max(diffs[i][0], diffs[i][1])

print(np.max(np.abs(errs)))
print(np.max(np.abs(max_errs)))

# diffs1 = diffs[:, 0]
# diffs2 = diffs[:, 1]
# plt.plot(diffs1, 'r')
# plt.plot(diffs2, 'b')

# plt.plot(ben_outputs[:, 0], ben_outputs[:, 1], 'r--')
plt.plot(my_outputs[:, 0], my_outputs[:, 1], 'b')

plt.show()