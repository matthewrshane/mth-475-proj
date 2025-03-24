import gmpy2
import numpy as np
import matplotlib.pyplot as plt

eps_32 = float('1.1920929e-7')
eps_64 = float('2.220446049250313e-16')

# time over number of time steps
dt = np.divide(1.0, [1, 10, 100, 1000, 10000, 100000, 1000000])
# dt = np.divide(1.0, [10, 20, 40, 80, 160, 320])

mixed_precision_outputs = np.loadtxt("../vanderpol/implicit_midpoint/64_32_sol.txt", dtype=np.float64, delimiter=',')
full_precision_outputs = np.loadtxt("../vanderpol/implicit_midpoint/64_64_sol.txt", dtype=np.float64, delimiter=',')
ref_outputs = np.loadtxt("../vanderpol/implicit_midpoint/128_128_sol.txt", dtype=np.float64, delimiter=',')

# gmpy2.get_context().precision = 128
# x = gmpy2.mpfr(ref_outputs[0])
# y = gmpy2.mpfr(ref_outputs[1])

mixed_precision_errors = np.linalg.norm(np.subtract(ref_outputs, mixed_precision_outputs), 2, 1)
full_precision_errors = np.linalg.norm(np.subtract(ref_outputs, full_precision_outputs), 2, 1)

plt.rcParams['text.usetex'] = True
plt.loglog(dt, mixed_precision_errors, 'o:r')
plt.loglog(dt, full_precision_errors, ':b')
# plt.plot(dt, [eps_32] * len(dt), 'r--')
# plt.plot(dt, [eps_64] * len(dt), 'b--')
plt.title('L2 Errors: Mixed vs. Full Precision')
plt.xlabel(r"$\log_{10}(\Delta t)$")
plt.ylabel(r"$\log_{10}($Error$)$")
plt.show()