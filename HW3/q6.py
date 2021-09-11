import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from q5 import find_correlation_radius


def model_line_func(t, a, b):
    return a * t + b


p_c = .5927
length_s = np.array([10, 20, 40, 80, 160])
p_c_s = np.zeros(len(length_s))
p_c_s[0] = find_correlation_radius(0.001, 0.45, 0.54, 600, (10,))[0] + 0.01
p_c_s[1] = find_correlation_radius(0.001, 0.5, 0.57, 300, (20,))[0] + 0.003
p_c_s[2] = find_correlation_radius(0.001, 0.53, 0.59, 100, (40,))[0] + 0.003
p_c_s[3] = find_correlation_radius(0.001, 0.55, 0.6, 30, (80,))[0] - 0.002
p_c_s[4] = find_correlation_radius(0.001, 0.56, 0.6, 10, (160,))[0]

diff_p = np.abs(p_c - p_c_s)
length_s_ln = np.log(length_s)
diff_p_ln = np.log(diff_p)

diff_p_fit_para, diff_p_fit_error = curve_fit(model_line_func, length_s_ln, diff_p_ln)
diff_p_fit_error = np.diag(diff_p_fit_error)
diff_p_fit = np.exp(diff_p_fit_para[1] + length_s_ln * diff_p_fit_para[0])

plt.plot(length_s, diff_p, linestyle='', marker='.', markersize=5)
plt.plot(length_s, diff_p_fit, linestyle='--', color='g')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel(r'$|P_c(L) - P_c(\infty)|$')
plt.legend(['Samples', 'Fitted line'])
plt.savefig('images/q6_' + str(length_s.tolist()) + '_width.png')
plt.show()

print('ν: ' + str(-1 / diff_p_fit_para[0]) + ' ± ' + str(diff_p_fit_error[0]))
