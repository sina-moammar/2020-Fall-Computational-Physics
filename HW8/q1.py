import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ODE_methods import euler


def model_linear_func(t, a, b):
    return a * t + b


def rc_circuit(q, t, params):
    return 1 - q


def analytic_solution(t_s):
    return 1 - np.exp(-t_s)


def find_error(derivatives_func, analytic_func, t, h_start, h_stop, h_samples, initial_conditions):
    h_s = np.exp(np.linspace(np.log(h_start), np.log(h_stop), h_samples))
    error_s = np.zeros(len(h_s))

    for h_index, h in enumerate(h_s):
        t_s, q_s = euler(derivatives_func, 0, t, h, initial_conditions)
        q_analytic = analytic_func(t_s[-1:])[0]
        error_s[h_index] = np.abs((q_analytic - q_s[-1]) / q_analytic)

    return h_s, error_s


if __name__ == "__main__":
    q_scale = 10 ** -5
    t_scale = 3 * (10 ** -3)
    t_start = 0
    t_stop = 21 * (10 ** -3) / t_scale
    h = 10 ** -4
    initial_conditions = [0]

    t_s, q_s = euler(rc_circuit, t_start, t_stop, h, initial_conditions)
    analytic_q_s = analytic_solution(t_s)

    plt.plot(t_s, q_s[:, 0])
    plt.plot(t_s, analytic_q_s)
    plt.xlabel(r'$t$ ($\times' + str(t_scale) + '$) s')
    plt.ylabel(r'$Q(t)$ $(\times' + str(q_scale) + ')$ C')
    plt.legend(['Euler Method', 'Analytic Solution'])
    plt.savefig(f'images/q1_euler_analytic_{t_start}_{t_stop}_{h}.jpg')
    plt.show()

    h_start = 10 ** -1
    h_stop = 10 ** -5
    h_samples = 100
    h_s, error_s = find_error(rc_circuit, analytic_solution, t_stop, h_start, h_stop, h_samples, initial_conditions)
    h_s_log = np.log(h_s)
    error_s_log = np.log(error_s)
    fit_para, fit_error = curve_fit(model_linear_func, h_s_log, error_s_log)
    fit_error = np.diag(fit_error)
    fit = np.exp(fit_para[0] * h_s_log + fit_para[1])
    plt.plot(h_s, error_s, linestyle='', marker='.', markersize=5)
    plt.plot(h_s, fit, linestyle='--', color='g')
    plt.legend(['Samples', 'Fitted Line'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$h$ ($\times' + str(t_scale) + '$) s')
    plt.ylabel(r'$\Delta Q / Q$')
    plt.savefig(f'images/q1_euler_error_{h_start}_{h_stop}_{h_samples}_{t_stop}.jpg')
    plt.show()

    print(f'{fit_para[0]} ± {fit_error}')
