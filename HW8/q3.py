import numpy as np
import matplotlib.pyplot as plt


def rc_circuit(q, t, params):
    return 1 - q


def analytic_solution(t_s):
    return 1 - np.exp(-t_s)


def numerical_integration(derivatives, t_start, t_stop, h, initial_conditions, params=()):
    t_s = np.arange(t_start, t_stop, h)
    variable_numbers = int(len(initial_conditions) / 2)
    X_s = np.zeros((len(t_s) + 1, variable_numbers))
    X_s[0] = initial_conditions[:variable_numbers]
    X_s[1] = initial_conditions[variable_numbers:]
    h_double = 2 * h

    for t_index, t in enumerate(t_s[:-1]):
        X_s[t_index + 2] = X_s[t_index] + derivatives(X_s[t_index + 1], t_s[t_index + 1], params) * h_double

    return t_s, X_s[1:]


if __name__ == "__main__":
    q_scale = 10 ** -5
    t_scale = 3 * (10 ** -3)
    t_start = 0
    t_stop = 25 * (10 ** -3) / t_scale
    h = 10 ** -1
    initial_conditions = [0]
    extended_initial_conditions = [analytic_solution(t_start - h)] + initial_conditions

    t_s_1, q_s_1 = numerical_integration(rc_circuit, t_start, t_stop, h, extended_initial_conditions)
    analytic_q_s = analytic_solution(t_s_1)

    h = 7 * 10 ** -2
    initial_conditions = [0]
    extended_initial_conditions = [analytic_solution(t_start - h)] + initial_conditions
    t_s_2, q_s_2 = numerical_integration(rc_circuit, t_start, t_stop, h, extended_initial_conditions)

    plt.plot(t_s_1, q_s_1[:, 0])
    plt.plot(t_s_2, q_s_2[:, 0])
    plt.plot(t_s_1, analytic_q_s)
    plt.xlabel(r'$t$ ($\times' + str(t_scale) + '$) s')
    plt.ylabel(r'$Q(t)$ $(\times' + str(q_scale) + ')$ C')
    plt.legend([r'$h = 0.1$', r'$h = 0.07$', 'Analytic Solution'])
    plt.savefig(f'images/q3_{t_start}_{t_stop}.jpg')
    plt.show()
