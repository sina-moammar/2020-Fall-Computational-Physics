import matplotlib.pyplot as plt
from ODE_methods import *


def analytic_solution(t_s, x_0, v_0):
    A = np.sqrt(x_0 ** 2 + v_0 ** 2)
    if v_0 == 0:
        phi = np.pi / 2
    else:
        phi = np.arctan(x_0 / v_0)

    X_s = np.zeros((len(t_s), 2))
    X_s[:, 0] = A * np.sin(t_s + phi)
    X_s[:, 1] = A * np.cos(t_s + phi)
    return X_s


def x_dot(v, t, params):
    return v


def v_dot(x, t, params):
    return -x


def derivatives(X_s, t, params):
    return np.array([
        x_dot(X_s[1], t, params),
        v_dot(X_s[0], t, params)
    ])


if __name__ == "__main__":
    t_start = 0
    t_stop = 10 * (2 * np.pi)
    h = 10 ** -2
    t_s = np.arange(t_start, t_stop, h)
    initial_conditions = [1, 0]
    prev_X = analytic_solution(np.array([t_start - h]), *initial_conditions)[0]
    extended_initial_conditions = [prev_X[0]] + initial_conditions
    method_names = ['Analytic Solution', 'Euler Method', 'Euler-Cromer Method',
                    'Verlet Method', 'Velocity Verlet Method', 'Beeman Method']
    method_X_s = []

    method_X_s.append(euler(derivatives, t_start, t_stop, h, initial_conditions)[1])
    method_X_s.append(euler_cromer([x_dot, v_dot], t_start, t_stop, h, initial_conditions)[1])
    method_X_s.append(verlet(v_dot, t_start, t_stop, h, extended_initial_conditions)[1])
    method_X_s.append(velocity_verlet(v_dot, t_start, t_stop, h, initial_conditions)[1])
    method_X_s.append(beeman(v_dot, t_start, t_stop, h, extended_initial_conditions)[1])
    method_X_s.insert(0, analytic_solution(t_s, *initial_conditions))

    plt.figure(figsize=(10, 5))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$X(t)$')

    for X_s in method_X_s:
        plt.plot(t_s, X_s[:, 0])

    plt.legend(method_names)
    plt.savefig(f'images/q2_x_t_{t_start}_{t_stop}_{h}_{initial_conditions}.jpg')
    plt.show()

    for index, X_s in enumerate(method_X_s[1:]):
        plt.figure(figsize=(6, 5))
        plt.xlabel(r'$X$')
        plt.ylabel(r'$V$')
        plt.plot(X_s[:, 0], X_s[:, 1])
        plt.axis('equal')
        plt.savefig(f'images/q2_v_x_{method_names[index + 1]}_{t_start}_{t_stop}_{h}_{initial_conditions}.jpg')
        plt.show()
