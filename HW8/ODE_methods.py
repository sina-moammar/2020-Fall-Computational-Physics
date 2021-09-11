import numpy as np


def euler(derivatives, t_start, t_stop, h, initial_conditions, params=()):
    t_s = np.arange(t_start, t_stop, h)
    X_s = np.zeros((len(t_s), len(initial_conditions)))
    X_s[0] = initial_conditions

    for t_index, t in enumerate(t_s[:-1]):
        X_s[t_index + 1] = X_s[t_index] + derivatives(X_s[t_index], t, params) * h

    return t_s, X_s


def euler_cromer(derivatives, t_start, t_stop, h, initial_conditions, params=()):
    t_s = np.arange(t_start, t_stop, h)
    X_s = np.zeros((len(t_s), len(initial_conditions)))
    X_s[0] = initial_conditions

    for t_index, t in enumerate(t_s[:-1]):
        X_s[t_index + 1][1] = X_s[t_index][1] + derivatives[1](X_s[t_index][0], t, params) * h
        X_s[t_index + 1][0] = X_s[t_index][0] + derivatives[0](X_s[t_index + 1][1], t, params) * h

    return t_s, X_s


def verlet(accelerate_func, t_start, t_stop, h, initial_conditions, params=()):
    t_s = np.arange(t_start, t_stop, h)
    X_s = np.zeros((len(t_s), len(initial_conditions) - 1))
    X_s[0] = initial_conditions[1:]
    prev_x = initial_conditions[0]
    h_square = h ** 2
    h_double = 2 * h

    for t_index, t in enumerate(t_s[:-1]):
        new_x = 2 * X_s[t_index][0] - prev_x + accelerate_func(X_s[t_index][0], t, params) * h_square
        X_s[t_index + 1][1] = (new_x - prev_x) / h_double
        X_s[t_index + 1][0], prev_x = new_x, X_s[t_index][0]

    return t_s, X_s


def velocity_verlet(accelerate_func, t_start, t_stop, h, initial_conditions, params=()):
    t_s = np.arange(t_start, t_stop, h)
    X_s = np.zeros((len(t_s), len(initial_conditions)))
    X_s[0] = initial_conditions
    h_half = h / 2
    current_accelerate_diff = None
    next_accelerate_diff = accelerate_func(X_s[0][0], t_s[0], params) * h_half

    for t_index, t in enumerate(t_s[:-1]):
        current_accelerate_diff = next_accelerate_diff
        X_s[t_index + 1][0] = X_s[t_index][0] + (X_s[t_index][1] + current_accelerate_diff) * h
        next_accelerate_diff = accelerate_func(X_s[t_index + 1][0], t_s[t_index + 1], params) * h_half
        X_s[t_index + 1][1] = X_s[t_index][1] + current_accelerate_diff + next_accelerate_diff

    return t_s, X_s


def beeman(accelerate_func, t_start, t_stop, h, initial_conditions, params=()):
    t_s = np.arange(t_start, t_stop, h)
    X_s = np.zeros((len(t_s), len(initial_conditions) - 1))
    prev_x = initial_conditions[0]
    X_s[0] = initial_conditions[1:]
    h_reduced = h / 6
    prev_accelerate_diff = None
    current_accelerate_diff = accelerate_func(prev_x, t_s[0] - h, params) * h_reduced
    next_accelerate_diff = accelerate_func(X_s[0][0], t_s[0], params) * h_reduced

    for t_index, t in enumerate(t_s[:-1]):
        current_accelerate_diff, prev_accelerate_diff = next_accelerate_diff, current_accelerate_diff
        X_s[t_index + 1][0] = X_s[t_index][0] + X_s[t_index][1] * h + (4 * current_accelerate_diff - prev_accelerate_diff) * h
        next_accelerate_diff = accelerate_func(X_s[t_index + 1][0], t_s[t_index + 1], params) * h_reduced
        X_s[t_index + 1][1] = X_s[t_index][1] + 2 * next_accelerate_diff + 5 * current_accelerate_diff - prev_accelerate_diff

    return t_s, X_s
