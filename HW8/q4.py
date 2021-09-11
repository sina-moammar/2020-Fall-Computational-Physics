import numpy as np
import matplotlib.pyplot as plt


def logistic_map(r, x_0, n):
    final_x = x_0

    for step in range(n):
        final_x = (4 * r) * final_x * (1 - final_x)

    return final_x


def bifurcation_diagram(r_start, r_end, r_samples, sample_numbers, max_n):
    r_s = np.linspace(r_start, r_end, r_samples)
    x_0_s = np.random.rand(sample_numbers)
    x_r_s = np.zeros((r_samples, sample_numbers))

    for r_index, r in enumerate(r_s):
        x_r_s[r_index] = logistic_map(r, x_0_s, max_n)

    data = {
        'r_start': r_start,
        'r_end': r_end,
        'r_samples': r_samples,
        'sample_numbers': sample_numbers,
        'max_n': max_n,
        'x_r_s': x_r_s
    }
    file_name = f'data/q4_{r_start}_{r_end}_{r_samples}_{sample_numbers}_{max_n}.npy'
    np.save(file_name, data)

    return file_name


def show(file_name, fig_width = 10, fig_height=5, r_lim=None, x_lim=None, show_x_m=False, save=True):
    data = np.load('data/' + file_name, allow_pickle=True).tolist()
    r_start = data['r_start']
    r_end = data['r_end']
    r_samples = data['r_samples']
    sample_numbers = data['sample_numbers']
    max_n = data['max_n']
    x_r_s = data['x_r_s']

    r_lim_start = r_start if r_lim is None else r_lim[0]
    r_lim_end = r_end if r_lim is None else r_lim[1]
    r_lim_ratio = (r_lim_end - r_lim_start) / 1

    r_s = np.linspace(r_start, r_end, r_samples)
    ones = 4 * np.ones(sample_numbers)

    plt.figure(figsize=(fig_width, fig_height))
    for r_index, r in enumerate(r_s):
        plt.plot(r * ones, x_r_s[r_index], linestyle='', marker='.', color='black',
                 markersize=133 * fig_width / r_samples * r_lim_ratio)

    if show_x_m:
        plt.axhline(0.5)
    if r_lim is not None:
        plt.xlim(r_lim)
    if x_lim is not None:
        plt.ylim(x_lim)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$x$')
    if save:
        plt.savefig(f'images/q4_{r_lim_start}_{r_lim_end}_{r_samples}_{sample_numbers}_{max_n}.jpg')
    plt.show()


if __name__ == "__main__":
    file_name = bifurcation_diagram(0, 1, 10000, 100, 10000)
    show(file_name=file_name, fig_height=5, fig_width=10)
