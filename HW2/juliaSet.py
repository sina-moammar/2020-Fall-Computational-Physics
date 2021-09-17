import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def julia_set(c: complex, iterations: int, x_limit: float, y_limit: float, ppl: int = 1000) -> np.ndarray:
    threshold = (1 + np.sqrt(1 + 4 * np.abs(c))) / 2
    n_x = int(ppl * x_limit)
    n_y = int(ppl * y_limit)
    y, x = np.ogrid[y_limit:-y_limit:n_y * 1j, -x_limit:x_limit:n_x * 1j]
    points = x + 1j * y
    iterations_to_diverge = np.ones(points.shape) * iterations
    is_diverged = np.zeros(points.shape, dtype=bool)

    for iteration in tqdm(range(iterations)):
        points = np.square(points) + c
        points_distance = np.abs(points)
        new_diverged = (points_distance > threshold)
        is_diverged[new_diverged] = True
        iterations_to_diverge[new_diverged] = iteration
        points[is_diverged] = 0

    return iterations_to_diverge


def show_julia_set(iterations_to_diverge: np.ndarray, file_name: str = None) -> None:
    fig = plt.figure(figsize=np.array(iterations_to_diverge.shape)[::-1] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(iterations_to_diverge, cmap='twilight_shifted')
    Path("images").mkdir(parents=True, exist_ok=True)
    if file_name:
        plt.savefig(file_name)
    plt.show()


if __name__ == '__main__':
    c = -0.7 + .3j
    iterations = 1000
    iterations_to_diverge = julia_set(
        c=c,
        iterations=iterations,
        x_limit=1.55,
        y_limit=0.95,
        ppl=1000,
    )
    file_name = f'images/julia_c={c}_itr={iterations}_w={iterations_to_diverge.shape[1]}_h={iterations_to_diverge.shape[0]}.jpg'
    show_julia_set(iterations_to_diverge, file_name)
