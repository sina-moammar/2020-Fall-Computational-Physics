import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def julia_set(c: complex, iterations: int, x_limit: float, y_limit: float, ppl: int = 1000, x_origin: float = 0., y_origin: float = 0.) -> np.ndarray:
    threshold = (1 + np.sqrt(1 + 4 * np.abs(c))) / 2
    n_x = int(ppl * x_limit)
    n_y = int(ppl * y_limit)
    y, x = np.ogrid[y_limit:-y_limit:n_y * 1j, -x_limit:x_limit:n_x * 1j]
    points = (x + x_origin) + 1j * (y + y_origin)
    iterations_to_diverge = np.ones(points.shape) * iterations
    is_diverged = np.zeros(points.shape, dtype=bool)

    for iteration in range(iterations):
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
    # plt.show()
    plt.close()
    
    
def make_zoom_video():
    c = -0.7 + .3j
    # x_origin, y_origin = 2.087805e-9, (0.2499 + 3.53e-10)
    x_origin, y_origin = 2.0877695e-9, (0.2499 + 3.53e-10)
    length_limit = 1
    zoom_factor = 1.1
    ppl = 1000
    iterations = 200
    frames = 320
    
    dir_name = f'videos/julia_c={c}_itr={iterations}_l={length_limit}_zoom={zoom_factor}_temp_temp'
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    for frame in tqdm(range(frames)):
        iterations_to_diverge = julia_set(
            c=c,
            iterations=int(iterations),
            x_limit=length_limit,
            y_limit=length_limit,
            ppl=ppl,
            x_origin=x_origin,
            y_origin=y_origin
        )
        show_julia_set(iterations_to_diverge, f"{dir_name}/{frame}.jpg")
        length_limit /= zoom_factor
        ppl *= zoom_factor
        iterations += 0.5


if __name__ == '__main__':
    make_zoom_video()
    # c = -0.7 + .3j
    # iterations = 1000
    # iterations_to_diverge = julia_set(
    #     c=c,
    #     iterations=iterations,
    #     x_limit=1.55,
    #     y_limit=0.95,
    #     ppl=1000,
    # )
    # file_name = f'images/julia_c={c}_itr={iterations}_w={iterations_to_diverge.shape[1]}_h={iterations_to_diverge.shape[0]}.jpg'
    # show_julia_set(iterations_to_diverge, file_name)
