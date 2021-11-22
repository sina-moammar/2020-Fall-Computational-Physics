import numpy as np
from scipy.optimize import curve_fit
from typing import Callable, List, Tuple, TypeVar


def auto_correlation(data: List[float], lag: int) -> float:
    """Calculates auto-correlation of `data` with `lag` time delay

    Args:
        data (List[float]): data
        lag (int): lag of comparision

    Returns:
        float: auto-correlation of `data[lag:]` with `data[:-lag]
    """
    
    if lag == 0:
        return 1
    else:
        correlation = np.corrcoef(data[:len(data) - lag], data[lag:])[0, 1]
        return 0 if np.isnan(correlation) else correlation
    
    
def _exp_model(x: float, char_length: float) -> float:
    """Model of `f(x) = exp(- x / l)`

    Args:
        x (float): input
        char_length (float): characteristic length (l)

    Returns:
        float: output of function
    """
    
    return np.exp(-x / char_length)


def exp_characteristic_length(x_s: List[float], y_s: List[float]) -> Tuple[float, float]:
    """Finds characteristic length of `exp(-x / l)` function.

    Args:
        x_s (List[float]): inputs
        y_s (List[float]): outputs

    Returns:
        Tuple[float, float]: characteristic length and its error
    """
    
    try:
        x_length = x_s[-1] - x_s[0]
        x_length = 1 if x_length == 0 else x_length
        # should normalized x
        fit_para, fit_error = curve_fit(_exp_model, x_s / x_length, y_s, p0=(.5,))
        # because of normalization
        fit_para = fit_para[0] * x_length
        fit_error = np.sqrt(fit_error[0]) * x_length
    except:
        fit_para, fit_error = 0, 0

    return fit_para, fit_error


X = TypeVar('X')

def bootstrap_error(data: List[X], function: Callable[[List[X]], float], size: int = 100) -> float:
    """Calculates bootstrap error of `data`.

    Args:
        data (List[X]): list of any data
        function (Callable[[List[X]], float]): function gives the value that we want to calculate its error
        size (int, optional): number of batches. Defaults to 100.

    Returns:
        float: bootstrap error
    """
    
    ensemble_values = np.zeros(size)

    for i in range(size):
        # select `len(data)` values from data randomly
        random_numbers = np.random.randint(0, len(data), len(data))
        # get function output of this batch
        ensemble_values[i] = function(data[random_numbers])

    # return standard deviation of function output of batches
    return np.std(ensemble_values)
