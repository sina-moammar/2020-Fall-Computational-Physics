import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import scipy.constants as CONSTANTS
from typing import Dict, List, Tuple

from .utils import auto_correlation, exp_characteristic_length


class Ising2D:
    # in infinty temperature every move is accepted
    _diff_energy_exp_map = np.ones(17)

    def __init__(self, length: int, J: float = 1.0, name: str = None) -> None:
        """Constructs an 2D Ising model

        Args:
            length (int): length of 2D lattice
            J (float): interaction strength. Defaults to 1.0.
            name (str, optional): model name. Defaults to None.
        """
        
        self.length = length
        self.size = length * length
        self.J = J
        self.name = name
        self.reduced_T = np.Inf
        # because in infinty temperature the grid is completely random.
        self.grid = np.random.choice((-1, 1), (length, length))
        # because it is used in many places.
        self._length_range = np.arange(self.length)
        # find coordinates of spins that are not in eachother neighborhood,
        # so we can update them by Metropolise step concurrentlly.
        self.non_interacting_sets = self._non_interacting_sets()
        # neighbors coordinates of non interacting 
        self.sets_neighbors = {set: self._neighbors_coords(*self.non_interacting_sets[set])
                               for set in self.non_interacting_sets.keys()}
        self.reduced_energy = self._get_reduced_energy()
        self.total_magnetisation = self._get_total_magnetization()
 
        
    def _non_interacting_sets(self) -> Dict[str, List[ndarray]]:
        """Finds coordinates of two sets of spins that are not in eachother neighborhood.

        Returns:
            Dict[str, List[np.ndarray]]: keys ars `['set_1', 'set_2']` and values are a list that 
                                         index 0 is rows number list and index 1 is columns number list.
        """
        
        # value of `(i, j)` cell is `i + j`
        coords_sum_grid = self._length_range[:, np.newaxis] + self._length_range
        is_sum_even = (coords_sum_grid % 2 == 0)
        # on set of non-interacting spins coordinates.
        non_inter_coords_1 = np.where(is_sum_even)
        # another set of non-interacting spins coordinates
        non_inter_coords_2 = np.where(~is_sum_even)
        
        return {
            'set_1': non_inter_coords_1,
            'set_2': non_inter_coords_2,
        }


    def _neighbors_coords(self, rows: ndarray, cols: ndarray) -> Tuple[List[ndarray]]:
        """Finds neighbors coordinates of spins at `(row, col)`.

        Args:
            rows (ndarray): list of row numbers of spins
            cols (ndarray): list of column numbers of spins

        Returns:
            Tuple[List[ndarray]]: index 0 is list of list of row numbers of neighbors of that
                                  set of spins and index 1 is like index 0 but for column numbers.
        """
        
        # relative coordinates of neighbors of a cell
        neighbors_rel_coords = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        return (
            np.concatenate([(rows + rel_row)[:, np.newaxis]
                            for rel_row, _ in neighbors_rel_coords], axis=1) % self.length,
            np.concatenate([(cols + rel_col)[:, np.newaxis]
                            for _, rel_col in neighbors_rel_coords], axis=1) % self.length
        )


    def _set_diff_energy_exp(self) -> None:
        """Updates exponents values of boltzmann distribution of `delta_Es in [-8, 8]`
        """
        
        dE_s = np.arange(-8, 9)
        self._diff_energy_exp_map[dE_s] = np.exp(-dE_s / self.reduced_T)


    def _diff_energies_of_set(self, set: str) -> List[int]:
        """Calculates energies difference of `set` spins if flip them.

        Args:
            set (str): `non_interacting_sets` set key

        Returns:
            List[int]: list of changes in energies
        """
        
        return 2 * self.grid[self.non_interacting_sets[set]] *\
            np.sum(self.grid[self.sets_neighbors[set]], axis=1)


    def _get_reduced_energy(self) -> float:
        """Computes reduced energy of the system.

        Returns:
            float: reduced energy
        """
        
        # consider interactions between every spin with its left and top neighbor
        return -np.sum(self.grid * (np.roll(self.grid, 1, axis=1) + np.roll(self.grid, 1, axis=0)))


    def _get_total_magnetization(self) -> int:
        """Computes total magnetization of the system.

        Returns:
            int: total manetixation
        """
        
        return np.sum(self.grid)

    
    def set_temperature(self, temperature: float, is_reduced: bool = True) -> None:
        """Changes system temperature.

        Args:
            temperature (float): new temperature
            is_reduced (bool, optional): is temperature in reduced units. Defaults to True.
        """
        
        self.reduced_T = temperature if is_reduced else temperature / (self.J / CONSTANTS.Boltzmann)
        self._set_diff_energy_exp()
        
        
    def set_grid(self, grid: List[List[int]]) -> None:
        """Set grid of the system and update energy and magnetization.

        Args:
            grid (List[List[int]]): square matrix of spins
        """
        
        self.grid = np.array(grid)
        # update energy and magnetization of the system
        self.reduced_energy = self._get_reduced_energy()
        self.total_magnetisation = self._get_total_magnetization()


    def equilibrate(self, test_steps: int = 100, threshold: float = np.exp(-2)) -> None:
        """Evovles the system to equilibrium.

        Args:
            test_steps (int): number of Metropolis steps before testing. Defaults to 100.
            threshold (float, optional): threshold for auto-correlation to considerthe system
                                         in equilibrium. Defaults to np.exp(-2).
        """
        
        reduced_energies = []
        lag_s = np.arange(int(test_steps / 2))
        is_relaxed = False
        round = 0
        
        while not is_relaxed:
            # Makes `test_size` Mont Carlo steps till auto correlation of the system energy crosse `threshold`
            for _ in range(test_steps):
                self.monte_carlo_move()
                reduced_energies.append(self.reduced_energy)
                
            # pick `test_size` values with equal steps
            test_sized_energies = reduced_energies[::round + 1]
            corr_s = np.array([auto_correlation(test_sized_energies, lag) for lag in lag_s])
            is_relaxed = np.any(corr_s < threshold)
            round += 1


    def energy_relaxation_time(self, test_steps: int = 100, threshold: float = np.exp(-2)) -> int:
        """Finds energy relaxation time that energies are uncorrelated.

        Args:
            test_steps (int): number of Metropolis steps before testing. Defaults to 100.
            threshold (float, optional): threshold for auto-correlation to considerthe energies
                                         are uncorrelated. Defaults to np.exp(-2).
                                         
        Returns:
            int: energy relaxation time
        """
        
        reduced_energies = []
        lag_s = np.arange(int(test_steps / 2))
        round = 0
        
        while True:
            # Makes `test_size` Mont Carlo steps till auto correlation of the system energy crosse `threshold`
            for _ in range(test_steps):
                self.monte_carlo_move()
                reduced_energies.append(self.reduced_energy)
                
            # pick `test_size` values with equal steps
            test_sized_energies = reduced_energies[::round + 1]
            corr_s = np.array([auto_correlation(test_sized_energies, lag) for lag in lag_s])
            # lags indices that their correlations are below threshold
            are_below = corr_s < threshold
            # lags that their correlations are below threshold
            lag_s_below = lag_s[are_below]
            
            if len(lag_s_below) > 0:
                #return first lag that its correlation is below threshold
                # plt.plot(lag_s, corr_s) // TODO delete it
                # plt.show()
                return lag_s_below[0] * (round + 1)

            round += 1

    
    def monte_carlo_move(self) -> float:
        """Make a Monte Carlo step

        Returns:
            float: acceptance rate
        """
        
        acceptance_count = 0
        
        for set_name, set_coords in self.non_interacting_sets.items():
            # flip all spins of each non-interacting set and check metropolise conditions.
            set_energy_diff_s = self._diff_energies_of_set(set_name)
            # finde indices of accepted fliped spins in the set
            are_accepted = np.random.rand(len(set_coords[0])) < self._diff_energy_exp_map[set_energy_diff_s]
            # flip accepted spins
            self.grid[set_coords[0][are_accepted], set_coords[1][are_accepted]] *= -1
            # add accepted energy differences to total energy
            self.reduced_energy += np.sum(set_energy_diff_s[are_accepted])
            # count accepted Metropolise steps
            acceptance_count += np.count_nonzero(are_accepted)

        return acceptance_count / self.size


    def evolve(self, steps: int = 1) -> None:
        """Make Monte Carlo steps as mush as `steps`.

        Args:
            steps (int, optional): Monte Carlo steps. Defaults to 1.
        """
        
        for _ in range(steps):
            self.monte_carlo_move()
            
        # update total magnetization
        self.total_magnetisation = self._get_total_magnetization()


    def spatial_correlation_length(self, threshold: float = 0.1) -> Tuple[float, float]:
        """Calculates spatial correlation length of the grid.

        Args:
            threshold (float, optional): correlation threshold to be confident. Defaults to 0.1.

        Returns:
            Tuple[float, float]: spatial correlation length
        """
        
        # because total magnetization is sum of the spins
        grid_mean_square = np.square((self.total_magnetisation / self.size), )
        grid_var = np.var(self.grid)
        # (<x * y> - <x>^2) / var = 'threshold' => <x * y> = 'threshold' * var + <x>^2 = 'product_threshold'
        product_threshold = 0.1 * grid_var + grid_mean_square
        # because of periodic conditions, Corr(lag) = Corr(length - 1 - lag)
        max_lag = int(np.ceil(self.length / 2))
        lag = 0
        mean_products = []
        is_crossed_threshold = False

        while not is_crossed_threshold and lag <= max_lag:
            grid_rolled = np.roll(self.grid, -lag, axis=1)
            mean_products.append(np.mean(self.grid * grid_rolled))
            
            if mean_products[lag] < product_threshold:
                is_crossed_threshold = True
                
            lag += 1

        mean_products = np.array(mean_products)

        if grid_var != 0:
            correlations = (mean_products - grid_mean_square) / grid_var
            return exp_characteristic_length(self._length_range[:lag], correlations)
        else:
            return 0, 0


    def show(self, save: bool = False, base_dir: str = ".") -> None:
        """Shows the grid.

        Args:
            save (bool, optional): save image. Defaults to False.
            base_dir (bool, optional): base directory. Defaults to ".".
        """
        
        _, ax = plt.subplots()
        plt.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_aspect('equal', 'box')
        plt.pcolormesh(self.grid, cmap='CMRmap_r')
        if save: 
            plt.savefig(f"{base_dir}/{self.name or 'Ising'}_l_{self.length}_T_{self.reduced_T}.png",
                        pad_inches=0, bbox_inches='tight')
        plt.show()
