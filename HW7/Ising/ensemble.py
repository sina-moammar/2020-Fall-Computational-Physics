from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as CONSTANTS

from .ising import Ising2D
from .tools import bootstrap_error


class Ensemble:
    def __init__(self, model: Ising2D) -> None:
        """Construct ensemble wrapper for given 2D Ising `model`.

        Args:
            model (Ising2D): 2D Ising model
        """
        
        self.model = model
        self.ensemble_size = 0
        
        self.name = self.model.name
        self.length = self.model.length
        self.size = self.model.size
        self.J = self.model.J
        self.reduced_T = self.model.reduced_T
        
        # ensemble data
        self.reduced_energy_s = np.array([])
        self.total_magnetisation_s = np.array([])
        self.spatial_corr_length_s = np.array([])
        
        # calculated values
        self._reduced_energy = None
        self._reduced_energy_error = None
        self._magnetisation = None
        self._magnetisation_error = None
        self._spatial_corr_length = None
        self._spatial_corr_length_error = None
        self._reduced_heat_capacity = None
        self._reduced_heat_capacity_error = None
        self._reduced_susceptibility = None
        self._reduced_susceptibility_error = None


    def save(self, base_dir: str = "") -> None:
        """Saves ensemble data.
        
        Args:
            base_dir (bool, optional): base directory. Defaults to "".
        """
        
        data = {
            'name': self.name,
            'model_grid': self.model.grid,
            'length': self.length,
            'size': self.size,
            'J': self.J,
            'reduced_T': self.reduced_T,
            'ensemble_size': self.ensemble_size,
            'reduced_energy_s': self.reduced_energy_s,
            'total_magnetisation_s': self.total_magnetisation_s,
            'spatial_corr_length_s': self.spatial_corr_length_s,
            '_reduced_energy': self._reduced_energy,
            '_reduced_energy_error': self._reduced_energy_error,
            '_magnetisation': self._magnetisation,
            '_magnetisation_error': self._magnetisation_error,
            '_spatial_corr_length': self._spatial_corr_length,
            '_spatial_corr_length_error': self._spatial_corr_length_error,
            '_reduced_heat_capacity': self._reduced_heat_capacity,
            '_reduced_heat_capacity_error': self._reduced_heat_capacity_error,
            '_reduced_susceptibility': self._reduced_susceptibility,
            '_reduced_susceptibility_error': self._reduced_susceptibility_error,
        }

        np.save(f"{base_dir}/{self.name or 'Ising'}_l_{self.length}_T_{self.reduced_T}_ens_{self.ensemble_size}", data)


    @staticmethod
    def load(file_name: str) -> Ensemble:
        """Make Ensemble from `file_name`

        Args:
            file_name (str): file name of saved data

        Returns:
            Ensemble: built ensemble from data
        """
        # load data from file
        data = np.load(file_name, allow_pickle=True).tolist()
        
        # make Ising model from data
        model = Ising2D(data['length'], data['J'], data['name'])
        model.set_temperature(data['reduced_T'])
        model.set_grid(data['model_grid'])
        
        # make ensemble from data
        ensemble = Ensemble(model)
        ensemble.ensemble_size = data['ensemble_size']
        
        ensemble.reduced_energy_s = data['reduced_energy_s']
        ensemble.total_magnetisation_s = data['total_magnetisation_s']
        ensemble.spatial_corr_length_s = data['spatial_corr_length_s']

        ensemble._reduced_energy = data['_reduced_energy']
        ensemble._reduced_energy_error = data['_reduced_energy_error']
        ensemble._magnetisation = data['_magnetisation']
        ensemble._magnetisation_error = data['_magnetisation_error']
        ensemble._spatial_corr_length = data['_spatial_corr_length']
        ensemble._spatial_corr_length_error = data['_spatial_corr_length_error']
        ensemble._reduced_heat_capacity = data['_reduced_heat_capacity']
        ensemble._reduced_heat_capacity_error = data['_reduced_heat_capacity_error']
        ensemble._reduced_susceptibility = data['_reduced_susceptibility']
        ensemble._reduced_susceptibility_error = data['_reduced_susceptibility_error']

        return ensemble


    def get_data(self, size: int, step: int = 1) -> None:
        """Gets and adds new data by step size of `step`

        Args:
            size (int): number of new data. For uncorrelated data it should be `relaxation time`.
            step (int, optional): step size to evolve the model before getting each data. Defaults to 1.
        """
        
        new_reduced_energy_s = []
        new_total_magnetisation_s = []
        new_spatial_corr_length_s = []
        
        for _ in range(size):
            # evolve the model as mush as `step`
            self.model.evolve(step)
            new_reduced_energy_s.append(self.model.reduced_energy)
            new_total_magnetisation_s.append(np.abs(self.model.total_magnetisation))
            new_spatial_corr_length_s.append(self.model.spatial_correlation_length()[0])
            self.ensemble_size += 1
            
        # append new data to old
        self.reduced_energy_s = np.append(self.reduced_energy_s, new_reduced_energy_s)
        self.total_magnetisation_s = np.append(self.total_magnetisation_s, new_total_magnetisation_s)
        self.spatial_corr_length_s = np.append(self.spatial_corr_length_s, new_spatial_corr_length_s)
    

    def get_reduced_energy(self) -> float:
        """Calculates average of ensemble reduced energies.

        Returns:
            float: average of ensemble reduced energies
        """
        # calculated reduced energy if it is not done before
        if self._reduced_energy is None:
            self._reduced_energy = np.mean(self.reduced_energy_s)

        return self._reduced_energy


    def get_reduced_energy_error(self) -> float:
        """Calculates standard deviation of ensemble reduced energies.

        Returns:
            float: standard deviation of ensemble reduced energies
        """
        # calculated reduced energy error if it is not done before
        if self._reduced_energy_error is None:
            self._reduced_energy_error = np.std(self.reduced_energy_s, ddof=1) / np.sqrt(self.ensemble_size)

        return self._reduced_energy_error


    def get_reduced_heat_capacity(self) -> float:
        """Calculates ensemble reduced heat capacity.

        Returns:
            float: ensemble reduced heat capacity
        """
        # calculated reduced heat capacity if it is not done before
        if self._reduced_heat_capacity is None:
            self._reduced_heat_capacity = np.var(self.reduced_energy_s) / np.square(self.reduced_T)

        return self._reduced_heat_capacity


    def get_reduced_heat_capacity_error(self) -> float:
        """Calculates error of ensemble reduced heat capacity.

        Returns:
            float: error of ensemble reduced heat capacity
        """
        # calculated reduced heat capacity error if it is not done before
        if self._reduced_heat_capacity_error is None:
            self._reduced_heat_capacity_error = bootstrap_error(self.reduced_energy_s, np.var) / np.square(self.reduced_T)

        return self._reduced_heat_capacity_error


    def get_reduced_susceptibility(self) -> float:
        """Calculates ensemble reduced susceptibility.

        Returns:
            float: ensemble reduced susceptibility
        """
        # calculated reduced susceptibility if it is not done before
        if self._reduced_susceptibility is None:
            self._reduced_susceptibility = np.var(self.total_magnetisation_s) / self.reduced_T

        return self._reduced_susceptibility


    def get_reduced_susceptibility_error(self) -> float:
        """Calculates error of ensemble reduced susceptibility.

        Returns:
            float: error of ensemble reduced susceptibility
        """
        # calculated reduced susceptibility error if it is not done before
        if self._reduced_susceptibility_error is None:
            self._reduced_susceptibility_error = bootstrap_error(self.total_magnetisation_s, np.var) / self.reduced_T

        return self._reduced_susceptibility_error


    def get_magnetisation(self) -> float:
        """Calculates average of ensemble total magnetisation.

        Returns:
            float: average of ensemble total magnetisation
        """
        # calculated total magnetisation if it is not done before
        if self._magnetisation is None:
            self._magnetisation = np.mean(self.total_magnetisation_s)

        return self._magnetisation


    def get_magnetisation_error(self) -> float:
        """Calculates error of ensemble total magnetisation.

        Returns:
            float: error of ensemble total magnetisation
        """
        # calculated total magnetisation error if it is not done before
        if self._magnetisation_error is None:
            self._magnetisation_error = np.std(self.total_magnetisation_s, ddof=1) / np.sqrt(self.ensemble_size)

        return self._magnetisation_error


    def get_spatial_corr_length(self) -> float:
        """Calculates average of ensemble spatial correlation length.

        Returns:
            float: average of ensemble spatial correlation length
        """
        # calculated spatial correlation length if it is not done before
        if self._spatial_corr_length is None:
            self._spatial_corr_length = np.mean(self.spatial_corr_length_s)

        return self._spatial_corr_length


    def get_spatial_corr_length_error(self) -> float:
        """Calculates error of ensemble spatial correlation length.

        Returns:
            float: error of ensemble spatial correlation length
        """
        # calculated spatial correlation length error if it is not done before
        if self._spatial_corr_length_error is None:
            self._spatial_corr_length_error = np.std(self.spatial_corr_length_s, ddof=1) / np.sqrt(self.ensemble_size)

        return self._spatial_corr_length_error


    def get_energy(self) -> float:
        """Calculate energy in normal units.

        Returns:
            float: energy of the system
        """
        
        return self.get_reduced_energy() * self.J


    def get_energy_error(self) -> float:
        """Calculate energy error in normal units.

        Returns:
            float: energy error of the system
        """
        
        return self.get_reduced_energy_error() * self.J


    def get_heat_capacity(self) -> float:
        """Calculate heat capacity in normal units.

        Returns:
            float: heat capacity of the system
        """
        
        return self.get_reduced_heat_capacity() * CONSTANTS.Boltzmann


    def get_heat_capacity_error(self) -> float:
        """Calculate heat capacity error in normal units.

        Returns:
            float: heat capacity error of the system
        """
        
        return self.get_reduced_heat_capacity_error() * CONSTANTS.Boltzmann


    def get_susceptibility(self) -> float:
        """Calculate susceptibility in normal units.

        Returns:
            float: susceptibility of the system
        """
        
        return self.get_reduced_susceptibility() / self.J


    def get_susceptibility_error(self) -> float:
        """Calculate susceptibility error in normal units.

        Returns:
            float: susceptibility error of the system
        """
        
        return self.get_reduced_susceptibility_error() / self.J
