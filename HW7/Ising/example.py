import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

from .ising import Ising2D
from .ensemble import Ensemble


def example(
    length: int = 100,
    ensemble_size: int = 4000,
    beta_s: List[float] = np.concatenate((np.linspace(0.2, 0.41, 15, endpoint=False), 
                                         np.linspace(0.41, 0.46, 20, endpoint=False),
                                         np.linspace(0.46, 0.7, 15, endpoint=True)))
    ) -> None:
    plt.style.use('seaborn-dark')

    energy_s = np.zeros(len(beta_s))
    energy_err_s = np.zeros(len(beta_s))
    magnetization_s = np.zeros(len(beta_s))
    magnetization_err_s = np.zeros(len(beta_s))
    susceptibility_s = np.zeros(len(beta_s))
    susceptibility_err_s = np.zeros(len(beta_s))
    spatial_corr_length_s = np.zeros(len(beta_s))
    spatial_corr_length_err_s = np.zeros(len(beta_s))
    heat_capacity_s = np.zeros(len(beta_s))
    heat_capacity_err_s = np.zeros(len(beta_s))
    
    model = Ising2D(length, name='example')
    
    with tqdm(beta_s) as t:
        for i, beta in enumerate(t):
            t.set_description(f"beta = {beta}")
            
            model.set_temperature(1 / beta)
            model.equilibrate()
            relaxation_time = model.energy_relaxation_time()
            
            ensemble = Ensemble(model)
            ensemble.get_data(ensemble_size, relaxation_time)
            energy_s[i] = ensemble.get_reduced_energy() / ensemble.size
            energy_err_s[i] = ensemble.get_reduced_energy_error() / ensemble.size
            magnetization_s[i] = ensemble.get_magnetisation() / ensemble.size
            magnetization_err_s[i] = ensemble.get_magnetisation_error() / ensemble.size
            susceptibility_s[i] = ensemble.get_reduced_susceptibility() / ensemble.size
            susceptibility_err_s[i] = ensemble.get_reduced_susceptibility_error() / ensemble.size
            spatial_corr_length_s[i] = ensemble.get_spatial_corr_length()
            spatial_corr_length_err_s[i] = ensemble.get_spatial_corr_length_error()
            heat_capacity_s[i] = ensemble.get_reduced_heat_capacity() / ensemble.size
            heat_capacity_err_s[i] = ensemble.get_reduced_heat_capacity_error() / ensemble.size
        
    plt.errorbar(beta_s, energy_s, yerr=energy_err_s, ecolor='red', capsize=2, marker='o',
                 markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5, label=f'L = {length}')
    plt.xlabel(r'$\beta \ (1/J)$')
    plt.ylabel(r'Average Energy per Spin ($J$)')
    plt.legend()
    plt.show()

    plt.errorbar(beta_s, magnetization_s, yerr=magnetization_err_s, ecolor='red', capsize=2, marker='o',
                 markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5, label=f'L = {length}')
    plt.xlabel(r'$\beta \ (1/J)$')
    plt.ylabel(r'Average Magnetisation per Spin')
    plt.legend()
    plt.show()

    plt.errorbar(beta_s, heat_capacity_s, yerr=heat_capacity_err_s, ecolor='red', capsize=2, marker='o',
                 markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5, label=f'L = {length}')
    plt.xlabel(r'$\beta \ (1/J)$')
    plt.ylabel(r'Heat Capacity per Spin')
    plt.legend()
    plt.show()

    plt.errorbar(beta_s, susceptibility_s, yerr=susceptibility_err_s, ecolor='red', capsize=2, marker='o',
                 markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5, label=f'L = {length}')
    plt.xlabel(r'$\beta \ (1/J)$')
    plt.ylabel(r'Magnetic Susceptibility per Spin ($\chi$)')
    plt.legend()
    plt.show()

    plt.errorbar(beta_s, spatial_corr_length_s, yerr=spatial_corr_length_err_s, ecolor='red', capsize=2, marker='o',
                 markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5, label=f'L = {length}')
    plt.xlabel(r'$\beta \ (1/J)$')
    plt.ylabel(r'Correlation Length ($\xi$)')
    plt.legend()
    plt.show()
