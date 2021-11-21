# import numpy as np
# import matplotlib.pyplot as plt
# from Simulation.HW7.ising import Ising2D
# from scipy.optimize import curve_fit
# import scipy.constants as constants


# def calc_ising_macro_quantities(length_s=(100, 125, 160, 200), beta_end=.7, beta_start=.2, beta_step=0.015, sample_numbers=100, save=False, file_name=None):
#     if file_name is None:
#         beta_s = np.round(np.arange(beta_start, beta_end, beta_step), 3)

#         energy_ensemble_data = np.zeros((len(length_s), len(beta_s)))
#         energy_error_ensemble_data = np.zeros((len(length_s), len(beta_s)))

#         magnetization_ensemble_data = np.zeros((len(length_s), len(beta_s)))
#         magnetization_error_ensemble_data = np.zeros((len(length_s), len(beta_s)))

#         heat_capacity_ensemble_data = np.zeros((len(length_s), len(beta_s)))
#         heat_capacity_error_ensemble_data = np.zeros((len(length_s), len(beta_s)))

#         susceptibility_ensemble_data = np.zeros((len(length_s), len(beta_s)))
#         susceptibility_error_ensemble_data = np.zeros((len(length_s), len(beta_s)))

#         correlation_ensemble_data = np.zeros((len(length_s), len(beta_s)))
#         correlation_error_ensemble_data = np.zeros((len(length_s), len(beta_s)))

#         for length_index, length in enumerate(length_s):
#             model = Ising2D(length, constants.Boltzmann)

#             for beta_index, beta in enumerate(beta_s):
#                 T = 1 / beta
#                 print(f'\rLength: {length}\tBeta: {beta}')
#                 render_data = model.render(T, sample_numbers)
#                 # model.show()

#                 energy_ensemble_data[length_index][beta_index] = render_data.get_reduced_energy() / render_data.size
#                 energy_error_ensemble_data[length_index][beta_index] = render_data.get_reduced_energy_error() / render_data.size

#                 magnetization_ensemble_data[length_index][beta_index] = render_data.get_magnetisation() / render_data.size
#                 magnetization_error_ensemble_data[length_index][beta_index] = render_data.get_magnetisation_error() / render_data.size

#                 heat_capacity_ensemble_data[length_index][beta_index] = render_data.get_reduced_heat_capacity() / render_data.size
#                 heat_capacity_error_ensemble_data[length_index][beta_index] = render_data.get_reduced_heat_capacity_error() / render_data.size

#                 susceptibility_ensemble_data[length_index][beta_index] = render_data.get_reduced_susceptibility() / render_data.size
#                 susceptibility_error_ensemble_data[length_index][beta_index] = render_data.get_reduced_susceptibility_error() / render_data.size

#                 correlation_ensemble_data[length_index][beta_index] = render_data.get_correlation_length()
#                 correlation_error_ensemble_data[length_index][beta_index] = render_data.get_correlation_length_error()

#         data = {
#             'lengths': length_s,
#             'beta_s': beta_s,
#             'steps': sample_numbers,
#             'energy': energy_ensemble_data,
#             'energy_error': energy_error_ensemble_data,
#             'magnetization': magnetization_ensemble_data,
#             'magnetization_error': magnetization_error_ensemble_data,
#             'correlation': correlation_ensemble_data,
#             'correlation_error': correlation_error_ensemble_data,
#             'heat_capacity': heat_capacity_ensemble_data,
#             'heat_capacity_error': heat_capacity_error_ensemble_data,
#             'susceptibility': susceptibility_ensemble_data,
#             'susceptibility_error': susceptibility_error_ensemble_data
#         }
#         np.save('data/ising_render_data_' + str(length_s) + '_' + str(beta_s[0]) + '_' + str(beta_s[-1]) + '_' + str(beta_step) + '_' + str(sample_numbers), data)
#     else:
#         data = np.load('data/' + file_name, allow_pickle=True).tolist()
#         length_s = data['lengths']
#         beta_s = data['beta_s']
#         sample_numbers = data['steps']
#         energy_ensemble_data = data['energy']
#         energy_error_ensemble_data = data['energy_error']

#         magnetization_ensemble_data = data['magnetization']
#         magnetization_error_ensemble_data = data['magnetization_error']

#         heat_capacity_ensemble_data = data['heat_capacity']
#         heat_capacity_error_ensemble_data = data['heat_capacity_error']

#         susceptibility_ensemble_data = data['susceptibility']
#         susceptibility_error_ensemble_data = data['susceptibility_error']

#         correlation_ensemble_data = data['correlation']
#         correlation_error_ensemble_data = data['correlation_error']

#     legends = []
#     for length_index, length in enumerate(length_s):
#         plt.errorbar(beta_s, energy_ensemble_data[length_index], yerr=energy_error_ensemble_data[length_index], ecolor='red', capsize=2, marker='o', markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5)
#         legends.append(f'Length = {length}')

#     plt.xlabel(r'$\beta \ (1/J)$')
#     plt.ylabel(r'Average Energy per Spin ($J$)')
#     plt.legend(legends)
#     if save:
#         plt.savefig(
#             'images/ising_render_data_e' + str(length_s) + '_' + str(beta_s[0]) + '_' + str(beta_s[-1]) + '_' + str(np.round(beta_s[1] - beta_s[0], 3)) + '_' + str(sample_numbers) + '.jpg')
#     plt.show()

#     for length_index, length in enumerate(length_s):
#         plt.errorbar(beta_s, magnetization_ensemble_data[length_index], yerr=magnetization_error_ensemble_data[length_index], ecolor='red', capsize=2, marker='o', markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5)

#     plt.xlabel(r'$\beta \ (1/J)$')
#     plt.ylabel(r'Average Magnetisation per Spin')
#     plt.legend(legends)
#     if save:
#         plt.savefig(
#             'images/ising_render_data_m' + str(length_s) + '_' + str(beta_s[0]) + '_' + str(beta_s[-1]) + '_' + str(np.round(beta_s[1] - beta_s[0], 3)) + '_' + str(sample_numbers) + '.jpg')
#     plt.show()

#     for length_index, length in enumerate(length_s):
#         plt.errorbar(beta_s, heat_capacity_ensemble_data[length_index], yerr=heat_capacity_error_ensemble_data[length_index], ecolor='red', capsize=2, marker='o', markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5)

#     plt.xlabel(r'$\beta \ (1/J)$')
#     plt.ylabel(r'Heat Capacity per Spin')
#     plt.legend(legends)
#     if save:
#         plt.savefig(
#             'images/ising_render_data_c' + str(length_s) + '_' + str(beta_s[0]) + '_' + str(beta_s[-1]) + '_' + str(np.round(beta_s[1] - beta_s[0], 3)) + '_' + str(sample_numbers) + '.jpg')
#     plt.show()
#     print(beta_s[np.argmax(heat_capacity_ensemble_data[0])])
#     print(np.max(heat_capacity_ensemble_data[0]))

#     for length_index, length in enumerate(length_s):
#         plt.errorbar(beta_s, susceptibility_ensemble_data[length_index], yerr=susceptibility_error_ensemble_data[length_index], ecolor='red', capsize=2, marker='o', markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5)

#     plt.xlabel(r'$\beta \ (1/J)$')
#     plt.ylabel(r'Magnetic Susceptibility per Spin')
#     plt.legend(legends)
#     if save:
#         plt.savefig(
#             'images/ising_render_data_x' + str(length_s) + '_' + str(beta_s[0]) + '_' + str(beta_s[-1]) + '_' + str(np.round(beta_s[1] - beta_s[0], 3)) + '_' + str(sample_numbers) + '.jpg')
#     plt.show()
#     print(beta_s[np.argmax(susceptibility_ensemble_data[0])])
#     print(np.max(susceptibility_ensemble_data[0]))

#     for length_index, length in enumerate(length_s):
#         plt.errorbar(beta_s, correlation_ensemble_data[length_index], yerr=correlation_error_ensemble_data[length_index], ecolor='red', capsize=2, marker='o', markersize=3, linestyle='-', linewidth=1.5, elinewidth=1.5)

#     plt.xlabel(r'$\beta \ (1/J)$')
#     plt.ylabel(r'Correlation Length ($\xi$)')
#     plt.legend(legends)
#     if save:
#         plt.savefig(
#             'images/ising_render_data_xi' + str(length_s) + '_' + str(beta_s[0]) + '_' + str(beta_s[-1]) + '_' + str(np.round(beta_s[1] - beta_s[0], 3)) + '_' + str(sample_numbers) + '.jpg')
#     plt.show()
#     print(beta_s[np.argmax(correlation_ensemble_data[0])])
#     print(np.max(correlation_ensemble_data[0]))


# def model_linear_func(x, a, b):
#     return a * x + b


# def fit_line(x_s, y_s):
#     para, error = curve_fit(model_linear_func, x_s, y_s)

#     return para


# def critical_exponents():
#     length_s = np.array([8, 16, 32, 64])
#     length_s_ln = np.log(length_s)
#     T_c = 2 / np.log(1 + np.sqrt(2))

#     xi_s = np.array([.88, 1.47, 2.39, 5.28])
#     xi_s_ln = np.log(xi_s)
#     T_xi = np.array([0.352, 0.408, 0.425, 0.435]) ** -1
#     diff_T = np.abs(T_xi - T_c)
#     diff_T_ln = np.log(diff_T)
#     nu_para = fit_line(diff_T_ln, length_s_ln)
#     nu = -nu_para[0]
#     nu_fit = np.exp(diff_T_ln * nu_para[0] + nu_para[1])
#     plt.plot(diff_T, length_s, linestyle='', marker='.', markersize=5)
#     plt.plot(diff_T, nu_fit, linestyle='--', color='g')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel(r'$|T_c(L) - T_c(\infty)|$')
#     plt.ylabel(r'$L$')
#     plt.legend(['Samples', 'Fitted Curve'])
#     plt.savefig('images/nu.jpg')
#     plt.show()

#     x_s = np.array([1.94, 6.95, 22.67, 74.88])
#     x_s_ln = np.log(x_s)
#     gamma_para = fit_line(length_s_ln, x_s_ln)
#     gamma_nu = gamma_para[0]
#     gamma_fit = np.exp(length_s_ln * gamma_para[0] + gamma_para[1])
#     plt.plot(length_s, x_s, linestyle='', marker='.', markersize=5)
#     plt.plot(length_s, gamma_fit, linestyle='--', color='g')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel(r'$L$')
#     plt.ylabel(r'$\chi / N$')
#     plt.legend(['Samples', 'Fitted Curve'])
#     plt.savefig('images/gamma.jpg')
#     plt.show()

#     c_s = np.array([1.32, 1.77, 2.15, 2.60])
#     c_para = fit_line(length_s_ln, c_s)
#     c_nu = -c_para[0]
#     c_fit = length_s_ln * c_para[0] + c_para[1]
#     plt.plot(length_s, c_s, linestyle='', marker='.', markersize=5)
#     plt.plot(length_s, c_fit, linestyle='--', color='g')
#     plt.xscale('log')
#     plt.xlabel(r'$L$')
#     plt.ylabel(r'$C / N$')
#     plt.legend(['Samples', 'Fitted Curve'])
#     plt.savefig('images/c_0.jpg')
#     plt.show()

#     beta_s = np.array([0.5873, 0.5038, 0.45275, 0.3814])
#     beta_s_ln = np.log(beta_s)
#     beta_para = fit_line(length_s_ln, beta_s_ln)
#     beta_nu = -beta_para[0]
#     beta_fit = np.exp(length_s_ln * beta_para[0] + beta_para[1])
#     plt.plot(length_s, beta_s, linestyle='', marker='.', markersize=5)
#     plt.plot(length_s, beta_fit, linestyle='--', color='g')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel(r'$L$')
#     plt.ylabel(r'$|M| / N$')
#     plt.legend(['Samples', 'Fitted Curve'])
#     plt.savefig('images/beta.jpg')
#     plt.show()

#     print(nu)
#     print(gamma_nu)
#     print(c_nu)
#     print(beta_nu)


# if __name__ == "__main__":
#     calc_ising_macro_quantities([108, 125, 140, 160], 0.7, 0.2, 0.015, 100, save=True)
#     critical_exponents()
