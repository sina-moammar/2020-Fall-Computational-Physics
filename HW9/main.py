import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Simulation.HW9.md import SingleAtomMD


def line(t, a, b):
    return a * t + b


sigma = 3.405 * 10 ** -10
mass = 6.63 * 10 ** -26
epsilon = 1.653 * 10 ** -21
length = 30
N = 100
v_max = 1.5
h = 1e-3

model = SingleAtomMD(length, N, 2, sigma, mass, epsilon, v_max, h=h, saving_period=500)
data = model.render(200000)
print(data.get_reduced_temperature())
print(data.get_temperature())
print(data.get_reduced_pressure())
print(data.get_pressure())
data.plot_energies()
data.plot_left_side_atom_numbers()
data.plot_pressure()
data.plot_temperature()
print(data.auto_correlation(True))
data.animate()


### Phase Transition ###
size = 50
scale = .9
scales = 0.9 * np.ones(7)
scales = np.concatenate((scales, 0.98 * np.ones(15)))
scales = np.concatenate((scales, 0.95 * np.ones(10)))
scales = np.concatenate((scales, 0.9 * np.ones(10)))
scales = np.concatenate((scales, 0.85 * np.ones(10)))
scales = np.concatenate((scales, 0.8 * np.ones(10)))
scales = np.concatenate((scales, 0.75 * np.ones(10)))

model = SingleAtomMD(length, N, 2, sigma, mass, epsilon, v_max, h=h, saving_period=500)
data = model.render(100000)

new_model = SingleAtomMD(length, N, 2, sigma, mass, epsilon, v_max, h=h, saving_period=500)
new_model.initiate_from_positions_velocities(model.positions, model.velocities)
model = new_model
data = model.render(20000)
E_s = np.zeros(len(scales))
E_error_s = np.zeros(len(scales))
T_s = np.zeros(len(scales))
T_error_s = np.zeros(len(scales))

for index in range(len(scales)):
    model.velocities = model.velocities * scales[index]
    data = model.render(20000)
    T_s[index], T_error_s[index] = data.get_reduced_temperature()
    E_s[index], E_error_s[index] = data.get_reduced_energy()


data.animate(suffix='phase_transition')

data = {
    'E_s': E_s,
    'E_error_s': E_error_s,
    'T_s': T_s,
    'T_error_s': T_error_s
}
np.save(f'data/MD_{length}_{N}_2_{h}_E_T.npy', data)
data = np.load('data/MD_30_100_2_0.001_E_T.npy', allow_pickle=True).tolist()
E_s = data['E_s']
E_error_s = data['E_error_s']
T_s = data['T_s']
T_error_s = data['T_error_s']
plt.errorbar(T_s, E_s, xerr=T_error_s, yerr=E_error_s, linestyle='', marker='o', markersize=3)
plt.xlabel(r'T $(\times \epsilon / k_B)$')
plt.ylabel(r'Energy $(\times \epsilon)$')
plt.savefig(f'images/MD_{length}_{N}_2_{h}_E_T.jpg')
plt.show()


### P vs T ###
v_m_s = np.linspace(1.05, 2, 20)
T_s = np.zeros(len(v_m_s))
T_error_s = np.zeros(len(v_m_s))
P_s = np.zeros(len(v_m_s))
P_error_s = np.zeros(len(v_m_s))
for index, v_m in enumerate(v_m_s):
    print(v_m)
    model = SingleAtomMD(length, N, 2, sigma, mass, epsilon, v_m, h=h, saving_period=100)
    data = model.render(100000)
    T_s[index], T_error_s[index] = data.get_reduced_temperature()
    P_s[index], P_error_s[index] = data.get_reduced_pressure()

data = {
    'T_s': T_s,
    'T_error_s': T_error_s,
    'P_s': P_s,
    'P_error_s': P_error_s
}
np.save(f'data/MD_{length}_{N}_2_{h}_P_T.npy', data)

data = np.load(f'data/MD_{length}_{N}_2_{h}_P_T.npy', allow_pickle=True).tolist()
T_s = data['T_s']
P_s = data['P_s']

para, err = curve_fit(line, T_s, P_s)
err = np.diag(err)
fit = T_s * para[0] + para[1]

plt.errorbar(T_s, P_s, linestyle='', marker='o', markersize=5)
plt.plot(T_s, fit)
plt.legend(['Fitted Line', 'Samples'])
plt.xlabel(r'T $(\times \epsilon / k_B)$')
plt.ylabel(r'P $(\times \epsilon / \sigma^' + f'{2}' + ')$')
plt.savefig(f'images/MD_{length}_{N}_2_{h}_P_T.jpg')
plt.show()
print(para)
print(err)


### L = 90, N = 1000 ###
length = 90
N = 1000

model = SingleAtomMD(length, N, 2, sigma, mass, epsilon, v_max, h=h, saving_period=500)
data = model.render(200000)
print(data.get_reduced_temperature())
print(data.get_temperature())
print(data.get_reduced_pressure())
print(data.get_pressure())
data.plot_energies()
data.plot_left_side_atom_numbers()
data.plot_pressure()
data.plot_temperature()
print(data.auto_correlation(True))
data.animate()
