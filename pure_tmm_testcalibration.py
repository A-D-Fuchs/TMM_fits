import lmfit
import tmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Parameters
import re
import os
import load


matplotlib.use('qt5agg')

# wavelengths = np.linspace(628,970,201)
# angles = np.linspace(0,17.7*np.pi/180,11)
# for wavelength in wavelengths:
#     intensity = 0
#     for angle in angles:
#         result = tmm.coh_tmm('p', [1,2.6,1,2.6], [np.inf,546,1240,np.inf],
#                      th_0=angle, lam_vac=wavelength)
#         intensity += result['R']
#
#     plt.scatter(wavelength, intensity/len(angles), c='blue')

def plot_spec(wavelengths, d_mem, d_vac, amplitude,ax):
    for wavelength in wavelengths:
        result = tmm.coh_tmm('p', [1,2.6,1,2.6], [np.inf,d_mem,d_vac,np.inf],
                     th_0=0, lam_vac=wavelength)
        intensity = result['R']
        ax.scatter(wavelength, intensity*amplitude, s=1,c='red')

def tmm_fit(params, wavelengths, measured_intensity):
    def optimization_fun(pars):
        parvals = pars.valuesdict()
        d_mem = parvals['d_mem']
        d_vac = parvals['d_vac']
        amplitude = parvals['amplitude']
        reflectivity = []
        for wavelength in wavelengths:
            result = tmm.coh_tmm('s', [1, 2.58, 1, 2.58], [np.inf, d_mem, d_vac, np.inf],
                                 th_0=0, lam_vac=wavelength)
            reflectivity.append(result['R'])
        reflectivity = np.array(reflectivity)*amplitude
        return np.array((reflectivity-measured_intensity)**2, dtype=float)
    fit = lmfit.minimize(optimization_fun, params)
    return fit.params.valuesdict()

def get_regex_matches(folder_path):
    match_list = []
    search_pattern = re.compile(r'^spectru(.*).txt$')
    for file in os.listdir(folder_path):
        print(file)
        search_result = search_pattern.search(file)
        if search_result:
            filepath = os.path.join(folder_path, search_result[0])
            match_list.append(filepath)
    return match_list


# data_path = get_regex_matches(r'C:\Users\fulapuser\Documents\AlexanderFuchs\Data\xy_map_spectrum_unten17-25-40\spectrum_x1131.95_y465.308642.txt')
data_path = r"C:\Users\fulapuser\Documents\AlexanderFuchs\Data\xy_map_71x71_mitte22-13-44\spectrum_x-0.7_y28.918919.txt"

lamb, intensity=np.loadtxt(data_path,skiprows=4,unpack=True)
intensity=intensity-310
lamb=np.array(lamb)
# calib_spec_path = r'C:\Users\fulapuser\Documents\Measurement Data\Morris\FP_morris\161\mirror_for_fabryperot_calibration\iris2.5mm_18-01-10\spec_fabry_x371.45_y-383.4.txt'
# calib_spec_path = get_regex_matches(r"C:\Users\fulapuser\Documents\AlexanderFuchs\Data\xy_map_spectrum_unten17-25-40\spectrum_x1290_y550.txt")
calib_spec_path = (r"C:\Users\fulapuser\Documents\AlexanderFuchs\Data\xy_map_71x71_mitte22-13-44\spectrum_x110_y210.txt")
calib_data=np.loadtxt(calib_spec_path,skiprows=4,unpack=True)
lamb_calib,intensity_calib= calib_data[0,:],calib_data[1,:]
# lamb_min = min(lamb[0], lamb_calib[0])
# lamb_max = max(lamb[-1], lamb_calib[-1])
#
# i_min_lamb = np.argmin(np.abs(lamb - lamb_min))
# i_min_lamb_calib = np.argmin(np.abs(lamb_calib - lamb_min))
# i_max_lamb = np.argmin(np.abs(lamb - lamb_max))+1
# i_max_lamb_calib = np.argmin(np.abs(lamb_calib - lamb_max))+1
I_normed = intensity/(intensity_calib-310)
I_normed = I_normed / np.max(I_normed) * 0.8
lamb_normed = lamb


params = Parameters()
params.add('d_mem',600,min=550,max=650, vary=True)
params.add('d_vac',1040,min=400,max=1200, vary=True)
params.add('amplitude',1,min=0.1,max=3, vary=False)

tail_cutoff = 860
index_ignore_tail = np.argmin(np.abs(lamb-tail_cutoff))

fit_result = tmm_fit(params, lamb_normed[:index_ignore_tail], I_normed[:index_ignore_tail])
print(fit_result)
intensity_subtraction = (intensity-intensity_calib)/(np.max(intensity-intensity_calib))
fit_result_subtraction = tmm_fit(params, lamb_normed[:43], intensity_subtraction[:43])
print(fit_result_subtraction)
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.plot(lamb, intensity)
ax1.set_title('Raw Data')
ax2.plot(lamb_calib,intensity_calib)
ax2.set_title('Raw Calib Data')
ax3.plot(lamb_normed,I_normed)
ax3.set_title('Raw Data / Calib Data')
ax4.plot(lamb_normed,intensity_subtraction)
ax4.set_title('Raw Data - Calib Data')

plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], fit_result['amplitude'],ax3)
plot_spec(lamb_normed, fit_result_subtraction['d_mem'], fit_result_subtraction['d_vac'], fit_result_subtraction['amplitude'],ax4)
# plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], 1.4,ax)


ax3.set_ylim(-0.1,1.1)
plt.show()
print(fit_result['d_mem']+fit_result['d_vac'])




