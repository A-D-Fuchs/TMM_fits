import lmfit
import tmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Parameters
import os
import re
from joblib import Parallel, delayed
from datetime import datetime
from refractive_index.refractive_index import n_from_string
startTime = datetime.now()

matplotlib.use('Agg')
plt.ioff()
n_SiC = n_from_string('SiC')

global fitted_distance
fitted_distance = 1600
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


def plot_spec(wavelengths, d_mem, d_vac, ax):
    for wavelength in wavelengths:
        result = tmm.coh_tmm('s', [1,2.6,1,2.6], [np.inf,d_mem,d_vac,np.inf],
                     th_0=0, lam_vac=wavelength)
        intensity = result['R']
        ax.scatter(wavelength, intensity, s=1,c='red')


def tmm_fit(params, wavelengths, measured_intensity):
    def optimization_fun(pars):
        parvals = pars.valuesdict()
        d_mem = parvals['d_mem']
        d_vac = parvals['d_vac']
        amplitude = parvals['amplitude']
        reflectivity = []
        for wavelength in wavelengths:
            n_SiC_lamb = n_SiC.get_n(wavelength*1e-9)
            result = tmm.coh_tmm('s', [1, n_SiC_lamb, 1, n_SiC_lamb], [np.inf, d_mem, d_vac, np.inf],
                                 th_0=0, lam_vac=wavelength)
            reflectivity.append(result['R'])
        # reflectivity = np.array(reflectivity)*amplitude
        return np.array((reflectivity-measured_intensity)**2, dtype=float)
    print('running brute')
    brute_params = lmfit.minimize(fcn=optimization_fun, params=params, method='brute',  keep=5)
    fit = lmfit.minimize(optimization_fun, brute_params.candidates[0].params, max_nfev=200)
    print(f'{fit.nfev=}')
    return fit.params.valuesdict()


def get_fit_results(measurement_path, calibration_path, plot=False):
    search_pattern = re.compile(r'^.*spectrum_x(.*)_y(.*).txt$')
    search_result = search_pattern.search(measurement_path)
    lamb, intensity=np.loadtxt(measurement_path,skiprows=4,unpack=True)
    intensity=intensity-310
    lamb=np.array(lamb)
    calib_spec_path = calibration_path
    calib_data=np.loadtxt(calib_spec_path,skiprows=4,unpack=True)
    lamb_calib,intensity_calib= calib_data[0,:],calib_data[1,:]
    I_normed = intensity / (intensity_calib-310)
    I_normed = I_normed/np.max(I_normed)*0.8
    lamb_normed = lamb
    # lamb_min = min(lamb[0], lamb_calib[0])
    # lamb_max = max(lamb[-1], lamb_calib[-1])
    #
    # i_min_lamb = np.argmin(np.abs(lamb - lamb_min))
    # i_min_lamb_calib = np.argmin(np.abs(lamb_calib - lamb_min))
    # i_max_lamb = np.argmin(np.abs(lamb - lamb_max))
    # # print(len(np.abs(lamb_calib - lamb_max)))
    # i_max_lamb_calib = np.argmin(np.abs(lamb_calib - lamb_max))
    #
    # # print(i_min_lamb, i_max_lamb, i_min_lamb_calib, i_max_lamb_calib)
    #
    # I_normed = intensity[i_min_lamb:i_max_lamb] / (intensity_calib[
    #                                               i_min_lamb_calib:i_max_lamb_calib]-300)
    # # Remove this line to not norm to 0.7 any more!!!!#
    # I_normed = I_normed / np.max(I_normed)
    # lamb_normed = lamb[i_min_lamb:i_max_lamb]

    params = Parameters()
    params.add('d_mem',592,min=580,max=605,vary=True,brute_step=15)
    params.add('d_vac',1500,min=1200,max=1600,vary=True,brute_step=10)
    params.add('amplitude',1,min=0.1,max=2,vary=False,brute_step=0.5)

    tail_cutoff = 800
    index_ignore_tail = np.argmin(np.abs(lamb - tail_cutoff))

    circle_mid = (-161, -1391)
    first_lim = 75
    if (np.sqrt((float(search_result[1])-circle_mid[0])**2+(float(search_result[2])-circle_mid[1])**2)) < first_lim:
        # print(f'small {first_lim}: x{search_result[1]}_y{search_result[2]}')
        params['d_vac'].set(max=1300, min=800)
    # if (np.sqrt((float(search_result[1])-circle_mid[0])**2+(float(search_result[2])-circle_mid[1])**2)) < 62:
    #     print(f'small 88.3: x{search_result[1]}_y{search_result[2]}')
    #     params['d_vac'].set(max=1280, min=800)

    # elif (np.sqrt((float(search_result[1])-circle_mid[0])**2+(float(search_result[2])-circle_mid[1])**2)) < 75:
    #     params['d_vac'].set(max=1300, min =840)
    #     print(f'small 75: x{search_result[1]}_y{search_result[2]}')
    ### For a single line for a fixed x ###
    # if float(search_result[2]) < 20:
    #     params['d_vac'].set(max=1600, min=1100)
    # elif 20 < float(search_result[2]) < 30:
    #     params['d_vac'].set(max=1400, min=1000)
    # elif float(search_result[2]) < 185:
    #     params['d_vac'].set(max=1080, min=830)
    # elif float(search_result[2]) < 192:
    #     params['d_vac'].set(max=1280, min=800)
    # elif float(search_result[2]) > 202:
    #     params['d_vac'].set(max=1900, min=1300)




    fit_result = tmm_fit(params, lamb_normed[:index_ignore_tail], I_normed[:index_ignore_tail])

    fitted_distance = fit_result["d_vac"]
    print(fit_result)
    if plot:
        plt.ioff()
        fig_fit, ax_fit = plt.subplots()
        ax_fit.plot(lamb_normed, I_normed)
        ax_fit.set_title(f'{fit_result["d_mem"]=:4.0f} {fit_result["d_vac"]=:4.0f} x{search_result[1]}_y{search_result[2]}')
        plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], ax_fit)
        fig_fit.savefig(rf'C:\Users\fulapuser\Documents\AlexanderFuchs\Data\PN-89\membrane_OL_xy_map_10-14-17\plots/fit_x{search_result[1]}_y{search_result[2]}.png')
        plt.close(fig_fit)


    # print(fit_result['d_mem']+fit_result['d_vac'])
    return fit_result['d_mem'], fit_result['d_vac']

folder_path = os.path.abspath(r'C:\Users\fulapuser\Documents\AlexanderFuchs\Data\PN-89\membrane_OL_xy_map_10-14-17')
#search_pattern = re.compile(r'^spec_fabry_x(.*)_y(.*).txt$')
# search_pattern = re.compile(r'^spec_fabry.*.txt$')
i = 0
results_mem = []
results_vac = []
x_list = []
y_list = []
lamb_list = []



def get_regex_matches(folder_path):
    match_list = []
    search_pattern = re.compile(r'^spec.*_x(.*)_y(.*).txt$')
    for file in os.listdir(folder_path):
        search_result = search_pattern.search(file)
        if search_result:
            if (-3000 < float(search_result[1]) < 11110) and (-4330 < float(search_result[2]) < 10500):
                filepath = os.path.join(folder_path, search_result[0])
                match_list.append(filepath)
    return match_list


def use_parallel(filepath):
    search_pattern = re.compile(r'^.*spec.*_x(.*)_y(.*).txt$')
    search_result = search_pattern.search(filepath)
    print(f'x{search_result[1]}, y{search_result[2]}')
    filepath = os.path.join(folder_path,search_result[0])
    calib_spec_path = os.path.abspath(r"C:\Users\fulapuser\Documents\AlexanderFuchs\Data\PN-89\membrane_OL_xy_map_10-14-17\spectrum_x-64_y-1290.txt")
    # calib_spec_path = os.path.abspath(r"\\confocal2\Measurement_Data\Morris\FP_morris\161\mirror_for_fabryperot_calibration\iris2.5mm_18-01-10\spec_fabry_x371.45_y-383.4.txt")
    result_mem, result_vac = get_fit_results(filepath, calib_spec_path,plot=False)
    plt.show()
    results_mem.append(result_mem)
    results_vac.append(result_vac)
    x_list.append(search_result[1])
    y_list.append(search_result[2])
    return search_result[1], search_result[2], float(result_vac), float(result_mem)


plot_each_fit = False
parallel_result = Parallel(n_jobs=-1)(delayed(use_parallel)(file_path) for file_path in get_regex_matches(folder_path))
for result in parallel_result:
    if result:
        x_list.append(result[0])
        y_list.append(result[1])
        results_vac.append(result[2])
        results_mem.append(result[3])





x_array = np.array(x_list, dtype=float)
y_array = np.array(y_list, dtype=float)
matplotlib.use('qt5agg')
plt.ion()
# x_array, y_array, results_vac, results_mem = np.loadtxt(r'C:\Users\fulapuser\Documents\AlexanderFuchs\TMM_fits\plots\fit_data_61x61.txt', unpack=True)
norm = plt.Normalize(np.min(results_vac),np.max(results_vac))
cmap = plt.cm.viridis
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
scatter1 = ax1.scatter(x_array, y_array, c=results_vac, marker='s', norm=norm, cmap=cmap)
scatter2 = ax2.scatter(x_array, y_array, c=results_mem, marker='s')
ax1.set_title('Vacuume distance')
ax2.set_title('Membrane distance')
fig1.colorbar(scatter1, ax = ax1)
fig2.colorbar(scatter2, ax = ax2)
print(datetime.now() - startTime)

annot = ax1.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):
    pos = scatter1.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    # print(ind)
    # print(results_vac[ind["ind"][0]])
    text = f'{results_vac[ind["ind"][0]]:4.0f}'
    # text = "{}".format(" ".join(list(map(str,results_vac[ind["ind"][0]]))))
    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(norm(results_vac[ind["ind"][0]]))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax1:
        cont, ind = scatter1.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig1.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig1.canvas.draw_idle()

fig1.canvas.mpl_connect("motion_notify_event", hover)

# fig3,ax3=plt.subplots(1)
# ax3.plot(np.array(x_list, dtype=float), results_vac, 'x')

# data = np.column_stack([x_array,y_array,results_vac,results_mem])
# np.savetxt(r'C:\Users\fulapuser\Documents\AlexanderFuchs\TMM_fits\plots\fit_data_61x61.txt', data, header='x_pos ypos distance_vac distance_mem')


fig_1d, ax_1d = plt.subplots()
ax_1d.plot(np.array(y_list, dtype=float), results_vac, 'x')
ax_1d.set_xlabel('$\mu$m')
ax_1d.set_ylabel('distance in nm')




