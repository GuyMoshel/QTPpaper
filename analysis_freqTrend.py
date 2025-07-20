# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 13:42:40 2025

@author: PHshayg-lab3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cft
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Cambria'
plt.rcParams["mathtext.default"] = 'it'
plt.rcParams["mathtext.fontset"] = 'cm'

# Frequency vector [Hz]
f_vec = np.array([2.270e+09, 3.530e+09, 7.380e+09, 1.134e+10, 1.575e+10])
f_vec_GHz = f_vec * 1e-9

# Parameters
params = {}

params['0'] = {
'zeta_ms': 371777.8559460485 ,
'delta_ms': 0.002099027120467712 ,
'A0_qp': 54.21068979793384 ,
'Q_other': 1183261.214729182
}

params['1'] = {
'zeta_ms': 4276263.034626217 ,
'delta_ms': 0.005130731952475221 ,
'A0_qp': 40.00240423362706 ,
'Q_other': 2546376.5363866873
}

params['2'] = {
'zeta_ms': 582735.8803509705 ,
'delta_ms': 0.00726490542723537 ,
'A0_qp': 45.13275808364142 ,
'Q_other': 3996489.712323376
}

params['3'] = {
'zeta_ms': 149602.62737042792 ,
'delta_ms': 0.013259771103969127 ,
'A0_qp': 45.96770187862424 ,
'Q_other': 1365499.794336109
}

params['4'] = {
'zeta_ms': 288354.67224022973 ,
'delta_ms': 0.022616137528606677 ,
'A0_qp': 62.01438871319051 ,
'Q_other': 826932.4820800313
}



params['0']['zeta_ms_err'] = 133575.0018971345
params['0']['delta_ms_err'] = 0.00012721208364672057
params['0']['A0_qp_err'] = 1.2629837175294583
params['0']['Q_other_err'] = 5481.325210435782

params['1']['zeta_ms_err'] = 2374674.9899582323
params['1']['delta_ms_err'] = 0.000722415863203219
params['1']['A0_qp_err'] = 1.5912728712198394
params['1']['Q_other_err'] = 71326.00280441134

params['2']['zeta_ms_err'] = 64563.2314334486
params['2']['delta_ms_err'] = 0.0005210748812147837
params['2']['A0_qp_err'] = 0.6594549804626446
params['2']['Q_other_err'] = 99530.3142492231

params['3']['zeta_ms_err'] = 13986.542541698418
params['3']['delta_ms_err'] = 0.0009301550523287599
params['3']['A0_qp_err'] = 0.6130014963834434
params['3']['Q_other_err'] = 23431.163777744747

params['4']['zeta_ms_err'] = 18520.229558254116
params['4']['delta_ms_err'] = 0.00143312859661845
params['4']['A0_qp_err'] = 5.583791405653702
params['4']['Q_other_err'] = 21868.994124716497


# Helper function
def extract(param_key, scale=1.0, err_scale=2.0):
    values = np.array([params[str(i)][param_key] for i in range(5)]) * scale
    errors = np.array([params[str(i)][param_key + '_err'] for i in range(5)]) * err_scale * scale
    return values, errors

# Extract data
zeta_ms, zeta_ms_err = extract('zeta_ms', scale=1e-7)
delta_ms, delta_ms_err = extract('delta_ms', scale=1e2)
A0_qp, A0_qp_err = extract('A0_qp', scale=1e-1)
Q_other, Q_other_err = extract('Q_other', scale=1e-6)

fig, axs = plt.subplots(4, 1, figsize=(5, 6.5), sharex=True)
main_color = 'C0'
extra_color1 = 'C3'
extra_color2 = 'C1'
fontsize = 14

# ---------- Plot 1: zeta_ms ----------
axs[0].errorbar(f_vec_GHz, zeta_ms, yerr=zeta_ms_err, fmt='o', color=main_color, capsize=5)
# axs[0].errorbar(f_vec_GHz[1], zeta_ms[1], yerr=zeta_ms_err[1], fmt='o', color=extra_color1, capsize=5)
axs[0].errorbar(f_vec_GHz[1], 8213806.62 * 1e-7, yerr=zeta_ms_err[1], fmt='o', color=extra_color2, capsize=5)
axs[0].set_ylabel(r'$\zeta_{\mathrm{MS}} / 10^7$', fontsize=fontsize)
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# def zeta_fit(x, a, b): return a / (x - b)
# mask = [0, 2, 3, 4]
# popt, _ = cft(zeta_fit, f_vec_GHz[mask], zeta_ms[mask], p0=[0.7, 0])
fit_x = np.linspace(min(f_vec_GHz), max(f_vec_GHz), 200)
# axs[0].plot(fit_x, zeta_fit(fit_x, *popt), '--', color=main_color)

# zeta_formula = r'$\zeta_{\mathrm{MS}} = a/(f-b)$'
# axs[0].annotate(zeta_formula, (0.67,0.81), xycoords='figure fraction',fontsize=fontsize)

# ---------- Plot 2: delta_ms ----------
axs[1].errorbar(f_vec_GHz, delta_ms, yerr=delta_ms_err, fmt='o', color=main_color, capsize=5)
axs[1].errorbar(f_vec_GHz[1], 0.0043 * 1e2, yerr=delta_ms_err[1], fmt='o', color=extra_color2, capsize=5)
axs[1].set_ylabel(r'$\delta_{\mathrm{MS}}^0 / 10^{-2}$', fontsize=fontsize)
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

def delta_fit(x, a, b): return a * x + b
popt, _ = cft(delta_fit, f_vec_GHz, delta_ms, p0=[0.14, -0.15])
axs[1].plot(fit_x, delta_fit(fit_x, *popt), '--', color=main_color)

delta_formula = r'$\delta_{\mathrm{MS}}^0 = a \cdot f + b$'
axs[1].annotate(delta_formula, (0.68,0.57), xycoords='figure fraction',fontsize=fontsize)

# ---------- Plot 3: A0_qp ----------
axs[2].errorbar(f_vec_GHz, A0_qp, yerr=A0_qp_err, fmt='o', color=main_color, capsize=5)
axs[2].errorbar(f_vec_GHz[1], 43.7597 / 1e1, yerr=A0_qp_err[1], fmt='o', color=extra_color2, capsize=5)
axs[2].set_ylabel(r'$A_0^{\mathrm{qp}} / 10^1$', fontsize=fontsize)
axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# ---------- Plot 4: Q_other ----------
axs[3].errorbar(f_vec_GHz, Q_other, yerr=Q_other_err, fmt='o', color=main_color, capsize=5)
axs[3].errorbar(f_vec_GHz[1], 2922475.90 / 1e6, yerr=Q_other_err[1], fmt='o', color=extra_color2, capsize=5)
axs[3].set_ylabel(r'$Q_{\mathrm{other}} / 10^6$', fontsize=fontsize)
axs[3].set_xlabel(r'Frequency [GHz]', fontsize=fontsize)
axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Legend (single, custom handles)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=main_color, label='Fitted model', markersize=8),
    # Line2D([0], [0], marker='o', color='w', markerfacecolor=extra_color1, label='Our model (excluded from fit)', markersize=8),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=extra_color2, label='Fitted model + Single TLS\n (excluded from fit)', markersize=8),
]
axs[0].legend(handles=legend_elements, loc='upper right', fontsize=fontsize-3, frameon=True)

# Final touches
for ax in axs:
    ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig(r'C:\Users\PHshayg-lab3\OneDrive - Technion\Guy\TLS spectroscopy\article_stuff\params_vs_freq_clean.pdf', dpi=400)
plt.show()
