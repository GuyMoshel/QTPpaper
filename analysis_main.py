# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:07:10 2024

@author: PHshayg-lab3
"""

import numpy as np
from scipy.special import ellipk, k0
from scipy.optimize import curve_fit as cft
from scipy.optimize import minimize, leastsq
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
from time import time
import cProfile
import pandas as pd
from scipy.integrate import quad, dblquad
from scipy.special import ellipkinc, ellipeinc
from functools import lru_cache
from scipy import interpolate
import matplotlib
matplotlib.rcParams['font.family'] = 'Cambria'
matplotlib.rcParams["mathtext.default"] = 'it'
matplotlib.rcParams["mathtext.fontset"] = 'cm'

#%% functions

def CPW_c(a,b,ep_s):

    k0 = a/b
    k0_tag = np.sqrt(1-k0**2)
    
    ep0 = 8.854e-12
    # c = 3e8
    
    C = 2*ep0*(ep_s+1)*ellipk(k0)/ellipk(k0_tag)
    # ep_eff = (1+ep_s/2)
    # v_ph = c/np.sqrt(ep_eff)
    # Z = 30*np.pi/np.sqrt(ep_eff) * ellipk(k0_tag)/ellipk(k0)
    
    return C


class fitClass:

    def __init__(self):
        pass
    
    
    def convert_params(self,A0,B0,t,ep):
        zeta = (self.U/self.kB)*B0*ep**2/self.ep_s**2
        delta0 = (self.kB/self.U)*A0*np.pi*zeta/(self.wavelength*t*ep)
        print('\'zeta_ms\':',zeta,',')
        print('\'delta_ms\':',delta0,',')
    
    def f_ab(self,x,y):
        w=x+1j*y
        return 1/np.sqrt((w**2-self.a**2)*(w**2-self.b**2))
    
    @staticmethod
    def integrand_m(x,V0,T,A0,B0,t,inst):
        C = (V0*inst.b/ellipk(inst.ktag))**2 * np.tanh(inst.U/2/inst.kB/T)
        m = -(1/T)*B0*C*inst.f_ab(x,t).real**2
        return A0*T*(ellipeinc(np.pi/2,m) - ellipkinc(np.pi/2,m))
        
    @lru_cache(maxsize = 100) 
    def P_m(self,V0,T,A0,B0,t):
        return np.sum(self.integrand_m(np.append(self.x1,self.x3),V0,T,A0,B0,t,self).flatten()) * self.dx
    
    @lru_cache(maxsize = 100) 
    def Q_qp(self,T,A0_qp):
        return A0_qp / ( np.exp(-self.delta/(self.kB*T))*k0(self.U/(2*self.kB*T))*np.sinh(self.U/(2*self.kB*T)) )
    
    @lru_cache(maxsize = 100) 
    def gamma1(self,T,C0,U_TLS):
        return (C0/hbar) * U_TLS/np.tanh(U_TLS/(2*self.kB*T))
    
    @lru_cache(maxsize = 100) 
    def gamma2(self,T,C0,U_TLS):
        return 0.5*self.gamma1(T,C0,U_TLS) + 1e-3*self.kB*T/self.hbar
    
    @lru_cache(maxsize = 100) 
    def kappa(self,V0,T,C0,U_TLS,D0):
        return D0*V0**2/(self.gamma1(T,C0,U_TLS)*self.gamma2(T,C0,U_TLS))

    @lru_cache(maxsize = 100) 
    def Q_single_TLS(self,V0,T,Q0_single_TLS,C0,U_TLS,D0):
        return (Q0_single_TLS/(2*np.pi*self.f)) * ( self.gamma2(T,C0,U_TLS)**2 * (1+self.kappa(V0,T,C0,U_TLS,D0)) + (U_TLS/self.hbar - 2*np.pi*self.f)**2 ) / ( self.gamma2(T,C0,U_TLS) * np.tanh(U_TLS/(2*self.kB*T)) )

    def Qi_model_sergei(self,V0,T,A0_ms,B0_ms,t_ms,A0_qp,Q_other):
        P_TLS = self.P_m(V0,T,A0_ms,B0_ms,t_ms)
        W = 2*self.wavelength*self.c_cpw*V0**2 
        Q_TLS = 2*np.pi*self.f * W / P_TLS
        Q_qp = self.Q_qp(T,A0_qp) 
        
        return 1/(1/Q_TLS + 1/Q_qp + 1/Q_other)
    
    def Qi_model_sergei_single_TLS(self,V0,T,A0_ms,B0_ms,t_ms,A0_qp,Q_other,Q0_single_TLS,C0,U_TLS,D0):
        P_TLS = self.P_m(V0,T,A0_ms,B0_ms,t_ms)
        W = 2*self.wavelength*self.c_cpw*V0**2 
        Q_TLS = 2*np.pi*self.f * W / P_TLS
        Q_qp = self.Q_qp(T,A0_qp) 
        
        Q_single_TLS = self.Q_single_TLS(V0,T,Q0_single_TLS,C0,U_TLS,D0)
        
        return 1/(1/Q_TLS + 1/Q_single_TLS + 1/Q_qp + 1/Q_other)

    def SE(self, params=None):
        for i,k in enumerate(self.params_to_minimize_list):
            self.params_dict[k] = params[i]
        
        se = 0
        if self.model_name == 'sergei':
            for Qi,Qie,V0i,Ti in zip(self.Qi,self.Qi_err,self.V0,self.T):
                se += (self.Qi_model_sergei(V0i,Ti,**self.params_dict) - Qi)**2 / Qie**2
        if self.model_name == 'sergei_single_TLS':
            for Qi,Qie,V0i,Ti in zip(self.Qi,self.Qi_err,self.V0,self.T):
                se += (self.Qi_model_sergei_single_TLS(V0i,Ti,**self.params_dict) - Qi)**2 / Qie**2
        return se
     
         
    def plot_Qi_vs_T_for_P(self,P,T,Qi,V0,title='',fignum=100,fontsize=15):
        plt.figure(fignum)
        plt.title(title,fontsize=fontsize)
        P_list = np.flip(np.unique(P))
        for p in P_list:
            T_temp = T[P==p]
            T_temp, indices = np.unique(T_temp,return_index=True)
            Qi_temp = Qi[P==p][indices]
            Qi_err_temp = Qi_err[P==p][indices]
            
            Qi_fit_temp = np.zeros(T_temp.shape)
            if self.model_name == 'sergei':
                for i,(V0i,Ti) in enumerate(zip(V0[P==p][indices],T_temp)):
                    Qi_fit_temp[i] = self.Qi_model_sergei(V0i,Ti,**self.params_dict)
            elif self.model_name == 'sergei_single_TLS':
                for i,(V0i,Ti) in enumerate(zip(V0[P==p][indices],T_temp)):
                    Qi_fit_temp[i] = self.Qi_model_sergei_single_TLS(V0i,Ti,**self.params_dict)
            elif self.model_name == 'sergei_3_TLS':
                for i,(V0i,Ti) in enumerate(zip(V0[P==p][indices],T_temp)):
                    Qi_fit_temp[i] = self.Qi_model_sergei_3_TLS(V0i,Ti,**self.params_dict)
            elif self.model_name == 'de_leon':
                for i,(V0i,Ti) in enumerate(zip(V0[P==p][indices],T_temp)):
                    Qi_fit_temp[i] = self.Qi_model_de_leon(V0i,Ti,**self.params_dict)
            
            T_temp_vec = np.linspace(T_temp[0],T_temp[-1],100)
            Qi_fit_temp_interp = interpolate.CubicSpline(T_temp,Qi_fit_temp)
            
            h = plt.errorbar(T_temp*1e3,Qi_temp*1e-6,Qi_err_temp*1e-6,fmt='o',capsize=4,label=f'{p}dBm')
            plt.plot(T_temp_vec*1e3, Qi_fit_temp_interp(T_temp_vec)*1e-6,'-',color=h[0].get_color())
            
        plt.xticks(fontsize=fontsize)    
        plt.yticks(fontsize=fontsize)    
        plt.xlabel('$T$ [mK]',fontsize=fontsize)
        plt.ylabel(r'$Q_{\mathrm{i}} / 10^6$',fontsize=fontsize)
        plt.legend(fontsize=fontsize-2, loc='upper right')
        plt.tight_layout()


    def plot_invQi_vs_P_for_T(self,P,T,Qi,V0,Tmax=0.2,title='',fignum=100,fontsize=15):
        plt.figure(fignum)
        plt.title(title,fontsize=fontsize)
        T_list = T[T<=Tmax]
        T_list = np.unique(T_list)
        for t in T_list:
            P_temp = P[T==t]
            P_temp, indices = np.unique(P_temp,return_index=True)
            Qi_temp = Qi[T==t][indices]
            Qi_err_temp = Qi_err[T==t][indices]
            
            P_temp_vec = np.linspace(P_temp[0],P_temp[-1],100)
            Qi_fit_temp = np.zeros(P_temp.size)
            if self.model_name == 'sergei':
                for i,(V0i,Ti) in enumerate(zip(V0[T==t][indices],T[T==t][indices])):
                    Qi_fit_temp[i] = self.Qi_model_sergei(V0i,Ti,**self.params_dict)
            elif self.model_name == 'de_leon':
                for i,(V0i,Ti) in enumerate(zip(V0[T==t][indices],T[T==t][indices])):
                    Qi_fit_temp[i] = self.Qi_model_de_leon(V0i,Ti,**self.params_dict)
                    
            Qi_fit_temp_interp = interpolate.CubicSpline(P_temp,Qi_fit_temp)
            
            h = plt.errorbar(P_temp,1/(Qi_temp*1e-6),Qi_err_temp/Qi_temp**2 * 1e6,fmt='o',capsize=4,label=f'{int(t*1e3)}mK')
            plt.plot(P_temp_vec, 1/(Qi_fit_temp_interp(P_temp_vec)*1e-6),'-',color=h[0].get_color())
            
        plt.xticks(fontsize=fontsize)    
        plt.yticks(fontsize=fontsize)    
        plt.xlabel('$P$ [dBm]',fontsize=fontsize)
        plt.ylabel(r'$1/(Q_i / 10^6)$',fontsize=fontsize)
        plt.legend(fontsize=fontsize-2, loc='upper right')
        plt.tight_layout()
        
        

#%% load data
       
inst = fitClass()

data_pd = pd.read_csv(r'C:\Users\PHshayg-lab3\OneDrive - Technion\Guy\TLS spectroscopy\reduce stderr\QTP_all_good.csv')

data = {}
f_pd = data_pd['f[GHz]']
f_names = np.unique(f_pd.to_numpy())
for i,f in enumerate(f_names):
    data[f'res{i}'] = {}
    data[f'res{i}']['f_nominal[GHz]'] = f
    data[f'res{i}']['T[mK]'] = data_pd[f_pd==f]['T[mK]'].to_numpy()
    data[f'res{i}']['P[dBm]'] = data_pd[f_pd==f]['P[dBm]'].to_numpy()
    data[f'res{i}']['f[Hz]'] = data_pd[f_pd==f]['fr'].to_numpy()
    data[f'res{i}']['f_err[Hz]'] = data_pd[f_pd==f]['fr_err'].to_numpy()
    data[f'res{i}']['Qi'] = data_pd[f_pd==f]['Qi_dia_corr'].to_numpy()
    data[f'res{i}']['Qi_err'] = data_pd[f_pd==f]['Qi_dia_corr_err'].to_numpy() 
    data[f'res{i}']['Ql'] = data_pd[f_pd==f]['Ql'].to_numpy()
    data[f'res{i}']['Ql_err'] = data_pd[f_pd==f]['Ql_err'].to_numpy()
    data[f'res{i}']['Qe'] = data_pd[f_pd==f]['Qc_dia_corr'].to_numpy()
    data[f'res{i}']['Qe_err'] = data_pd[f_pd==f]['absQc_err'].to_numpy() # !!! maybe mistake!!!


n = 4
P_valid = [10,0,-10,-20,-30,-40,-50]
attn_full = [-74.2,-75.5,-79.4,-83.4,-87.8]




a = 10.5e-6
b = 21.5e-6
ktag = np.sqrt(1-(a/b)**2)
inst.a = a
inst.b = b
inst.ktag = ktag


bmax = 100e-6 
dx = 0.1e-6
inst.dx = dx

f_vec = np.array([data[f'res{i}']['f_nominal[GHz]']*1e9 for i in np.arange(5)])
f = f_vec[n]
c = 3e8
kB = 1.38e-23
inst.kB = kB
hbar = 1.055e-34
inst.hbar = hbar
h = 6.626e-34
U = h*f
ep_s = 11
inst.ep_s = ep_s
ep_eff = (ep_s+1)/2
L = c/(4*f*np.sqrt(ep_eff))
dz = 200e-6
wavelength = c/(f*np.sqrt(ep_eff))
inst.wavelength = wavelength
inst.f = f
inst.U = U
inst.vol = L*2*b*180e-9

delta = 4.43*1.764*kB
inst.delta = delta

Ry = 13.6*1.6e-19
m_Ta = 3e-25
rho_Ta = 1665
N0 = 10/Ry/(m_Ta/rho_Ta)
inst.N0 = N0

c_cpw = CPW_c(a,b,ep_s)
inst.c_cpw = c_cpw


P = data[f'res{n}']['P[dBm]']
mask = np.where(np.isin(P,P_valid))
P = P[mask]
T = data[f'res{n}']['T[mK]'][mask]*1e-3
inst.T = T
Qi = data[f'res{n}']['Qi'][mask]
Qi_err = data[f'res{n}']['Qi_err'][mask]
Ql = data[f'res{n}']['Ql'][mask]
Qe = data[f'res{n}']['Qe'][mask]
inst.Qi = Qi
inst.Qi_err = Qi_err

attn = attn_full[n]
Pline = 10**((P+attn)/10) * 1e-3
Pres = Pline*2*Ql**2/(2*np.pi*f*Qe)
V0 = np.sqrt(Pres*2*50)
inst.V0 = V0

x1 = np.arange(0,a,dx)
x2 = np.arange(a,b,dx)
x3 = np.arange(b,bmax,dx)
inst.x1 = x1
inst.x2 = x2
inst.x3 = x3



#%% plot sergei model

inst.model_name = 'sergei'


ep_ms = 5
t_ms = 2e-9



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


zeta_ms = params[f'{n}']['zeta_ms']
delta_ms = params[f'{n}']['delta_ms']
A0_qp = params[f'{n}']['A0_qp']
Q_other = params[f'{n}']['Q_other']

A0_ms = wavelength*t_ms*delta_ms*ep_ms/(np.pi*zeta_ms)
B0_ms = zeta_ms*ep_s**2/ep_ms**2


inst.params_to_minimize_list = ['A0_ms','B0_ms','A0_qp','Q_other']
inst.params_dict = dict(
                        A0_ms = A0_ms,
                        B0_ms = B0_ms,
                        t_ms = t_ms,
                        A0_qp = A0_qp,
                        Q_other = Q_other)


p0 = np.zeros(len(inst.params_to_minimize_list))
for i,k in enumerate(inst.params_to_minimize_list):
    p0[i] = inst.params_dict[k]
inst.p0 = p0
print('p0 =',p0)
inst.convert_params(p0[0],p0[1],t_ms,ep_ms)


fontsize = 15
fignum = 1
plt.figure(fignum)
plt.clf()
title = r'$f_0 = $'+f'{np.round(f*1e-9,2)}GHz  '+inst.model_name
# P_list = np.flip(np.unique(P))
inst.plot_Qi_vs_T_for_P(P,T,Qi,V0,title=title,fignum=fignum,fontsize=fontsize)

# Tmax = 0.15
# inst.plot_invQi_vs_P_for_T(P,T,Qi,V0,model_name='sergei',Tmax=Tmax,title=title,fignum=fignum,fontsize=fontsize)
# inst.plot_invQi_vs_P_for_T(P,T,Qi,V0,Tmax=Tmax,title=title,fignum=fignum,fontsize=fontsize)
    

name = f'Qi_fit_n{n}'

# plt.annotate('(e)', (0.02,0.95), xycoords='figure fraction', fontsize=fontsize+4)
# plt.annotate(r'$f_0 = $'+f'{np.round(f*1e-9,2)}GHz', (0.43,0.25), xycoords='figure fraction', fontsize=fontsize+4)
# plt.tight_layout()
# plt.savefig(r'C:\Users\PHshayg-lab3\OneDrive - Technion\Guy\TLS spectroscopy\article_stuff'+'\\'+name+'.pdf', dpi = 400)



#%% plot sergei model + single TLS

inst.model_name = 'sergei_single_TLS'


ep_ms = 5
t_ms = 2e-9


params = {}


# params['1'] = {
# 'zeta_ms': 8213810.077163223 ,
# 'delta_ms': 0.0043015497821276795 ,
# 'A0_qp': 43.75974070582732,
# 'Q_other': 2922290.7815226577,
# 'Q0_single_TLS': 261.4558325132516,
# 'C0': 1.7135638426373753e-05,
# 'D0': 1.5857174838198894e+23,
# 'U_TLS': 2.6503999999999997e-23
# }

params['1'] = {
'zeta_ms': 8213810.077163223 ,
'delta_ms': 0.0043015497821276795e-40 ,
'A0_qp': 43.75974070582732e40,
'Q_other': 2922290.7815226577,
'Q0_single_TLS': 261.4558325132516,
'C0': 1.7135638426373753e-05,
'D0': 1.5857174838198894e+23,
'U_TLS': 40e9*h
}


# T_sample = 0.4
# V0_sample = max(V0)
# Gamma1 = params['1']['C0']*params['1']['U_TLS']/np.tanh(params['1']['U_TLS']/(2*kB*T_sample))
# Gamma2 = Gamma1/2 + 0.001*kB*T_sample/hbar
# print(Gamma1 /2/np.pi * 1e-6)
# kappa = params['1']['D0']*V0_sample**2/(Gamma1*Gamma2)
# print('f_TLS =',params['1']['U_TLS']/h * 1e-9,'GHz')
# print('Gamma_1/2 min =',params['1']['C0']*params['1']['U_TLS']/np.tanh(params['1']['U_TLS']/(2*kB*0.01)) /2 * 1e-9,'GHz')
# print('Gamma_1/2 max =',params['1']['C0']*params['1']['U_TLS']/np.tanh(params['1']['U_TLS']/(2*kB*0.5)) /2 * 1e-9,'GHz')
# print('Gamma res =',0.001*kB*0.5/hbar * 1e-9,'GHz')
# print('kappa =',kappa * 1e-9,'GHz')
# print('\n')

zeta_ms = params[f'{n}']['zeta_ms']
delta_ms = params[f'{n}']['delta_ms']
A0_qp = params[f'{n}']['A0_qp']
Q_other = params[f'{n}']['Q_other']
Q0_single_TLS = params[f'{n}']['Q0_single_TLS']
C0 = params[f'{n}']['C0']
U_TLS = params[f'{n}']['U_TLS']
D0 = params[f'{n}']['D0']

A0_ms = wavelength*t_ms*delta_ms*ep_ms/(np.pi*zeta_ms)
B0_ms = zeta_ms*ep_s**2/ep_ms**2


# inst.params_to_minimize_list = ['A0_ms','B0_ms','A0_qp','Q_other','Q0_single_TLS','C0','U_TLS','D0']
inst.params_to_minimize_list = ['A0_ms','B0_ms','A0_qp','Q_other','Q0_single_TLS','C0','D0']
# inst.params_to_minimize_list = ['Q_other','Q0_single_TLS','C0','D0']
inst.params_dict = dict(
                        A0_ms = A0_ms,
                        B0_ms = B0_ms,
                        t_ms = t_ms,
                        A0_qp = A0_qp,
                        Q_other = Q_other,
                        Q0_single_TLS = Q0_single_TLS,
                        C0 = C0,
                        D0 = D0,
                        U_TLS = U_TLS
                        )


p0 = np.zeros(len(inst.params_to_minimize_list))
for i,k in enumerate(inst.params_to_minimize_list):
    p0[i] = inst.params_dict[k]
inst.p0 = p0
print('p0 =',p0)
# inst.convert_params(p0[0],p0[1],t_ms,ep_ms,layer='ms')


fontsize = 15
fignum = 1
plt.figure(fignum)
plt.clf()
title = r'$f_0 = $'+f'{np.round(f*1e-9,2)}GHz  '+inst.model_name
# P_list = np.flip(np.unique(P))
inst.plot_Qi_vs_T_for_P(P,T,Qi,V0,title='',fignum=fignum,fontsize=fontsize)

# Tmax = 0.15
# inst.plot_invQi_vs_P_for_T(P,T,Qi,V0,model_name='sergei',Tmax=Tmax,title=title,fignum=fignum,fontsize=fontsize)
    

# name = f'Qi_fit_n{n}_1TLS'

# plt.annotate(r'$f_0 = $'+f'{np.round(f*1e-9,2)}GHz', (0.43,0.25), xycoords='figure fraction', fontsize=fontsize+4)
# plt.tight_layout()
# plt.savefig(r'C:\Users\PHshayg-lab3\OneDrive - Technion\Guy\TLS spectroscopy\article_stuff'+'\\'+name+'.pdf', dpi = 400)



#%% calcualte errors sergei model

from scipy.optimize import curve_fit as cft
from scipy.signal import find_peaks

zeta_ms = params[f'{n}']['zeta_ms']
delta_ms = params[f'{n}']['delta_ms']
A0_qp_0 = params[f'{n}']['A0_qp']
Q_other_0 = params[f'{n}']['Q_other']

A0_ms_0 = wavelength*t_ms*delta_ms*ep_ms/(np.pi*zeta_ms)
B0_ms_0 = zeta_ms*ep_s**2/ep_ms**2

def func(x,a,b,c):
    return a*x**2 + b*x + c

inst.params_to_minimize_list = ['A0_ms','B0_ms','A0_qp','Q_other']
p = np.zeros(len(inst.params_to_minimize_list))

N = 11
se = np.zeros(N)

fontsize = 15

fignum = 10
plt.figure(fignum)
plt.subplots(1,4,num=fignum)
plt.clf()

inst.params_dict = dict(
                        A0_ms = A0_ms_0,
                        B0_ms = B0_ms_0,
                        t_ms = t_ms,
                        A0_qp = A0_qp_0,
                        Q_other = Q_other_0)
se0 = inst.SE()


A0_ms = np.linspace(A0_ms_0*0.91,A0_ms_0*1.1,N)
for i,A0_ms_i in enumerate(A0_ms):
    inst.params_dict = dict(
                            A0_ms = A0_ms_i,
                            B0_ms = B0_ms_0,
                            t_ms = t_ms,
                            A0_qp = A0_qp_0,
                            Q_other = Q_other_0)
    se[i] = inst.SE()
se_rel = (se-se0)/se0 * 100
plt.subplot(141)
plt.xlabel('A0_ms',fontsize=fontsize)
plt.plot(A0_ms, se_rel,'o')
plt.xticks(fontsize=fontsize)  
plt.plot([A0_ms[0],A0_ms[-1]],[5,5])
plt.yticks(fontsize=fontsize)    
plt.ylabel(r'$\Delta$SE / SE0 [%]',fontsize=fontsize)

# a = 1e42
# x0 = 2.4e-20
# p0 = [a,-2*a*x0,a*x0**2]
popt,pcov = cft(func,A0_ms,se_rel,[0,0,0])
xvec = np.linspace(A0_ms[0],A0_ms[-1],100)
plt.plot(xvec,func(xvec,*popt),'--k')
idx = find_peaks(-abs(func(xvec,*popt) - 5),distance=30)[0]
plt.plot(xvec[idx],func(xvec[idx],*popt),'xr')
A0_ms_err = (xvec[idx[1]]-xvec[idx[0]])/2


B0_ms = np.linspace(B0_ms_0*0.9,B0_ms_0*1.15,N)
for i,B0_ms_i in enumerate(B0_ms):
    inst.params_dict = dict(
                            A0_ms = A0_ms_0,
                            B0_ms = B0_ms_i,
                            t_ms = t_ms,
                            A0_qp = A0_qp_0,
                            Q_other = Q_other_0)
    se[i] = inst.SE()
se_rel = (se-se0)/se0 * 100 
plt.subplot(142)      
plt.xlabel('B0_ms',fontsize=fontsize)
plt.plot(B0_ms, se_rel,'o')
plt.xticks(fontsize=fontsize)  
plt.plot([B0_ms[0],B0_ms[-1]],[5,5])
plt.yticks(fontsize=fontsize)    

popt,pcov = cft(func,B0_ms,se_rel,[0,0,0])
xvec = np.linspace(B0_ms[0],B0_ms[-1],100)
plt.plot(xvec,func(xvec,*popt),'--k')
idx = find_peaks(-abs(func(xvec,*popt) - 5),distance=30)[0]
plt.plot(xvec[idx],func(xvec[idx],*popt),'xr')
B0_ms_err = (xvec[idx[1]]-xvec[idx[0]])/2


A0_qp = np.linspace(A0_qp_0*0.93,A0_qp_0*1.03,N)
for i,A0_qp_i in enumerate(A0_qp):
    inst.params_dict = dict(
                            A0_ms = A0_ms_0,
                            B0_ms = B0_ms_0,
                            t_ms = t_ms,
                            A0_qp = A0_qp_i,
                            Q_other = Q_other_0)
    se[i] = inst.SE()
se_rel = (se-se0)/se0 * 100
plt.subplot(143)
plt.xlabel('A0_qp',fontsize=fontsize)
plt.plot(A0_qp, se_rel,'o')
plt.xticks(fontsize=fontsize)  
plt.plot([A0_qp[0],A0_qp[-1]],[5,5])
plt.yticks(fontsize=fontsize)    

popt,pcov = cft(func,A0_qp,se_rel,[0,0,0])
xvec = np.linspace(A0_qp[0],A0_qp[-1],100)
plt.plot(xvec,func(xvec,*popt),'--k')
idx = find_peaks(-abs(func(xvec,*popt) - 5),distance=30)[0]
plt.plot(xvec[idx],func(xvec[idx],*popt),'xr')
A0_qp_err = (xvec[idx[1]]-xvec[idx[0]])/2


Q_other = np.linspace(Q_other_0*0.95,Q_other_0*1.02,N)
for i,Q_other_i in enumerate(Q_other):
    inst.params_dict = dict(
                            A0_ms = A0_ms_0,
                            B0_ms = B0_ms_0,
                            t_ms = t_ms,
                            A0_qp = A0_qp_0,
                            Q_other = Q_other_i)
    se[i] = inst.SE()
se_rel = (se-se0)/se0 * 100
plt.subplot(144)
plt.xlabel('Q_other',fontsize=fontsize)
plt.plot(Q_other, se_rel,'o')
plt.xticks(fontsize=fontsize)   
plt.plot([Q_other[0],Q_other[-1]],[5,5])
plt.yticks(fontsize=fontsize)    

popt,pcov = cft(func,Q_other,se_rel,[0,0,0])
xvec = np.linspace(Q_other[0],Q_other[-1],100)
plt.plot(xvec,func(xvec,*popt),'--k')
idx = find_peaks(-abs(func(xvec,*popt) - 5),distance=30)[0]
plt.plot(xvec[idx],func(xvec[idx],*popt),'xr')
Q_other_err = (xvec[idx[1]]-xvec[idx[0]])/2


plt.tight_layout()


zeta_ms_err = zeta_ms * B0_ms_err/B0_ms_0
delta_ms_err = delta_ms * np.sqrt( (zeta_ms_err/zeta_ms)**2 + (B0_ms_err/B0_ms_0)**2 )


print(f'n={n}')
print(f"params['{n}']['zeta_ms_err'] = {zeta_ms_err}")
print(f"params['{n}']['delta_ms_err'] = {delta_ms_err}")
print(f"params['{n}']['A0_qp_err'] = {A0_qp_err}")
print(f"params['{n}']['Q_other_err'] = {Q_other_err}")


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

    

#%% perform minimization and plot

# inst.model_name = 'sergei_single_TLS'
inst.model_name = 'sergei'

t1 = time()
res = minimize(inst.SE, inst.p0, method='Nelder-Mead', bounds=[(0,None)]*len(inst.p0)) 
p_fit = res.x
t2 = time()
print('t minimization =',t2-t1)
if inst.model_name == 'sergei':
    inst.convert_params(p_fit[0],p_fit[1],t_ms,ep_ms)
    print('\'A0_qp\':',p_fit[2],',')
    print('\'Q_other\':',p_fit[3])

if inst.model_name == 'sergei_single_TLS':
    p_final = {}
    for k in inst.params_dict:
        if k in inst.params_to_minimize_list:
            i = inst.params_to_minimize_list.index(k)
            p_final[k] = p_fit[i]
            print('\''+k+'\':',p_final[k],', # (fit)')
        else:
            p_final[k] = inst.params_dict[k]
            print('\''+k+'\':',p_final[k],',')
            
    inst.convert_params(p_fit[0],p_fit[1],t_ms,ep_ms)
    # print('\'A0_qp\':',p_fit[2],',')
    # print('\'Q_other\':',p_fit[3],',')
    # print('\'Q0_single_TLS\':',p_fit[4],',')
    # print('\'C0\':',p_fit[5],',')
    # print('\'U_TLS\':',inst.params_dict['U_TLS'],',')
    # print('\'D0\':',p_fit[6])
    # print('\'U_TLS\':',p_fit[6],',')
    # print('\'D0\':',p_fit[7])
    
    print('\n')
    T_sample = 0.4
    V0_sample = max(V0)
    Gamma1 = p_final['C0']*p_final['U_TLS']/np.tanh(p_final['U_TLS']/(2*kB*T_sample))
    Gamma2 = Gamma1/2 + 0.001*kB*T_sample/hbar
    kappa = p_final['D0']*V0_sample**2/(Gamma1*Gamma2)
    f_TLS = p_final['U_TLS']/h
    print('f_TLS =',f_TLS*1e-9,'GHz')
    print('Gamma_1/2 min =',p_final['C0']*p_final['U_TLS']/np.tanh(p_final['U_TLS']/(2*kB*min(T))) * 1e-9,'GHz')
    print('Gamma_1/2 max =',p_final['C0']*p_final['U_TLS']/np.tanh(p_final['U_TLS']/(2*kB*max(T))) * 1e-9,'GHz')
    print('Gamma res =',0.001*kB*(max(T)-min(T))/hbar * 1e-9,'GHz')
    print('kappa =',kappa*1e-9,'GHz')

for i,k in enumerate(inst.params_to_minimize_list):
    inst.params_dict[k] = p_fit[i]
    

    
fontsize = 15
fignum = 2
plt.figure(fignum)
plt.clf()
title = r'$f_0 = $'+f'{np.round(f*1e-9,2)}GHz  '+inst.model_name
inst.plot_Qi_vs_T_for_P(P,T,Qi,V0,title=title,fignum=fignum,fontsize=fontsize)




