#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:35:33 2023

@author: mzannese
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 12}

matplotlib.rc('font', **font)
import os
os.chdir('//Users/mzannese/Documents/Recherche/CDD-IAS/Script_Alice/')
import integrate_lines as lines
from astropy.io import fits
import pandas as pd

#%%GTO Horsehead
path='/Users/mzannese/Documents/Recherche/CDD-IAS/Data/Horsehead/MIRI_MRS/'
data = np.loadtxt(path+'2025_02_IAS_v1.7/Template/marion/marion_small/DF1_ch3_long.dat')


wave=np.array(data[:,0])  
spectre=data[:,1]


#%%
window=[[12.2,12.3],[17.0,17.1]] #H2 v=0
lamb0=[12.2780,17.0340]

if path=='/Users/mzannese/Documents/Recherche/CDD-IAS/Data/Horsehead/MIRI_MRS/':

    R=2700
    dR=[1500,4500]

else:
    R=1000
    dR=[800,1200]


print(R)

intensity=np.zeros(len(window))
lam=np.zeros(len(window))
sigma=np.zeros(len(window))
err=np.zeros(len(window))
RR=np.zeros(len(window))
window_ref=[0,0]
for i in range(len(window)):
    fig,ax=plt.subplots(figsize=(7,20))
    
    window_ref[0]=window[i][0]
    window_ref[1]=window[i][1]
    
    count=0
    x1=np.argmin(np.absolute(wave-window_ref[0]))
    x2=np.argmin(np.absolute(wave-window_ref[1]))
    

    if window_ref[0] <lamb0[i] < window_ref[1] and x2>x1:
        intensity[i]=lines.measure_flux_gauss(window_ref,wave,spectre,lamb0[i],R,dR)[0]
        lam[i]=lines.measure_flux_gauss(window_ref,wave,spectre,lamb0[i],R,dR)[2]
        err[i]=lines.measure_flux_gauss(window_ref,wave,spectre,lamb0[i],R,dR)[1]
        sigma[i]=lines.measure_flux_gauss(window_ref,wave,spectre,lamb0[i],R,dR)[3]
        RR[i]=lines.measure_flux_gauss(window_ref,wave,spectre,lamb0[i],R,dR)[4]
        flux=lines.fit(window_ref,wave,spectre,lamb0[i],R,dR,pied=30)
        
        ax.plot(wave,spectre,c='black')
        ax.plot(wave,flux,c='r')
        ax.set_xlim(window[i][0]-0.01,window[i][1]+0.01)
        ax.set_ylim(min(flux[x1:x2])-50,max(flux[x1:x2])+50)



for i in range(0,len(intensity)):
    print(lamb0[i],lam[i],intensity[i],err[i],RR[i])
