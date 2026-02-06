#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:01:46 2023

@author: mzannese
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
import scipy

def cont_sub(wlth, flux,wlthr0,dwlth=0.05,cont=False,fit_kind='linear'):
    # if wlthr0=='None':
    #     wlthr0=[]
    #     for i in range(0,len(wlth),5):
    #         wlthr0.append(wlth[i])
    wlthred = [wlth[i] for i in range(len(wlth)) if wlth[i]<=max(wlthr0)+dwlth and wlth[i]>=min(wlthr0)-dwlth]
    fluxred=np.zeros((len(wlthred)))
    k=0
    for i in range(len(flux)):
        if wlth[i]<=max(wlthr0)+dwlth and wlth[i]>min(wlthr0)-dwlth:
            fluxred[k] = flux[i]
            k=k+1
    count    = np.zeros((len(wlthr0)))
    fluxsamp = np.zeros((len(wlthr0)))
    for j in range(len(wlthr0)):
        if dwlth>0:
            diff=np.absolute(np.array(wlthred)-(wlthr0[j]-dwlth))
            x1=np.argmin(diff)
            diff=np.absolute(np.array(wlthred)-(wlthr0[j]+dwlth))
            x2=np.argmin(diff)
            fluxsamp[j] =  np.median(fluxred[x1:x2])
        else:
            diff=np.absolute(np.array(wlthred)-(wlthr0[j]))
            x=np.argmin(diff)
            fluxsamp[j] =  fluxred[x]

    wlthred = [wlth[i] for i in range(len(wlth)) if wlth[i]<=max(wlthr0) and wlth[i]>=min(wlthr0)]
    fluxred=np.zeros((len(wlthred)))
    
    flux_sub = np.zeros((np.shape(fluxred)))
    k=0
    for i in range(len(flux)):
        if wlth[i]<=max(wlthr0) and wlth[i]>min(wlthr0):
            fluxred[k] = flux[i]
            k=k+1
            
    flux_sub = np.zeros((np.shape(fluxred)))
    f2 = scipy.interpolate.interp1d(wlthr0, fluxsamp,kind=fit_kind)
    if cont==True:
        plt.step(wlth, flux,c='black')
        plt.plot(wlthred,f2(wlthred),c='r')
        plt.plot(wlthr0,fluxsamp,'+',c='b',markersize=5)
    flux_sub = fluxred- f2(wlthred)
    wlthred=np.array(wlthred)
    return wlthred,flux_sub 

def fit(windflux,wltharray,fluxarray,lamb0,R=3000,dR=[1000,4000],pied=3,plot=False):
    flux=np.zeros(len(fluxarray))
    def gaus_2(x,H,b,a,x0,sigma):
        return H*x+b+a*np.exp(-(x-x0)**2/(2*sigma**2)) 
    if plot==True:
        plt.step(wltharray,fluxarray)
    difference_array = np.absolute(wltharray-windflux[0])
    xmin= difference_array.argmin()
    difference_array = np.absolute(wltharray-windflux[1])
    xmax= difference_array.argmin()
    x = wltharray[xmin:xmax]
    y = fluxarray[xmin:xmax]/max(fluxarray[xmin:xmax])
    xmean= np.argmax(y)
    sig=lamb0/(R*2*np.sqrt(2*np.log(2)))
    sig1=lamb0/(dR[1]*2*np.sqrt(2*np.log(2)))
    sig2=lamb0/(dR[0]*2*np.sqrt(2*np.log(2)))
    try:
        popt,pcov = curve_fit(gaus_2,x,y,p0=[y[0]/4,0,y[xmean],lamb0,sig],bounds=[[-np.inf,-np.inf,0,max(windflux[0],lamb0-5e-3),sig1],[np.inf,np.inf,np.inf,min(lamb0+5e-3,windflux[1]),sig2]])
        if plot==True:
            plt.step(wltharray[xmin-50:xmax+50],gaus_2(wltharray[xmin-50:xmax+50],popt[0]*max(fluxarray[xmin:xmax]),popt[1]*max(fluxarray[xmin:xmax]),popt[2]*max(fluxarray[xmin:xmax]),popt[3],popt[4]),'--',c='r',label='fit')
        flux[xmin-pied:xmax+pied]=gaus_2(wltharray[xmin-pied:xmax+pied],popt[0]*max(fluxarray[xmin:xmax]),popt[1]*max(fluxarray[xmin:xmax]),popt[2]*max(fluxarray[xmin:xmax]),popt[3],popt[4])
    
    except:
        print(wltharray[xmean+xmin],'mic: Gaussien fit impossible')
        
    return flux




def measure_flux_gauss(windflux,wltharray,fluxarray,lamb0,R=3000,dR=[1000,5000],plot=False,printt=True):
    def gaus(x,a,x0,sigma):
        return a/(x*1e-6)**2*np.exp(-(x-x0)**2/(2*sigma**2))*3e8*1e-23
    def gaus_2(x,H,b,a,x0,sigma):
        return H*x+b+a*np.exp(-(x-x0)**2/(2*sigma**2)) 
    intflux=0
    err=0
    lam=0
    sig=0
    sigg=0
    RR=0
    if plot==True:
        plt.step(wltharray,fluxarray,c='black')
    difference_array = np.absolute(wltharray-windflux[0])
    xmin= difference_array.argmin()
    difference_array = np.absolute(wltharray-windflux[1])
    xmax= difference_array.argmin()
    if xmin-50<0:
        xmiin=0
    else:
        xmiin=xmin-50
    if xmax+50>len(fluxarray):
        xmaax=-1
    else:
        xmaax=xmax+50
    x = wltharray[xmin:xmax]
    y = fluxarray[xmin:xmax]/max(fluxarray[xmin:xmax])
    xmean= np.argmax(y)
    sig=lamb0/(R*2*np.sqrt(2*np.log(2)))
    sig1=lamb0/(dR[1]*2*np.sqrt(2*np.log(2)))
    sig2=lamb0/(dR[0]*2*np.sqrt(2*np.log(2)))
    try:
        popt,pcov = curve_fit(gaus_2,x,y,p0=[y[0]/4,0,y[xmean],lamb0,sig],bounds=[[-np.inf,-np.inf,0,max(windflux[0],lamb0-5e-3),sig1],[np.inf,np.inf,np.inf,min(windflux[1],lamb0+5e-3),sig2]])
        if plot==True:
            plt.plot(wltharray[max(xmin-50,0):min(xmax+50,len(wltharray)-1)],gaus_2(wltharray[xmiin:xmaax],popt[0]*max(fluxarray[xmin:xmax]),popt[1]*max(fluxarray[xmin:xmax]),popt[2]*max(fluxarray[xmin:xmax]),popt[3],popt[4]),'--',c='r',label='fit')
        
        delta_y=np.sqrt((np.absolute(np.diag(pcov))[2]/popt[2]**2)+(np.absolute(np.diag(pcov))[4]/popt[4]**2))
        y=integrate.quad(lambda x: gaus(x,popt[2]*max(fluxarray[xmin:xmax]),popt[3],popt[4]),wltharray[xmiin],wltharray[xmaax])
        if y[0]<1e-30:
            y=integrate.quad(lambda x: gaus(x,popt[2]*max(fluxarray[xmin:xmax]),popt[3]+2e-6,popt[4]),wltharray[xmiin],wltharray[xmaax])
        
        if y[0]<0 and printt==True:    
            print(wltharray[xmean+xmin],'mic: Negative intensity')
        else:
            intflux=y[0]
            lam=popt[3]
            sigg=popt[4]
            err=y[0]*delta_y#/popt[2]
            RR=lam/(sigg*2*np.sqrt(2*np.log(2)))
            
    except:
        if printt==True:
            print(wltharray[xmean+xmin],'mic: Gaussien fit impossible')
    return intflux,err,lam,sigg,RR

