# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:20:26 2022

@author: Lab User
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import eqsig.single
import scipy.integrate as integrate
from getPGA import *

def lin_interp(X,x2,x1,y2,y1):
    Y= y1 + (y2-y1)*(X-x1)/(x2-x1)
    return Y

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def get_IMcomb(rec,T1=1.75,plot=False,toggle_print=False):
    """

    Parameters
    ----------
    rec : str
        file name (no path or suffix).
    T1 : float, optional
        fundamental period [s]. The default is 1.75.
    plot : bool, optional
        Will plot if true. The default is False.

    Returns
    -------
    IMcomb: float
    outdict:

    """
    # if toggle_print == False:
    #     blockPrint()
    a = np.loadtxt(r"C:/Users/Lab User/OneDrive/Documents/OpenSees/9st_mrf_mwc/GMs_Full_Suite/"+rec+".txt") # in g
    dt = 0.002  # time step of acceleration time series
    periods = np.linspace(0.0, 5., 100)  # compute the response for 100 periods between T=0.0s and 5.0s
    record = eqsig.AccSignal(a * 9.8, dt)
    record.generate_response_spectrum(response_times=periods)
    times = record.response_times
    # print('PGA test: {:0.3f}g'.format(getPGA(r"C:/Users/Lab User/OneDrive/Documents/OpenSees/9st_mrf_mwc/GMs_Full_Suite/"+rec+".txt")))
    

    
    #%% IMcomb
    # Procedure from Marafi, Berman, Eberhard 2016
    # Ductility-dependent intensity measure that accounts for ground-motion
    #   spectral shape and duration
    # Values from ASCE 7-10 Chapters 11, 12, and 18, and NEK dissertation
    # Table 8.4
    # Special steel moment frame
    # Office building
    
    R = 8.0         # Seismic response modification factor, Table 12.2-1
    Om = 3.0        # Overstrength factor, Table 12.2-1
    Ie = 1.0        # Importance Factor, Table 1.5.2
    T1 = 1.75       # Fundamental period of structure.
    T1D = 1.75      # [s], T1D taken equal to T1
    
    mu_max_a = 0.5*( (R/(Om*Ie))**2 + 1)
    mu_max_b = R/(Om*Ie)
    mu_max = max(mu_max_a,mu_max_b)
    
    Calf = 1.3      # Marafi
    alf  = 1.3*np.sqrt(mu_max)
    
    alfT = alf*T1
    
    
    # Get exponents
    system_ductility = 'ductile'    # 'brittle' or 'ductile' Marafi Table III
    if system_ductility == 'ductile':
        Cdur = 0.11
        Cshape = 0.72
    elif system_ductility == 'brittle':
        Cdur = 0.07
        Cshape =  0.49
        
    # Get intensity measures
    # IMdur
    IMdur = eqsig.im.calc_sig_dur(record)
    print('IMdur: {:0.2f}s'.format(IMdur))
    
    # IMshape
    area_shape = 0
    for i in range(len(times)):
        if times[i] >= T1 and times[i] <=alfT:
            area_shape += integrate.simps(record.s_a[i:i+2],times[i:i+2])
            
    Tn_idx = np.argmax(times > T1)
    Sa_Tn = lin_interp(T1,  times[Tn_idx],       times[Tn_idx-1],
                            record.s_a[Tn_idx],  record.s_a[Tn_idx-1])
    IMshape = area_shape/(Sa_Tn*(alf-1)*T1)
    
    print('Tn check: {:0.3f}s'.format(times[Tn_idx]))
    print('Sa(Tn): {:0.3f}g'.format(Sa_Tn))
    print('IMshape: {:0.3f}'.format(IMshape))
    
    IMcomb = Sa_Tn*(IMdur**Cdur)*(IMshape**Cshape)
    print('IMcomb: {:0.3f}'.format(IMcomb))
    print('PGA: {:0.3f}g'.format(record.pga/9.8))
    
    # Plotting
    if plot:
        bf, sub_fig = plt.subplots() 
        sub_fig.plot(times, record.s_a, label="eqsig")
        plt.xlabel('Period (s)'); plt.ylabel('$S_a$ (g)')
        plt.title(rec)
        plt.grid(True)
        sub_fig.vlines((T1,alfT,),0,max(record.s_a))
        plt.show()
    
    outdict = {'IMdur': IMdur,
               'Sa(Tn)': Sa_Tn,
               'IMshape': IMshape,
               'IMcomb':IMcomb,
               'PGA':PGA}
    
    # if toggle_print ==False:
    #     enablePrint()

    return IMcomb, outdict
    

