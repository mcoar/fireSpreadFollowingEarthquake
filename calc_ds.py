# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 08:40:32 2022

@author: Lab User
"""


import numpy as np
from scipy import stats, spatial

#%% Calculate Damage States
def calc_ds(e, PFA=0, IDR=0):
    # Note: assume no direct damage to main run or branch pipes due to displacement
    # because not enough information on obstructions. 
    # assume enough clearance not to cause damage.
    # --------------------------------------------
    # _____INPUT____________
    # PFA: Peak Floor acceleration [g]
    # IDR: Interstory drift ratio [%]
    # _____OUTPUT___________
    # ds: damage state [0, 3] 
    #       [0: no damage, 1: slight, 2: moderate, 3: extensive]
    # cap: ratio of capacity/maximum capacity based on random variable.
    # cap_idx: ratio of capacity/maximum capacity based on weighted average of ds params.
    # p_e: probability of damage state occuring given IM
    # r: random value [0 1]
    # --------------------------------------------------------------------
    params = e.params
    p_e = list()
    r = np.random.rand()
    ds = 0
    if e.type in ['branch','mainrun']:
        par_idx = 0
        IM = PFA
    elif e.type in ['riser']:
        par_idx = 1
        IM = IDR
    for i in range(3):
        #                       beta                 xm
        dist_a = stats.lognorm(s=params[par_idx][i][1], scale=params[par_idx][i][0], loc=0)
        p = dist_a.cdf(IM)
        p_e.append(p)
        if r <= p:
            ds = i+1
    c_i = (1.0,1.0,0.9,0.0)
    x_i = (1.-p_e[0], p_e[0]-p_e[1], p_e[1]-p_e[2], p_e[2])
    
    cap_idx = np.dot(c_i,x_i)
    cap = c_i[ds]
        
    
    return ds, cap_idx, cap, p_e, r