# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:25:29 2021

@author: Lab User
"""
import numpy as np
from scipy import stats

#%% Determine damage states
def get_params(e,case, end_type='threaded'):
    switcher_a = {
    # case no.
    #       diam   slight       moderate    extensive    
        1: {
            1.00:[(0.66,0.64),(1.50,0.64),(2.12,0.64)],
            1.25:[(0.62,0.61),(1.36,0.61),(1.99,0.61)],
            1.50:[(0.61,0.74),(1.13,0.74),(1.56,0.74)],
            2.00:[(0.98,0.72),(1.41,0.72),(1.76,0.72)],
            2.50:[(100.,0.44),(100.,0.44),(100.,0.44)],
            3.00:[(3.73,0.61),(100.,0.61),(100.,0.61)],
            4.00:[(3.61,0.55),(100.,0.55),(100.,0.55)]
            },
        6: {
            1.00:[(0.64,0.59),(1.45,0.59),(2.04,0.59)],
            1.25:[(0.63,0.58),(1.35,0.58),(1.96,0.58)],
            1.50:[(0.60,0.70),(1.07,0.70),(1.44,0.70)],
            2.00:[(0.91,1.04),(1.56,1.04),(1.89,1.04)],
            2.50:[(1.10,0.40),(1.75,0.40),(2.27,0.40)],
            3.00:[(1.01,0.34),(1.60,0.34),(2.17,0.34)],
            4.00:[(1.56,0.48),(1.01,0.41),(1.63,0.41)]
            }
                }
    switcher_d = {
    # end type
    #       diam   slight       moderate    extensive  
        'threaded': {
            0.75:[(0.5,0.206),(2.3,0.206),(4.0,0.206)],
            1.00:[(0.5,0.146),(1.8,0.146),(3.1,0.146)],
            1.25:[(0.5,0.133),(1.4,0.133),(2.3,0.133)],
            1.50:[(0.5,1.200),(1.3,0.120),(2.0,0.120)],
            2.00:[(0.5,0.094),(.94,0.094),(1.4,0.094)],
            2.50:[(0.5,0.125),(0.9,0.125),(1.3,0.125)],
            3.00:[(0.5,0.155),(0.5,0.155),(1.1,0.155)],
            3.50:[(0.5,0.186),(0.8,0.186),(1.0,0.186)],
            4.00:[(0.5,0.216),(1.0,0.216),(1.0,0.216)],
            5.00:[(0.5,0.210),(0.6,0.210),(0.7,0.210)],
            6.00:[(0.5,0.204),(0.6,0.204),(0.6,0.204)]    
            },
        'grooved': {
            2.0:[(1.5,0.170),(5.0,0.170),(7.7,0.170)],
            2.5:[(1.3,0.140),(2.6,0.140),(3.8,0.140)],
            3.0:[(1.0,0.110),(1.9,0.110),(2.9,0.110)],
            3.5:[(0.8,0.079),(1.6,0.079),(2.4,0.079)],
            4.0:[(0.5,0.049),(1.0,0.049),(2.1,0.049)],
            5.0:[(0.6,0.049),(1.1,0.049),(1.7,0.049)],
            6.0:[(0.5,0.049),(1.0,0.049),(1.4,0.049)]
            }
                }
    if case not in switcher_a.keys():
        raise ValueError ('case not implemented.')
    vals = switcher_a.get(case, lambda:1)
    if e['D'] not in vals.keys():
        raise ValueError ('diameter not implemented.')
    params_a = vals.get(e['D'],lambda:1.)
    
    if end_type not in switcher_d.keys():
        raise ValueError ('end_type not implemented.')
    vals = switcher_d.get(end_type, lambda:1)
    if e['D'] not in vals.keys():
        raise ValueError ('diameter not implemented.')
    params_d = vals.get(e['D'],lambda:1.)
    return params_a, params_d

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
    params = e['params']
    p_e = list()
    r = np.random.rand()
    ds = 0
    if e['pipe_type'] in ['branch','mainrun']:
        par_idx = 0
        IM = PFA
    elif e['pipe_type'] in ['riser']:
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    