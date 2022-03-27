# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:09:38 2021

@author: Lab User
"""

import numpy as np

username = "Lab User"
path = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/9st_mrf_mwc/"
city_list = ['LaGrande','Seattle']
GM_dir_list = ['EW','NS']
n_trials = 33

nc_list = []        # non-convergence list


print('Check each seismic acceleration response at second story for convergence')

for city in city_list:
    for direction in GM_dir_list:
        for idx in range(n_trials+1):
            try:             
                folder = 'Output-{:s}_{:003d}{:s}'.format(city,idx,direction)
                pathname = path+folder
                u = np.loadtxt(pathname+'/AccF2.out', usecols=1)
            except:
                # print(folder + ' doesn\'t exist')
                continue
            if u[-20]>1000:
                nc_list.append(folder)
print('======================================')
print('{} records don\'t converge'.format(len(nc_list)))     
print(nc_list)

d= {}
for rec in nc_list:
    pathname = path+rec
    u = np.loadtxt(pathname+'/AccF2.out', usecols=1)
    idx = next(i for i,val in enumerate(u) if val > 1000)
    d[rec]=idx
    
fast_fail = min(d,key=d.get)
print('{} fails fastest at timestep {}'.format(fast_fail,d[fast_fail]))
    
