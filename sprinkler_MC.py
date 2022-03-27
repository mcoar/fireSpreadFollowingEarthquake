# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:46:22 2021

@author: Lab User
"""

import wntr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytictoc import TicToc; t = TicToc()
import seaborn as sns
from scipy import stats, spatial
import pandas as pd
import pickle
import intensity as im
from calc_ds import *
from loadall import *

#%% Toggles
toggle_wn_damage = False

toggle_demo = False
toggle_plot = False
toggle_plot_flow = False
toggle_repickle = True


    
#%% UNITS: all wntr units are in SI: kg, m, s
m = 1.          # meters
sec  = 1.       # sec
inch = .0254*m  # inches
ft = 12*inch   # feet
yd = 3*ft      # yards 

s  = 10*ft       # sprinkler spacing along branch lines
xm = 10*ft      # sprinkler spacing across branch lines (distance between branches)
h  = 13*ft      # height of a floor
h1 = 18*ft      # height of the first floor
g  = 9.81*m/sec**2     # gravitational acceleration

meter2inch = 1/inch

#%% Geometry

# layout
M = 12          # number of branch line intersections with main lines
N = 6           # number of sprinklers per branch line
B = 2           # number of branch lines at each intersection
F = 9           # number of floors

storyheight = np.array([18.,13.,13.,13.,13.,13.,13.,13.,13.])*ft

#%% Get intensity measures

idx = 5
username = "Lab User"
path = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/9st_mrf_mwc/"
city = 'LaGrande'
GM_dir = 'EW'
folder = 'Output-{:s}_{:003d}{:s}'.format(city,idx,GM_dir)

pathname = path+folder
t.tic()
PFA, IDR = im.OpenSees_IMs(pathname,storyheight,ngrav=10)
t.toc('IM calc time is')
    
#%% Pipe schedule and dimensions
data = {'Nom':  [0.5,   0.75,   1.0,    1.25,   1.50,   2.0,    2.5,    3.0,    3.5,    4.0,    4.5,    5.0,    6.0],
        't':    [0.109, 0.113,  0.133,  0.140,  0.145,  0.154,  0.203,  0.216,  0.226,  0.237,  0.247,  0.258,  0.280],
        'OD':   [0.840, 1.050,  1.315,  1.660,  1.900,  2.375,  2.875,  3.500,  4.000,  4.500,  5.000,  5.563,  6.625]}
pipedf = pd.DataFrame(data)
ID = pipedf['OD']-2*pipedf['t']
pipedf['ID'] = ID
pipedf.index = pipedf['Nom']


#%% Take the network out of the pickle jar
t.tic()
f=open('wn.pickle','rb')
wn = pickle.load(f)
f.close()
# t.toc('Unpickle time is')

number_of_nodes = len(wn.node_name_list)
number_of_pipes = len(wn.pipe_name_list)
print('number of nodes: ' + str(number_of_nodes))
print('number of pipes: ' + str(number_of_pipes))
#%% Assess damage
"""
Some notes on statistics:
General - We are simulating the performance of non-structural fire protection
systems (FPS) following an earthquake. 
There are two systems, active (sprinklers) and passive (compartment dividers - 
walls and ceilings).
Each pipe in the AFPS and each wall in the PFPS will be probabilistically 
assigned to one of four damage states: 0: no damage, 1: slight, 2: moderate, 
3: extensive, based on existing fragility curves.

We want to sample enough iterations of the damage to be sure that our samples 
fully represent the possible outcomes, within some user-designated bounds.
In this case, we have decided that we want a mean number of pipes at damage 
state 1 within 1% of the actual value, with 95% confidence. We could adjust the
variable of interest, confidence level, and relative error, but these decisions
appear to suit our purposes.

We do this by setting up a while loop, where the stopping criteria is:
                              / Sn \
                    1.96 sqrt|------|  > 0.01
                              \ n  /
     
where Sn is the variance, n is the number of samples, 0.01 is the relative
error, and 1.96 is the Z-value for a 95% confidence lev
el.

It is further noted that the likelihood of the  occurence of any earthquake in 
our record set is assumed to be uniformly distributed. Therefore, the 
structural response of the structure in question is also uniformly distributed,
and will be treated as a uniformly distributed input in the damage state 
calculations.
See:
A Brief Survey of Stopping Rules in Monte Carlo simulations by Gilman, 1968
Quantitative Finance Stack Exchange Post:
    https://quant.stackexchange.com/questions/21764/stopping-monte-carlo-simulation-once-certain-convergence-level-is-reached?newreg=1bb0323ff12c4329a93acf0108746b59
"""


n_series = 3
        
case = 1
n = 1
params = []
dss = []
ds_list = []

# Initialize statistical values Xbar (mean of the samples), Ssq (standard 
#   deviation of the samples), eps (stopping criteria)

Xbar_ds1 = 0.
Xbar_old = 0.
Ssq_ds1 = 0.
eps = 100.
abs_err = 1000

t.tic()
# Loop through set of samples
while (abs_err > 0.01 and eps > 0.01) or n < 30:
    # Initialize counter for pipe damage states
    ds_counter = np.zeros((1,4))
    # Loop through each pipe in the network
    for ename in wn.pipe_name_list:
        e = wn.get_link(ename)
        # Calculate the damage state
        e.ds, e.cap_idx, e.cap, p_e, r = calc_ds(e,PFA[e.floor],IDR[e.floor])
        # Count the total number of pipes in each damage state
        ds_counter[0,e.ds]+=1
    # Append the ds counter to the damage state counter list
    ds_list.append(ds_counter)
    # Calculate statistical values for ds1
    X_ds1    = ds_counter[0,1]
    Xbar_old = Xbar_ds1
    Xbar_ds1 = Xbar_ds1 + (X_ds1-Xbar_ds1)/(n)
    if n>1:
        Ssq_ds1 = (1.-1./(n-1.))*Ssq_ds1 + n*(Xbar_ds1 - Xbar_old)**2
    eps = 1.96*np.sqrt(Ssq_ds1/n)
    abs_err = Xbar_ds1-Xbar_old
    print('Xbar: {0:0.2f}, Ssq: {1:0.2f}, abs_err: {2:0.2f}, eps: {2:0.2f}'.format(Xbar_ds1,Ssq_ds1,abs_err,eps))
    n += 1
# Convert the damage state counter list to a numpy array.
# Note, this isn't an efficient way of tracking, but necessary since we don't 
#   know the number of samples ahead of time.
ds_array = np.array(ds_list)
print('Number of samples is: {:d}'.format(n))
t.toc('Damage calcs time is')


#%% Pickle the network 
path = r"C:/Users/Lab User/OneDrive/Documents/OpenSees/PythonFiles-OS/"
if toggle_repickle:
    t.tic()
    PIK = 'wn'+str(n)+'.pickle'
    with open(path+PIK,'wb') as f:
        pickle.dump([wn,ds_array,p_e,r],f)
        pickle.dump(ds_array,f)
    items = loadall(path+PIK)
    # test1 = next(items)
    # test2 = next(items)
    t.toc('repickle time is')
    


#%% Add a leak
# Shi ORourke 2008 Round Crack 5.4.4.2
# crack assumed to occur at the middle of pipe
# uses small-angle approximation so open angle must be converted to radians
if toggle_wn_damage:
    t.tic()
    for ename in wn.pipe_name_list:
        e = wn.get_link(ename)
        nname = e.start_node_name
        if e.ds == 2:
            theta = np.radians(0.5)             # radians
            area_modifier = 0.25
            leak_area = 0.5*np.pi*theta*np.power(e.diameter,2)*area_modifier
        elif e.ds == 3:
            theta = np.radians(0.5)             # radians
            area_modifier = 1.0
            leak_area = 0.5*np.pi*theta*np.power(e.diameter,2)*area_modifier
        if e.ds>=2:
            wn = wntr.morph.split_pipe(wn,ename,ename+'_B',ename+'_leak_node')
            leak_node = wn.get_node(ename+'_leak_node')
            leak_node.add_leak(wn,leak_area,start_time=0.,end_time=end_time)
    t.toc('Leak creation time is')
    number_of_leaks = len(wn.node_name_list) - number_of_nodes 
    print('Total number of leaks: ' + str(number_of_leaks))










