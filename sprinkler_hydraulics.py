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
import plotly
import pickle

#%% Toggles
toggle_demo = False
toggle_plot = False
toggle_plot_flow = False
toggle_repickle = False

def Sprinkler(PFA,IDR,wnin='',wnpathin='',wnpathout='',ns='',plot=False):

    
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
    
    #%% GEOMETRY
    
    # layout
    M = 12          # number of branch line intersections with main lines
    N = 6           # number of sprinklers per branch line
    B = 2           # number of branch lines at each intersection
    F = 9           # number of floors
    
    storyheight = np.array([18.,13.,13.,13.,13.,13.,13.,13.,13.])*ft
        
    #%% Pipe schedule and dimensions
    data = {'Nom':  [0.5,   0.75,   1.0,    1.25,   1.50,   2.0,    2.5,    3.0,    3.5,    4.0,    4.5,    5.0,    6.0],
            't':    [0.109, 0.113,  0.133,  0.140,  0.145,  0.154,  0.203,  0.216,  0.226,  0.237,  0.247,  0.258,  0.280],
            'OD':   [0.840, 1.050,  1.315,  1.660,  1.900,  2.375,  2.875,  3.500,  4.000,  4.500,  5.000,  5.563,  6.625]}
    pipedf = pd.DataFrame(data)
    ID = pipedf['OD']-2*pipedf['t']
    pipedf['ID'] = ID
    pipedf.index = pipedf['Nom']
    
    if wnin != '':
        wn = wnin
        print('Bringing in WN from MC setup')
    elif wnpathin == '':
        print('Generating WN from scratch')
        #%% Set up 
        wn = wntr.network.WaterNetworkModel()
        
        # add patterns
        
        wn.add_pattern('pat1', [1])
        # wn.add_pattern('pat2', [1,2,3,4,5,6,7,8,9,10])
        
        #%% generate and place nodes
        
        # place riser nodes
        for f in range(F):
            offset_x = f*xm/(2*F)
            offset_y = f*s/(F)
            iname = 'nr_{}'.format(f)
            jname = 'nr_{}'.format(f+1)
            wn.add_junction(jname, base_demand=0, elevation=h1+h*f, coordinates=(0.+offset_x,0.+offset_y))
            jnode = wn.get_node(jname)
            jnode.floor = f+1
            # place riser edges
            if f > 0:
                inode = wn.get_node(iname)
                ename = 'er_{:d}{:d}'.format(f+1,f)
                dist = spatial.distance.euclidean(inode.coordinates,jnode.coordinates)
                wn.add_pipe(ename,iname,jname,length=dist,diameter=4*inch, roughness=140.,
                            minor_loss=0.0, status='OPEN',check_valve_flag=False)
                pipe = wn.get_link(ename)
                pipe.diam_inch = 4
                pipe.material = 'steel'
                pipe.ds = 0
                pipe.end_type = 'threaded'
                pipe.type = 'riser'
                pipe.floor = f+1
                pipe.capacity = 1.
        
        # place main run nodes
        for f in range(F):
            offset_x = f*xm/(2*F)
            offset_y = f*s/(F)
            for m in range(M):
                jname = 'nm_{}_{:02d}'.format(f+1,m)
                wn.add_junction(jname, base_demand=0., elevation=h1+h*(f), 
                                coordinates=(xm/2+xm*m + offset_x,0.+offset_y))
                jnode = wn.get_node(jname)
                jnode.floor = f+1
        # place main run edges
                if m <= 3:
                    diam = 4*inch
                elif m <= 6:
                    diam = 4*inch
                elif m <= 9:
                    diam = 3*inch
                elif m <= 11:
                    diam = 2.5*inch
                if m == 0:  # use riser node
                    iname = 'nr_{}'.format(f+1)
                elif m > 0: # use previous node
                    iname = 'nm_{}_{:02d}'.format(f+1,m-1)
                inode = wn.get_node(iname)
                ename = 'em_{:d}_{:02d}'.format(f+1,m)
                dist = spatial.distance.euclidean(inode.coordinates,jnode.coordinates)
                wn.add_pipe(ename,iname,jname,length=dist,diameter=diam,roughness=140,
                            minor_loss=0.0, status='OPEN',check_valve_flag=False)
                pipe = wn.get_link(ename)
                pipe.material = 'steel'
                pipe.ds = 0
                pipe.end_type = 'threaded'
                pipe.capacity = 1.
                pipe.floor = f+1
                pipe.type = 'mainrun'
                
        # place sprinkler nodes
        for f in range(F):
            offset_x = f*xm/(2*F)
            offset_y = f*s/(F)
            for m in range(M):
                for b in range(B):
                    for n in range(N):
                        sgn = np.sign(b-0.5)        # top or bottom branch
                        jname = 'ns_{}_{:02d}_{}_{}'.format(f+1,m,b,n)
                        wn.add_junction(jname,base_demand=0., elevation=h1+h*(f),
                                    coordinates=(xm/2+xm*m+offset_x, sgn*(s/2+n*s)+offset_y))
                        jnode = wn.get_node(jname)
                        jnode.floor = f+1
        # place branch line edges
                        if n == 0:
                            diam = 2.*inch
                        elif n <= 3:
                            diam = 1.5*inch
                        elif n <= 6:
                            diam = 1.*inch
                        if n == 0:
                            iname = 'nm_{}_{:02d}'.format(f+1,m)
                        elif n > 0:
                            iname = 'ns_{}_{:02d}_{}_{}'.format(f+1,m,b,n-1)
                        inode = wn.get_node(iname)
                        dist = spatial.distance.euclidean(inode.coordinates,jnode.coordinates)
                        ename = 'eb_{:d}_{:02d}_{}_{}'.format(f+1,m,b,n)
                        wn.add_pipe(ename,iname,jname,length=dist,diameter=diam,roughness=140,
                                    minor_loss=0.0, status='OPEN',check_valve_flag=False)
                        pipe = wn.get_link(ename)
                        pipe.material='steel'
                        pipe.ds = 0
                        pipe.end_type = 'threaded'
                        pipe.capacity = 1.
                        pipe.floor = f+1
                        pipe.type = 'branch'
                        
        # place lines to city main (cm)
        wn.add_junction('nr_0', base_demand=0.0, demand_pattern=None, elevation=-h, coordinates=(0.,0.))
        wn.add_pipe('er_0','nr_0','nr_1',length=2*h,diameter=4*inch, roughness=120., 
                    minor_loss=0., status='OPEN', check_valve_flag=False)
        pipe = wn.get_link('er_0')
        pipe.material='steel'
        pipe.ds = 0
        pipe.end_type = 'threaded'
        pipe.capacity = 1.
        pipe.floor = f+1
        pipe.type = 'mainrun'
        
        wn.add_reservoir('res_cm', base_head=200*ft, head_pattern='pat1', coordinates=(xm*M,0.))
        wn.add_pipe('e_cm','res_cm','nr_0',length=xm*M, diameter=4*inch,roughness=120.,
                    minor_loss=0.,status='OPEN',check_valve_flag=False)
        pipe = wn.get_link('e_cm')
        pipe.material='steel'
        pipe.ds = 0
        pipe.end_type = 'threaded'
        pipe.capacity = 1.
        pipe.floor = f+1
        pipe.type = 'mainrun'
    else:
        print('Loading WN from pickle jar')
        f=open(wnpathin,'rb')
        wn = pickle.load(f)
        f.close()
        
    
    # Hydraulic options
    start_time = 0.
    end_time = 1000.
    wn.options.time.duration = end_time
    # wn.options.time.hydraulic_timestep = 3600.
    wn.options.hydraulic.demand_model = 'PDD'
    
    number_of_nodes = len(wn.node_name_list)
    number_of_pipes = len(wn.pipe_name_list)
    print('number of nodes: ' + str(number_of_nodes))
    
    #%% Plot
    toggle_plot = False
    if toggle_plot:
        wntr.graphics.network.plot_network(wn,node_attribute='floor',node_colorbar_label='floor',link_attribute='floor',link_colorbar_label='floor')
    
    
    
    #%% Assess damage
    
    case = 1
    if toggle_demo:
        IDR = np.ones(10)*.75
        PFA = np.ones(10)*200
    t.tic()
    # Initialize counter for pipe damage states
    ds_counter = np.zeros((1,4))
    for ename in wn.pipe_name_list:
        e = wn.get_link(ename)
        # if wnpathin == '':
        e.params = get_params(e,case)
        try:
            r = e.rvs[ns]
        except:
            r = ''
        e.ds, e.cap_idx, e.cap, p_e, r = calc_ds(e,PFA[e.floor],IDR[e.floor],r)
        # Count the total number of pipes in each damage state
        ds_counter[0,e.ds]+=1
    t.toc('Damage calcs time is')
    
    #%% Add a leak
    # Shi ORourke 2008 Round Crack 5.4.4.2
    # crack assumed to occur at the middle of pipe
    # uses small-angle approximation so open angle must be converted to radians
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
    

    
    #%% Simulation
    t.tic()
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    t.toc('Hydraulic sim time is')
    
    print(results.node['pressure'].iloc[:,0:9])
    
    #%% Pickle the network 
    if len(wnpathout) != 0:
        f=open(wnpathout,'wb')
        pickle.dump([wn,results,ds_counter,PFA,IDR],f)
        f.close()
    
    #%% Results
    
    if plot:
        # nodes
        pressure = results.node['pressure']
        pressure_at_1_hr = pressure.loc[0.,:]
        
        ax = wntr.graphics.plot_network(wn, node_attribute=pressure_at_1_hr,
                                    node_colorbar_label='Pressure (m)')
        
        ax = wntr.graphics.plot_interactive_network(wn, node_attribute=pressure_at_1_hr,
                                    filename='pressure.html',auto_open=False)
        
        # pipes
        flowrate = results.link['flowrate']
        flowrate_at_1_hr = flowrate.loc[0.,:]
        
        ax = wntr.graphics.plot_network(wn, link_attribute=flowrate_at_1_hr, 
                                        link_colorbar_label='Flowrate (m3/2)')

    return wn, results

#%% Determine damage states
def get_params(e,case, end_type='threaded'):
    meter2inch = 1./0.0254
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
    diameter_inch = float(round(e.diameter*meter2inch,2))
    if case not in switcher_a.keys():
        raise ValueError ('case not implemented.')
    vals = switcher_a.get(case, lambda:1)
    if diameter_inch not in vals.keys():
        print(e.name)
        print(e.diameter)
        print(diameter_inch)
        raise ValueError ('diameter not implemented.')
    params_a = vals.get(diameter_inch,lambda:1.)
    
    if end_type not in switcher_d.keys():
        raise ValueError ('end_type not implemented.')
    vals = switcher_d.get(end_type, lambda:1)
    if diameter_inch not in vals.keys():
        raise ValueError ('diameter not implemented.')
    params_d = vals.get(diameter_inch,lambda:1.)
    return params_a, params_d

#%% Calculate Damage States
def calc_ds(e, PFA=0, IDR=0, r=''):
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
    if r == '':
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
