# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:39:25 2021

@author: Lab User
"""


import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import damage as dam
from pytictoc import TicToc; t = TicToc()
import seaborn as sns
from scipy import stats
import intensity as im


#%% TOGGLES
toggle_plot = True
toggle_demo = True

def Sprinkler(PFA, IDR,plot=True):
    
    #%% UNITS: all units given in inches
    inch = 1.       # inches
    sec  = 1.       # sec
    ft = 12*inch   # feet
    yd = 3*ft      # yards 
    s  = 10*ft       # sprinkler spacing along branch lines
    xm = 10*ft      # sprinkler spacing across branch lines (distance between branches)
    h  = 13*ft      # height of a floor
    h1 = 18*ft      # height of the first floor
    g  = .00259*inch/sec**2     # gravitational acceleration
    
    
    #%% Geometry
    
    # layout
    M = 12          # number of branch line intersections with main lines
    N = 6           # number of sprinklers per branch line
    B = 2           # number of branch lines at each intersection
    F = 9           # number of floors
    
    storyheight = np.array([18.,13.,13.,13.,13.,13.,13.,13.,13.])*12.
    
    p = ig.Graph()
    
    #%% generate and place nodes
    ## initialize lists
    # node attributes
    el = []
    x  = []
    y  = []
    nname = []
    
    # edge attributes
    ename = []
    edges = []
    diam  = []
    mat   = []
    ds    = []
    end_type = []
    capacity = []
    pipe_type = []
    floor = []
    efloor = []
    
    
    # place riser nodes
    for f in range(F):
        nname.append('nr_{}'.format(f))
        el.append(h1+h*f)
        x.append(0.)
        y.append(0.)
        floor.append(f)
    # place riser edges
        if f > 0:
            ename.append('er_{:d}{:d}'.format(f,f-1))
            edges.append((nname[-1],nname[-2]))
            diam.append(4.)
            mat.append('steel')
            ds.append(0)
            end_type.append('threaded')
            pipe_type.append('riser')
            efloor.append(f)
            capacity.append(1.)
            
        
    # place main run nodes
    for f in range(F):
        for m in range(M):
            el.append(h1+h*f)
            y.append(0.)
            x.append(xm/2+xm*m)
            nname.append('nm_{}_{:02d}'.format(f,m))
            floor.append(f)
    # place main run edges
            ename.append('em_{:d}_{:02d}'.format(f,m))
            mat.append('steel')
            ds.append(0)
            end_type.append('threaded')
            capacity.append(1.)
            efloor.append(f)
            pipe_type.append('mainrun')
            if m <= 3:
                diam.append(4.)
            elif m <= 6:
                diam.append(4.)
            elif m <= 9:
                diam.append(3.)
            elif m <= 11:
                diam.append(2.5)
            if m == 0:
                edges.append((nname[-1],'nr_{}'.format(f)))
            elif m > 0:
                edges.append((nname[-1],nname[-2]))
                
    # place sprinkler nodes
    for f in range(F):
        for m in range(M):
            for b in range(B):
               for n in range(N):
                   sgn = np.sign(b-0.5)         # top branch or bottom branch        
                   el.append(h1 + h*f)          # elevation of the floor
                   x.append(xm/2 + xm*m)        # location along x-axis
                   y.append(sgn*(s/2+n*s))      # location along y-axis
                   floor.append(f)
                   nname.append('ns_{}_{:02d}_{}_{}'.format(f,m,b,n))
    # place branch line edges
                   ename.append('eb_{:d}_{:02d}_{}_{}'.format(f,m,b,n))
                   mat.append('steel')
                   ds.append(0)
                   end_type.append('threaded')
                   capacity.append(1.)
                   efloor.append(f)
                   pipe_type.append('branch')
                   if n == 0:
                       diam.append(2.)
                   elif n <= 3:
                       diam.append(1.5)
                   elif n <= 6:
                       diam.append(1.)
                   if n == 0:
                       edges.append((nname[-1],'nm_{}_{:02d}'.format(f,m)))
                   elif n > 0:
                       edges.append((nname[-1],nname[-2]))       
    
    v = {'name':nname, 'x':x, 'y':y, 'el':el,'floor':floor}
    e = {'name':ename,'D':diam,'material':mat,'end_type':end_type,
         'pipe_type':pipe_type,'floor':efloor}
    
    p.add_vertices(len(v['name']), v)
    p.add_edges(edges,e)
    
    if toggle_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        
        ax.scatter(x,y,el)
        ax.set_xlabel('X (ft)')
        ax.set_ylabel('Y (ft)')
        ax.set_zlabel('Elevation (ft)')
                
    # PFA, IDR = im.OpenSees_IMs(pathname,storyheight)
    
    #%% Assess damage
            
    case = 1
    count = 0
    params = []
    dss = []
    toggle_demo = True
    if toggle_demo:
        IDR = np.ones(10)*.05
        PFA = np.ones(10)*0.3
    t.tic()
    for e in p.es:
        e['params'] = get_params(e,case)
        e['ds'], e['cap_idx'], e['cap'], p_e, r = calc_ds(e,PFA[e['floor']+1],IDR[e['floor']+1])
    t.toc('Damage calcs time is')
    
    #%% Determine flow
    src = 'nr_0'
    count = 0
    t.tic()
    for v in p.vs:
        v['flow'] = p.maxflow_value(0,v.index,'cap')
    t.toc('Flow calcs time is')
    n,bins,patches = plt.hist(x=p.vs['flow'],bins='auto')
        
    #%% Plot results
    visual_style = {}
    visual_style['vertex_size'] = [10 if v['x'] == 0. and v['y'] == 0 else 5 for v in p.vs]
    visual_style['vertex_label'] = ['F{:d}'.format(v['floor']+1) if v['x'] == 0. and v['y'] == 0 else '' for v in p.vs]
    visual_style['vertex_label_dist'] = 2
    visual_style['vertex_label_angle'] = np.pi/2
    visual_style['layout'] = p.layout_reingold_tilford(root=[0])
    visual_style['bbox'] = (900,600)
    visual_style['margin'] = 20
    palette = sns.color_palette(palette='Set2',n_colors=F)
    visual_style['vertex_color'] = ['red' if v['flow']<0.1 else palette[v['floor']] for v in p.vs]
    # visual_style['vertex_color'] = [palette[v['floor']] for v in p.vs]
    visual_style['vertex_shape'] = ['triangle-down' if v['flow']<0.1 else 'circle' for v in p.vs]
    
    if plot:
        t.tic()
        ig.plot(p,**visual_style).show()
        t.toc('Plot time is')
    
    return p
    
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
    
    
    
                