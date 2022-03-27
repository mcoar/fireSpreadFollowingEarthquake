# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:49:09 2020

@author: Lab User
"""

from calc_ds import *
from compartment import Compartment
import igraph as ig
import intensity as im
from loadall import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pytictoc import TicToc; tt = TicToc()
import random
import sys
from scipy import stats
import sprinkler_hydraulics as spr
from wall import Wall
from window import Window, Door
# import wntr as wn

#%% Toggles
toggleDoor = True
toggleOpenNeighborDoor = False
toggleDamage = True
togglePlotSprinkler = False
togglePlot_BC_vs_time = False

fire_resistance = "unprotected"
occupancy = 'office'

# Converters
mH202psi = 1/1.42233 # meters of water to psi

# Constants and metrics
min_pressure = 7           # psi (default is 7)


def fireSpread(pathin, pathout,wnin='',ns='',ignloc=''):
    print('ns:'+str(ns))
    storyheight = np.array([18.,13.,13.,13.,13.,13.,13.,13.,13.])*12.
    
    # Get intensity measures
    PFA, IDR = im.OpenSees_IMs(pathin,storyheight,ngrav=10)
    # print(PFA)
    # print(IDR)
    
    # Hydraulic system damage and analysis
    wnfilein = "wn200ft.pickle"
    username = "Lab User"
    # PIKIN = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/"+wnfilein
    wnfileout = 'wn-{:s}.pickle'.format(pathout)
    PIKOUT = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/WNpickle1/"+wnfileout
    wn, wnresults = spr.Sprinkler(PFA,IDR,wnin=wnin,wnpathin='',wnpathout=PIKOUT,ns=ns,plot=togglePlotSprinkler)
    pressure = wnresults.node['pressure']
    
    #%% Define Compartments
    compartments_dict = {}
    num_floors = 9
    num_bays = 4
    wind_speed = 5                  # [m/s]
    wind_dir = 0                   # wind comes from this direction (deg CW from North)
    
    tt.tic()
    for b in range(num_bays):
        for f in range(num_floors):
            bay = b + 1; floor = f + 1;
            # Generate list of names of sprinklers
            spr_list = []
            for m in range(3):
                for n in range(3,6):
                    spr_list.append('ns_{}_{:02d}_0_{}'.format(floor,3*b+m,n))
            compartment_name = ("C"+str(floor)+str(bay))
            key = compartment_name
            compartments_dict.update( {compartment_name:Compartment(compartment_name,floor,bay,occupancy,spr_list=spr_list,pressure=pressure.loc[:,spr_list],min_pressure=min_pressure)} )
            # Add external wall
            wall_name = "eW" + str(floor) +'-'+ str(bay) + "_1"
            compartments_dict[key].add_wall(Wall(wall_name,floor,bay,'Exterior',fire_resistance),wall_name)
            # Add windows
            window_name_1 = wall_name + '_w1'
            window_name_2 = wall_name + '_w2'
            compartments_dict[key].walls[wall_name].add_window(Window(window_name_1, floor, bay, 9.*12,30.*12))
            # compartments_dict[key].walls[wall_name].add_window(Window(window_name_2, floor, bay, 1000.,100.))
            # Add another external wall if on the end
            if b == 0 or b == num_bays-1:
                wall_name = "eW" + str(floor) +"-"+ str(bay) + "_2"
                compartments_dict[key].add_wall(Wall(wall_name,floor,bay,'Exterior',fire_resistance,height=156.,length=360., orientation=0., material='concrete',thickness=254),wall_name)
                window_name_1 = wall_name + '_w1'
                window_name_2 = wall_name + '_w2'
                compartments_dict[key].walls[wall_name].add_window(Window(window_name_1, floor, bay, 9.*12, 30.*12))
                # compartments_dict[key].walls[wall_name].add_window(Window(window_name_2, floor, bay, 1000., 100.))
    print('--Building compartments, external walls and windows complete')      
    # Link Compartments
    for target_compartment in compartments_dict.values():
        for neighbor_compartment in compartments_dict.values():
            nei_bay = neighbor_compartment.get_bay(); nei_flr = neighbor_compartment.get_floor()
            tar_bay = target_compartment.get_bay(); tar_flr = target_compartment.get_floor()
            if nei_bay == tar_bay - 1 and nei_flr == tar_flr:
                target_compartment.add_neighbor(neighbor_compartment,'left')
                ## Add internal walls
                wall_name = 'iW-f'+str(tar_flr)+ '-b' + str(nei_bay)+str(tar_bay) + '_1'
                internal_wall = Wall(wall_name,str(tar_flr),str(nei_bay)+'-'+str(tar_bay),'Interior',fire_resistance)
                ## add door to wall
                if toggleDoor:
                    door_name = wall_name + '_d1'
                    internal_wall.add_door(Door(door_name))
                ## calculate damage state
                if toggleDamage:
                    internal_wall.calc_damage_state(IDR[tar_flr])
                ## put wall in compartments
                target_compartment.add_wall(internal_wall,wall_name)
                neighbor_compartment.add_wall(internal_wall,wall_name)
            elif nei_bay == tar_bay + 1 and nei_flr == tar_flr:
                target_compartment.add_neighbor(neighbor_compartment,'right')
            elif nei_bay == tar_bay and nei_flr == tar_flr - 1:
                target_compartment.add_neighbor(neighbor_compartment,'down')        
            elif nei_bay == tar_bay and nei_flr == tar_flr + 1:
                target_compartment.add_neighbor(neighbor_compartment,'up')  
                # Add ceilings
                wall_name = 'Clg-f'+str(tar_flr)+str(nei_flr)+'-b'+str(tar_bay)
                ceiling = Wall(wall_name,str(tar_flr)+'-'+str(nei_flr),str(tar_bay),'Ceiling',fire_resistance,360.,360.)
                target_compartment.add_wall(ceiling,wall_name)
                neighbor_compartment.add_wall(ceiling,wall_name)
    print('--Compartments linked and internal walls and doors created')
    tt.toc('Structure creation time is:')
    
    #%% Start a Fire
    burning_compartments_dict = {}
    
    if ignloc=='':
        key = random.choice(list(compartments_dict))
    else:
        if ignloc in list(compartments_dict):
            key = ignloc
        else:
            print('UserWarning: ignloc not in compartments list, random ignloc chosen')
            key = random.choice(list(compartments_dict))
            
    if toggleOpenNeighborDoor:
        compartments_dict[key].doors_open.update({'left':True,'right':True})
    ign_loc = np.random.randint(len(compartments_dict[key].sprinklers))
    compartments_dict[key].ignite(60.,0,1,ign_loc=ign_loc)
    burning_compartments_dict.update({key:compartments_dict[key]})
    print('--Fire ignited and burning compartments initialized')
    #%% Find current temperature
    current_temp = compartments_dict[key].get_current_temp(91)
    
    #%% Loop through time to determine fire spread
    print('--Fire spread loop started')
    tt.tic()
    t_start = 0; t_end = 43200; t_step = 60
    tVec = range(int(t_start/60),int(t_end/60),int(t_step/60))
    burnVec = [0]*len(tVec)
    fullVec = [0]*len(tVec)
    decayVec = [0.]*len(tVec)
    # Loop through time (t)
    t = t_start
    count = 0
    while t < t_end:
        count += 1
        new_ignitions = {}
        bt_walls = []
        # Loop through burning compartments (bc)
        bc_new = {}
        for name, bc in burning_compartments_dict.items(): 
            # Check sprinkler status
            [W,Dbar,Pbar] = bc.get_sprinkler_info()
            mH20_to_psi = 1.42233
            P_psi = mH20_to_psi*bc.pressure.iloc[0,:].to_numpy()      
            # Update compartment fire conditions
            bc.update_fire_phase(max(0,t-bc.get_time_of_ignition()))
            # Check if burning
            burnVec[count-1] += 1
            # Check if decayed
            if bc.get_is_decayed():
                decayVec[count-1] += 1
            # Check if fully developed
            if bc.get_is_fully_developed():
                fullVec[count-1] += 1
                # Check for burnthrough of walls (inc. passing through open doors)
                for wall_name, wall in bc.walls.items():
                    t_bt = wall.get_burnthrough()[0]*3600.
                    t_fd = bc.get_time_of_full_development()        
                    # Logic: exterior walls don't matter for burnthrough
                    # Floors don't burnthrough (compare names to test this)
                    # wall can't burn twice
                    # See Li Davidson 2010 sec. 3.4.
                    # NOTE: this might let burnthrough happen to a lower floor. That could be a problem.
                    if not (wall.get_type().lower() == 'exterior' )             and not \
                        (wall_name[0:6] == 'Clg-f'+str(bc.get_floor()-1))       and not \
                        ( wall.get_is_burned() )                                and \
                        ( ( any([door.get_is_open() for door in wall.doors]) )  or \
                        ( wall.get_ds() == 2 and (t-t_fd) > t_bt/2.)            or \
                        ( wall.get_ds() == 3)                                   or \
                        ( wall.get_num_of_doors() > 0 and ( t-t_fd ) > t_bt/2. ) or \
                        ( (t-t_fd) > t_bt ) ):
                        bt_walls.append(wall_name)
                        wall.set_is_burned()
                        print('Wall {0} just burnt through at time {1:0.2f} h.'.format(wall_name,t/3600.))
                        print('t_bt: {0:0.2f} min, t_fd: {1:0.2f} hr'.format(t_bt/60.,t_fd/3600.))
                    for win in wall.windows:
                        # TO DO
                        # need some trig here to adjust windspeed based on difference
                        # in orientation between wind and wall orientation.
                        h_w, x_w, w_w, lamb_w = \
                            win.flame_geometry(bc.get_has_draft(), bc.get_floor() == num_floors, \
                                               bc.get_mdot_r(), wind_speed, summary=False)
                        try:
                            tc = bc.neighbors['up']     # target compartment
                            # get distance from top of window to bottom of window next floor
                            min_sill_height_m = min(min([ [win.get_sill_height_m() for win in wall.windows] for wall in tc.walls.values() \
                                                       if len(wall.windows)>0]))
                            z = bc.get_dim_m()[0]-win.get_height_m()-win.get_sill_height_m() + \
                                min_sill_height_m
                            # get "time to leapfrog" = time needed to ignite upper comp, dependent of x_w
                            if x_w < 1:
                                t_lf = 3.*60
                            else: 
                                t_lf = min(12., 3. + x_w)*60
                            # check if window flame is tall enough to ignite next floor
                            if h_w >= z and (t-t_fd) > t_lf and not tc.get_is_burning():
                                bc_new.update({tc.name:tc})
                        except KeyError:
                            continue
                                                
        # Advance timestep and ignite rooms that share walls that just burnt through.
        t += t_step
        for name, c in bc_new.items():
            t_ignition = t
            time_curve = 0
            temp_curve = random.choice([1,2,3])
            c.ignite(t_ignition,time_curve,temp_curve)
            burning_compartments_dict.update({name:c})
        for name, c in compartments_dict.items():
            if bool( set(c.walls) & set(bt_walls) ) and not c.get_is_burning():
                t_ignition = t
                time_curve = 0
                temp_curve = random.choice([1,2,3])
                c.ignite(t_ignition,time_curve,temp_curve)
                burning_compartments_dict.update({name:c})
                print('Comp. {0} ignited at time {1:0.1f} h.'.format(name,c.get_time_of_ignition()/3600.))
    tt.toc('Fire Spread Sim time is') 
    
    
    if togglePlot_BC_vs_time:
        plt.plot(tVec,burnVec,color='black',label='Burned')
        plt.plot(tVec,fullVec,color='red',label='Full Dev.')
        plt.plot(tVec,decayVec,color='blue',label='Decayed')
        plt.xlabel('Time (min)')
        plt.ylabel('# of compartments')
        plt.xticks(range(0,500,60))
        plt.yticks(range(0,35,5))
        plt.ylim(0,37)
        plt.grid()
        # plt.title('EQ damage, have doors')
        plt.legend()
        
    return burning_compartments_dict,wn
        