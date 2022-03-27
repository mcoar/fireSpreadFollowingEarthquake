# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 07:18:17 2021

@author: Lab User
"""

import numpy as np



def OpenSees_IMs(pathname,storyheight,ngrav=10):
    """ Calculate story intensity measures from opensees outputs
    
    Arguments:
        ___INPUT___
    pathname [s]: file location of OpenSees output files
        Acceleration files should be "AccF{d}.out"
        Displacement files should be "DispF{d}.out"
        All files should be two columns, time (s) and acceleration (in/s2)
            or displacement (in)
        All files should be of the same length and the time columns should be 
            identical
    storyheight[np array]: story heights in inches, equal to the number of 
        floors
    ngrav [int]: number of gravity steps before seismic input
        ___OUTPUT___
    PFA: peak floor acceleration in [g].
    IDR: interstory drift ratio in [%].
    """
    # print(pathname)
    # UNITS
    inch = 1.       # inches
    sec  = 1.       # sec
    g  = .00259*inch/sec**2     # gravitational acceleration
    # get length of datafile
    count = len(open(pathname+'/DispF2.out').readlines(  ))
    storyheight = np.append([1e-10],[storyheight])
    nstory = len(storyheight)
    # initialize np array
    u = np.zeros(shape=[count-ngrav-1,nstory])
    acc = np.zeros(shape=[count-ngrav-1,nstory])
    print(acc.shape)
    d = np.zeros(shape=[nstory])
    # loop through each floor
    for i in range(1,10):
        # pull data from each floor displacement file
        u[:,i] = np.genfromtxt(pathname+'/DispF{}.out'.format(i+1), skip_header=ngrav, skip_footer=1, usecols=1)
        # find max difference between this floor and previous floor. 
        d[i] = np.max(np.abs(u[:,i]-u[:,i-1]))
        # pull data from each floor acceleration file
        acc[:,i] = np.genfromtxt(pathname+'/AccF{}.out'.format(i+1), skip_header=ngrav, skip_footer=1, usecols=1)
    # find peak floor acceleration
    PFA = np.max(np.abs(acc),axis=0)*g        # g
    # calculate IDR
    IDR = np.divide(d,storyheight)*100.
    
    return PFA, IDR