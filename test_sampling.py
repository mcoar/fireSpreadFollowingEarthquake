# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:11:45 2021

@author: Lab User
"""


# import pickle
# import wntr

# #%% Take the network out of the pickle jar
# f=open('wn.pickle','rb')
# wn = pickle.load(f)
# f.close()

# case = 1
# count = 0
# for ename in wn.pipe_name_list:
#     e = wn.get_link(ename)
#     e.ds, e.cap_idx, e.cap, p_e, r = calc_ds(e,PFA[e.floor],IDR[e.floor])
# t.toc('Damage calcs time is')

from sprinkler_hydraulics import *
import intensity as im



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

storyheight = np.array([18.,13.,13.,13.,13.,13.,13.,13.,13.])*ft

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

toggle_repickle = True

Sprinkler(PFA,IDR,plot=True)