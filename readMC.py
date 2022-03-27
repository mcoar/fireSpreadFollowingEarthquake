# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:19:30 2022

@author: Lab User
"""

#%% Import libraries

import beepy
from calc_ds import *
from compartment import Compartment
import fireSpreadMain as fs
from GMprocessing import *
import igraph as ig
import intensity as im
from loadall import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pytictoc import TicToc; t = TicToc()
import random
import seaborn as sns
from scipy import stats, optimize
import sprinkler_hydraulics as spr
import sys
from wall import Wall
from window import Window, Door
import wntr

#%% Toggles and Options
toggle_read = False
toggle_plot = False
toggle_plot_old = True
toggle_plot_Sa = False
pickleOut = 'outputDict1'
pickleIn = 'outputDict1'



#%% Some set up

nSamples = 100

# earthquake records path
username = "Lab User"
pathER = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/9st_mrf_mwc/GMs_Full_Suite/"

# fire spread samples path
pathBC = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/BCpickle1/"

# water network samples path
pathWN = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/WNpickle1/"


# main path
pathMAIN = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/"



# Grab all directory names in the burning compartment path
dn = []
fn = []
for (dirpath, dirnames, filenames) in os.walk(pathBC):
    dn.extend(dirnames)
    fn.extend(filenames)
    break

# Split the directories into lagrande and seattle
def splitList(x,a=0,b=8):
    recdict = dict()
    for recname in x:
        rec = recname[a:b]
        if rec in recdict.keys():
            recdict[rec].append(recname)
        else:
            recdict[rec] = [recname]
    return recdict

fd = splitList(fn)
shapeErrorList = []

# Define a function to grab PGA from file
def getPGA(PATHIN):
    with open(PATHIN,'r') as f:
        data = [float(val) for val in f]
        PGA = max(data, key=abs)
    return PGA

LOCd = {}

#%% Main Reader

if toggle_read:
    # Loop through locations
    for fk, fi in fd.items():
        print('==============================================')
        print('|       Record Set: {}                 |'.format(fk))
        print('==============================================')
        print()
        rd = splitList(fi,0,-13)
        RECd = {}
        # Loop through earthquake records at location
        for rk, ri in rd.items():
            print('  ==============================================')
            print('  |       Record: {}             |'.format(rk))
            print('  ==============================================')
            print()
            recname = rk[3::]+'.txt'
            PGA = abs(getPGA(pathER+recname))
            IMcomb,IMdetails = get_IMcomb(rk[3::],T1=1.75,plot=toggle_plot_Sa,toggle_print=False) 
            nSIM = 0
            nBClist = []
            nLEAKlist = []
            # Loop through MC sims
            for r in ri:
                nSIM += 1
                print('    MC sim: {:s}'.format(r[-12:-7]))
                print('    PGA: {:0.3f} [g]'.format(PGA))
                print('    IMcomb: {:0.3f}'.format(IMcomb))
                # Load the pickled fire sim results
                with open(pathBC+r,'rb') as f:
                    bc = pickle.load(f)
                
                    nBC = len(bc)
                    nBClist.append(nBC)
                    print('     No. of burning compartments: {:d}'.format(nBC))
                    # Loop through compartments
                    for ck, ci in bc.items():
                        try:
                            print('      {0:s} began burning at time {1:0.1f} and decayed at time {2:0.1f}'.format(ci.get_name(),ci.time_of_ignition,ci.time_of_ignition+ci.time_of_decay))
                        except TypeError:
                            print('      {0:s} began burning, (warning: TypeError)'.format(ci.get_name()))
                    print()
                # Load the pickled hydraulic sim results
                w = r.replace('BC_','wn-')
                try:
                    with open(pathWN+w,'rb') as f:
                        wn = pickle.load(f)
                        wn_ds = wn[2][0]
                        nleaks = sum(wn_ds[2:4])
                        nLEAKlist.append(nleaks)
                        
                        print('      No. of leaks: {}'.format(nleaks))
                except:
                    print('      No hydraulic data for this sim')
                    nLEAKlist.append(-1.)
                    print()
                    continue
                print()
                
                    
                    
                
        # lndist = stats.lognorm.fit(nBClist,loc=0)
        # print('      '+str(lndist))
            print()
            RECd[rk] = {'Loc':fk,'nSIM':nSIM,'PGA_g':PGA,'IMcomb':IMcomb,'nBC':nBClist,'nLEAK':nLEAKlist}
            
        LOCd[fk] = RECd
        f=open(pathMAIN+pickleOut+".pickle",'wb')
        pickle.dump(LOCd,f)
        f.close()   
    

#%% FORMATTING RESULTS
# Grab the pickled output file if not generated
if not toggle_read:
    with open(pathMAIN+pickleIn+".pickle",'rb') as f:
        LOCd = pickle.load(f)
frames = []
for lk, li in LOCd.items():
    frames.append(pd.DataFrame(li).transpose())
df = pd.concat(frames)
dfex = df.set_index(['Loc','nSIM','PGA_g','IMcomb']).apply(pd.Series.explode).reset_index()
# df = df.explode(["nBC","nLEAK"])
pd.set_option('display.max_columns',None)

ds0 = 1
ds1 = 4
ds2 = 8
ds3 = 100

ds = [ds0,ds1,ds2,ds3]

df.sort_values(by='PGA_g',inplace=True)

# df['ds0'] = df.apply(lambda row: sum((x == ds0) for x in row.nBC)/row.nSIM,axis=1)
df['ds1'] = df.apply(lambda row: sum((x > ds0) for x in row.nBC)/row.nSIM,axis=1)
df['ds2'] = df.apply(lambda row: sum((x > ds1) for x in row.nBC)/row.nSIM,axis=1)
df['ds3'] = df.apply(lambda row: sum((x > ds2) for x in row.nBC)/row.nSIM,axis=1)

#%% Fit Data
# Set up lognormal function - note, scipy has a weird formulation for this. Look it up.
f_ln = lambda x,mu,sigma: stats.lognorm(s=sigma, scale=np.exp(mu)).cdf(x)
f_norm = lambda x,mu,sigma: stats.norm(loc=sigma,scale=mu).cdf(x)
f_gamma = lambda x,a,b: stats.gamma(a=a,scale=1/b).cdf(x)
f_weibull_min = lambda x,c,s: stats.weibull_min(c=c,scale=s).cdf(x)
f_weibull_max = lambda x,c,s: stats.weibull_max(c=c,scale=s).cdf(x)
f_log = lambda x,th1,th2: np.exp(th1+th2*x)/(1+np.exp(th1+th2*x))

f = {'Norm':f_norm,
     'Lognorm':f_ln,
     # 'Gamma':f_gamma,
     # 'Weibull_min':f_weibull_min,
     # 'Weibull_max':f_weibull_max,
     'Logistic': f_log
     }


# PGA
for k, func in f.items():
    # LaGrande
    LGx1 = df['PGA_g'].loc[df['Loc']=='BC_LaGra'].to_list()
    LGy1 = df['ds1'].loc[df['Loc']=='BC_LaGra'].to_list()
    mu1_lg, sigma1_lg = optimize.curve_fit(func,LGx1,LGy1)[0]
    # _,_,r_lg1,_,_ = stats.linregress(LGy1,func(LGx1,mu1_lg,sigma1_lg))
    _,_,r_lg1,_,_ = stats.linregress(LGy1,[func(x,mu1_lg,sigma1_lg) for x in LGx1])
    r2_lg1 = r_lg1**2
    
    LGx2 = df['PGA_g'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
    LGy2 = df['ds2'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
    mu2_lg, sigma2_lg = optimize.curve_fit(func,LGx2,LGy2)[0]
    _,_,r_lg2,_,_ = stats.linregress(LGy2,[func(x,mu2_lg,sigma2_lg) for x in LGx2])
    r2_lg2 = r_lg2**2
    
    LGx3 = df['PGA_g'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
    LGy3 = df['ds3'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
    mu3_lg, sigma3_lg = optimize.curve_fit(func,LGx3,LGy3)[0]
    _,_,r_lg3,_,_ = stats.linregress(LGy3,[func(x,mu3_lg,sigma3_lg) for x in LGx3])
    r2_lg3 = r_lg3**2
    
    # Seattle
    SEx1 = df['PGA_g'].loc[df['Loc']=='BC_Seatt'].to_list()#[1:-1]
    SEy1 = df['ds1'].loc[df['Loc']=='BC_Seatt'].to_list()#[1:-1]
    mu1_se, sigma1_se = optimize.curve_fit(func,SEx1,SEy1)[0]
    # _,_,r_se1,_,_ = stats.linregress(SEy1,func(SEx1,mu1_se,sigma1_se))
    _,_,r_se1,_,_ = stats.linregress(SEy1,[func(x,mu1_se,sigma1_se) for x in SEx1])
    r2_se1 = r_se1**2
    
    SEx2 = df['PGA_g'].loc[df['Loc']=='BC_Seatt'].to_list()#[1:-1]
    SEy2 = df['ds2'].loc[df['Loc']=='BC_Seatt'].to_list()#[1:-1]
    mu2_se, sigma2_se = optimize.curve_fit(func,SEx2,SEy2)[0]
    _,_,r_se2,_,_ = stats.linregress(SEy2,[func(x,mu2_se,sigma2_se) for x in SEx2])
    r2_se2 = r_se2**2
    
    SEx3 = df['PGA_g'].loc[df['Loc']=='BC_Seatt'].to_list()#[1:-1]
    SEy3 = df['ds3'].loc[df['Loc']=='BC_Seatt'].to_list()#[1:-1]
    mu3_se, sigma3_se = optimize.curve_fit(func,SEx3,SEy3)[0]
    _,_,r_se3,_,_ = stats.linregress(SEy3,[func(x,mu3_se,sigma3_se) for x in SEx3])
    r2_se3 = r_se3**2
    
    # BOTH
    BOx1 = df['PGA_g'].to_list()[:] #[1:-1]
    BOy1 = df['ds1'].to_list()[:] #[1:-1]
    mu1_bo, sigma1_bo = optimize.curve_fit(func,BOx1,BOy1)[0]
    _,_,r_bo1,_,_ = stats.linregress(BOy1,[func(x,mu1_bo,sigma1_bo) for x in BOx1])
    r2_bo1 = r_bo1**2
    
    BOx2 = df['PGA_g'].to_list()[:] #[1:-1]
    BOy2 = df['ds2'].to_list()[:] #[1:-1]
    mu2_bo, sigma2_bo = optimize.curve_fit(func,BOx2,BOy2)[0]
    _,_,r_bo2,_,_ = stats.linregress(BOy2,[func(x,mu2_bo,sigma2_bo) for x in BOx2])
    r2_bo2 = r_bo2**2
    
    BOx3 = df['PGA_g'].to_list()[:] #[1:-1]
    BOy3 = df['ds3'].to_list()[:] #[1:-1]
    mu3_bo, sigma3_bo = optimize.curve_fit(func,BOx3,BOy3)[0]
    _,_,r_bo3,_,_ = stats.linregress(BOy3,[func(x,mu3_bo,sigma3_bo) for x in BOx3])
    r2_bo3 = r_bo3**2


    print()
    print('========================')
    print('Curve type: {:s}, IM = PGA (g)'.format(k))
    print('  LaGrande')
    print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1_lg,sigma1_lg,r2_lg1))
    print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2_lg,sigma2_lg,r2_lg2))
    print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3_lg,sigma3_lg,r2_lg3))
    print('  Seattle')
    print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1_se,sigma1_se,r2_se1))
    print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2_se,sigma2_se,r2_se2))
    print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3_se,sigma3_se,r2_se3))
    print('  Both')
    print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1_bo,sigma1_bo,r2_bo1))
    print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2_bo,sigma2_bo,r2_bo2))
    print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3_bo,sigma3_bo,r2_bo3))
 
    #%% PLOTTING
    if toggle_plot:
        x = np.linspace(0,0.5,100)
        
        plt.figure(11)
        ax11 = df.loc[df['Loc']=='BC_LaGra'].plot(x="PGA_g",y="ds1",kind="scatter", color='r', marker='d', s=20, label="ds1")
        ax11 = df.loc[df['Loc']=='BC_LaGra'].plot(x="PGA_g",y="ds2",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax11)
        ax11 = df.loc[df['Loc']=='BC_LaGra'].plot(x="PGA_g",y="ds3",kind="scatter", color='b', marker='o', s=10, label="ds3",ax=ax11)
        ax11.plot(x,func(x,mu1_lg,sigma1_lg),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_lg1))
        ax11.plot(x,func(x,mu2_lg,sigma2_lg),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_lg2))
        ax11.plot(x,func(x,mu3_lg,sigma3_lg),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_lg2))
        ax11.legend(loc="upper left")
        ax11.grid(visible=True)
        ax11.set_xlim(0.0,0.4)
        ax11.set_ylim(-0.05,1.1)
        ax11.set_xlabel("PGA (g)")
        ax11.set_ylabel("Likelihood")
        title = '{:s} CDF, LaGrande'.format(k)
        ax11.set_title(title)
        plt.show()
        
        plt.figure(12)
        ax12 = df.loc[df['Loc']=='BC_Seatt'].plot(x="PGA_g",y="ds1",kind="scatter", color='r', marker='d', s=20, label="ds1")
        ax12 = df.loc[df['Loc']=='BC_Seatt'].plot(x="PGA_g",y="ds2",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax12)
        ax12 = df.loc[df['Loc']=='BC_Seatt'].plot(x="PGA_g",y="ds3",kind="scatter", color='b', marker='o', s=10,  label="ds3",ax=ax12)
        ax12.plot(x,func(x,mu1_se,sigma1_se),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_se1))
        ax12.plot(x,func(x,mu2_se,sigma2_se),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_se2))
        ax12.plot(x,func(x,mu3_se,sigma3_se),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_se3))
        ax12.legend(loc="upper left")
        ax12.grid(visible=True)
        ax12.set_xlim(0.0,0.4)
        ax12.set_ylim(-0.05,1.1)
        ax12.set_xlabel("PGA (g)")
        ax12.set_ylabel("Likelihood")
        title = '{:s} CDF, Seattle'.format(k)
        ax12.set_title(title)
        plt.show()
        
        plt.figure(13)
        ax13 = df.plot(x="PGA_g",y="ds1",kind="scatter", color='r', marker='d', s=20, label="ds1")
        ax13 = df.plot(x="PGA_g",y="ds2",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax13)
        ax13 = df.plot(x="PGA_g",y="ds3",kind="scatter", color='b', marker='o', s=10,  label="ds3",ax=ax13)
        ax13.plot(x,func(x,mu1_bo,sigma1_bo),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_bo1))
        ax13.plot(x,func(x,mu2_bo,sigma2_bo),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_bo2))
        ax13.plot(x,func(x,mu3_bo,sigma3_bo),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_bo3))
        ax13.legend(loc="upper left")
        ax13.grid(visible=True)
        ax13.set_xlim(0.0,0.4)
        ax13.set_ylim(-0.05,1.1)
        ax13.set_xlabel("PGA (g)")
        ax13.set_ylabel("Likelihood")
        title = '{:s} CDF, Both Locations'.format(k)
        ax13.set_title(title)
        plt.show()
        
    
# IMcomb
for k, func in f.items():
    try:
        # LaGrande
        LGx1 = df['IMcomb'].loc[df['Loc']=='BC_LaGra'].to_list()[:]
        LGy1 = df['ds1'].loc[df['Loc']=='BC_LaGra'].to_list()[:]
        mu1_lg, sigma1_lg = optimize.curve_fit(func,LGx1,LGy1)[0]
        _,_,r_lg1,_,_ = stats.linregress(LGy1,[func(x,mu1_lg,sigma1_lg) for x in LGx1])
        r2_lg1 = r_lg1**2
        
        LGx2 = df['IMcomb'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
        LGy2 = df['ds2'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
        mu2_lg, sigma2_lg = optimize.curve_fit(func,LGx2,LGy2)[0]
        _,_,r_lg2,_,_ = stats.linregress(LGy2,[func(x,mu2_lg,sigma2_lg) for x in LGx2])
        r2_lg2 = r_lg2**2
        
        LGx3 = df['IMcomb'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
        LGy3 = df['ds3'].loc[df['Loc']=='BC_LaGra'].to_list()#[1:-1]
        mu3_lg, sigma3_lg = optimize.curve_fit(func,LGx3,LGy3)[0]
        _,_,r_lg3,_,_ = stats.linregress(LGy3,[func(x,mu3_lg,sigma3_lg) for x in LGx3])
        r2_lg3 = r_lg3**2
        
        # Seattle
        SEx1 = df['IMcomb'].loc[df['Loc']=='BC_Seatt'].to_list()[1:-1]
        SEy1 = df['ds1'].loc[df['Loc']=='BC_Seatt'].to_list()[1:-1]
        mu1_se, sigma1_se = optimize.curve_fit(func,SEx1,SEy1)[0]
        _,_,r_se1,_,_ = stats.linregress(SEy1,[func(x,mu1_se,sigma1_se) for x in SEx1])
        r2_se1 = r_se1**2
        
        SEx2 = df['IMcomb'].loc[df['Loc']=='BC_Seatt'].to_list()[:] #[1:-1]
        SEy2 = df['ds2'].loc[df['Loc']=='BC_Seatt'].to_list()[:] #[1:-1]
        mu2_se, sigma2_se = optimize.curve_fit(func,SEx2,SEy2)[0]
        _,_,r_se2,_,_ = stats.linregress(SEy2,[func(x,mu2_se,sigma2_se) for x in SEx2])
        r2_se2 = r_se2**2
        
        SEx3 = df['IMcomb'].loc[df['Loc']=='BC_Seatt'].to_list()[:] #[1:-1]
        SEy3 = df['ds3'].loc[df['Loc']=='BC_Seatt'].to_list()[:] #[1:-1]
        mu3_se, sigma3_se = optimize.curve_fit(func,SEx3,SEy3)[0]
        _,_,r_se3,_,_ = stats.linregress(SEy3,[func(x,mu3_se,sigma3_se) for x in SEx3])
        r2_se3 = r_se3**2
        
        # BOTH
        BOx1 = df['IMcomb'].to_list()[:] #[1:-1]
        BOy1 = df['ds1'].to_list()[:] #[1:-1]
        mu1_bo, sigma1_bo = optimize.curve_fit(func,BOx1,BOy1)[0]
        _,_,r_bo1,_,_ = stats.linregress(BOy1,[func(x,mu1_bo,sigma1_bo) for x in BOx1])
        r2_bo1 = r_bo1**2
        
        BOx2 = df['IMcomb'].to_list()[:] #[1:-1]
        BOy2 = df['ds2'].to_list()[:] #[1:-1]
        mu2_bo, sigma2_bo = optimize.curve_fit(func,BOx2,BOy2)[0]
        _,_,r_bo2,_,_ = stats.linregress(BOy2,[func(x,mu2_bo,sigma2_bo) for x in BOx2])
        r2_bo2 = r_bo2**2
        
        BOx3 = df['IMcomb'].to_list()[:] #[1:-1]
        BOy3 = df['ds3'].to_list()[:] #[1:-1]
        mu3_bo, sigma3_bo = optimize.curve_fit(func,BOx3,BOy3)[0]
        _,_,r_bo3,_,_ = stats.linregress(BOy3,[func(x,mu3_bo,sigma3_bo) for x in BOx3])
        r2_bo3 = r_bo3**2
    
        print()
        print('========================')
        print('Curve type: {:s}, IM: IMcomb'.format(k))
        print('  LaGrande')
        print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1_lg,sigma1_lg,r2_lg1))
        print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2_lg,sigma2_lg,r2_lg2))
        print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3_lg,sigma3_lg,r2_lg3))
        print('  Seattle')
        print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1_se,sigma1_se,r2_se1))
        print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2_se,sigma2_se,r2_se2))
        print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3_se,sigma3_se,r2_se3))
        print('  Seattle')
        print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1_bo,sigma1_bo,r2_bo1))
        print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2_bo,sigma2_bo,r2_bo2))
        print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3_bo,sigma3_bo,r2_bo3))
        
    except RuntimeError:
        print('Curve {:s}, parameters not found'.format(k))
                    
    
    #%% PLOTTING
    if toggle_plot:
        x = np.linspace(0,0.5,100)
        
        plt.figure(21)
        ax21 = df.loc[df['Loc']=='BC_LaGra'].plot(x="IMcomb",y="ds1",kind="scatter", color='r', marker='d', s=20, label="ds1")
        ax21 = df.loc[df['Loc']=='BC_LaGra'].plot(x="IMcomb",y="ds2",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax21)
        ax21 = df.loc[df['Loc']=='BC_LaGra'].plot(x="IMcomb",y="ds3",kind="scatter", color='b', marker='o', s=10, label="ds3",ax=ax21)
        ax21.plot(x,func(x,mu1_lg,sigma1_lg),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_lg1))
        ax21.plot(x,func(x,mu2_lg,sigma2_lg),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_lg2))
        ax21.plot(x,func(x,mu3_lg,sigma3_lg),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_lg2))
        ax21.legend(loc="upper left")
        ax21.grid(visible=True)
        ax21.set_xlim(0.0,0.025)
        ax21.set_ylim(-0.05,1.1)
        ax21.set_xlabel("IMcomb")
        ax21.set_ylabel("Likelihood")
        title = '{:s} CDF, LaGrande'.format(k)
        ax21.set_title(title)
        plt.show()
        
        plt.figure(22)
        ax22 = df.loc[df['Loc']=='BC_Seatt'].plot(x="IMcomb",y="ds1",kind="scatter", color='r', marker='d', s=20, label="ds1")
        ax22 = df.loc[df['Loc']=='BC_Seatt'].plot(x="IMcomb",y="ds2",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax22)
        ax22 = df.loc[df['Loc']=='BC_Seatt'].plot(x="IMcomb",y="ds3",kind="scatter", color='b', marker='o', s=10, label="ds3",ax=ax22)
        ax22.plot(x,func(x,mu1_se,sigma1_se),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_se1))
        ax22.plot(x,func(x,mu2_se,sigma2_se),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_se2))
        ax22.plot(x,func(x,mu3_se,sigma3_se),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_se3))
        ax22.legend(loc="upper left")
        ax22.grid(visible=True)
        ax22.set_xlim(0.0,0.025)
        ax22.set_ylim(-0.05,1.1)
        ax22.set_xlabel("IMcomb")
        ax22.set_ylabel("Likelihood")
        title = '{:s} CDF, Seattle'.format(k)
        ax22.set_title(title)
        plt.show()
        
        plt.figure(23)
        ax23 = df.plot(x="IMcomb",y="ds1",kind="scatter", color='r', marker='d', s=20, label="ds1")
        ax23 = df.plot(x="IMcomb",y="ds2",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax23)
        ax23 = df.plot(x="IMcomb",y="ds3",kind="scatter", color='b', marker='o', s=10, label="ds3",ax=ax23)
        ax23.plot(x,func(x,mu1_bo,sigma1_bo),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_bo1))
        ax23.plot(x,func(x,mu2_bo,sigma2_bo),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_bo2))
        ax23.plot(x,func(x,mu3_bo,sigma3_bo),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_bo3))
        ax23.legend(loc="upper left")
        ax23.grid(visible=True)
        ax23.set_xlim(0.0,0.025)
        ax23.set_ylim(-0.05,1.1)
        ax23.set_xlabel("IMcomb")
        ax23.set_ylabel("Likelihood")
        title = '{:s} CDF, Both Locations'.format(k)
        ax23.set_title(title)
        plt.show()        
        
        
        # print(df)    

if toggle_plot_old:
    # sns.scatterplot(data=df, x="PGA_g", y="nLEAK")
    new_title = 'Location'
    new_labels = ['LaGrande','Seattle']
    dfex['nLEAKnorm'] = dfex['nLEAK']/1414.
    dfex['nBCnorm'] = dfex['nBC']/36.
    plt.figure(1)
    plt.grid()
    # ax1 = plt.axis([0,0.5,0,50])
    g= sns.scatterplot(data=dfex, x="PGA_g", y="nBCnorm", hue="Loc", style="Loc",facet_kws={'legend_out':True}).set(title = 'No. of burning compartments vs. PGA', xlabel = 'PGA (g)', ylabel = 'Burning Compartments (Normalized, N=36)')
    plt.legend(loc='upper left')
    # g._legend.set_title(new_title)
    # for t, l in zip(g._legend.texts, new_labels):
    #     t.set_text(l)
    plt.figure(2)
    plt.grid()
    # ax2 = plt.axis([0,0.3,0,50])
    sns.scatterplot(data=dfex, x="IMcomb", y="nBCnorm", hue="Loc", style="Loc").set(title = 'No. of burning compartments vs. $IM_{comb}$', xlabel = '$IM_{comb}$', ylabel = 'Burning Compartments (Normalized, N=36)')
    plt.legend(loc='upper left')
    # plt.figure(2)
    plt.figure(3)
    plt.grid()
    # ax3 = plt.axis([0,1500,0,50])
    sns.scatterplot(data=dfex, x="nLEAK", y="nBC", hue="Loc", style="Loc").set(title = 'No. of burning compartments vs. No. of leaking pipes',xlabel = 'Number of Leaking Pipes', ylabel = 'Number of Burning Compartments')
    plt.legend(loc='upper left')
    # plt.figure(4)
    # ax3 = plt.axis([0,50,0,50])
    # sns.histplot(data=dfex, x="PGA_g",y="nBC",hue="Loc")
    
    # nLEAK vs. IM
    plt.figure(6)
    plt.grid()
    sns.scatterplot(data=dfex,x="PGA_g",y="nLEAKnorm",hue="Loc",style="Loc").set(title = 'No. of leaking pipes vs. PGA', xlabel = 'PGA (g)', ylabel = 'Leaking pipes (Normalized, N=1414)')
    plt.figure(7)
    plt.grid()
    sns.scatterplot(data=dfex,x="IMcomb",y="nLEAKnorm",hue="Loc",style="Loc").set(title = 'No. of leaking pipes vs. $IM_{comb}$', xlabel = '$IM_{comb}$', ylabel = 'Leaking pipes (Normalized, N=1414)')
    
    
    # plt.figure(8)
    # sns.jointplot(data=dfex,x="nLEAK",y="nBC",kind="hex",)
    # df.sort_values(by='IMcomb',inplace=True)
    
    # plt.figure(13)
    # ax13 = df.loc[df['Loc']=='BC_LaGra'].plot(x="IMcomb",y="ds1",kind="scatter", color='r',label="ds1 (>1 BC)")
    # ax13 = df.loc[df['Loc']=='BC_LaGra'].plot(x="IMcomb",y="ds2",kind="scatter", color='g',label="ds2 (>4 BC)",ax=ax13)
    # ax13 = df.loc[df['Loc']=='BC_LaGra'].plot(x="IMcomb",y="ds3",kind="scatter", color='b',label="ds3 (>8 BC)",ax=ax13)
    # ax13.grid(visible=True)
    # ax13.set_xlim(0.0,0.02)
    # ax13.set_ylim(-0.1,1.1)
    # ax13.set_xlabel("IM$_{comb}$")
    # ax13.set_ylabel("Likelihood")
    # ax13.set_title("Empirical CDF, LaGrande (IMcomb)")
    # plt.show()
    
    # plt.figure(14)
    # ax14 = df.loc[df['Loc']=='BC_Seatt'].plot(x="IMcomb",y="ds1",kind="scatter", color='r',label="ds1 (>1 BC)")
    # ax14 = df.loc[df['Loc']=='BC_Seatt'].plot(x="IMcomb",y="ds2",kind="scatter", color='g',label="ds2 (>4 BC)",ax=ax14)
    # ax14 = df.loc[df['Loc']=='BC_Seatt'].plot(x="IMcomb",y="ds3",kind="scatter", color='b',label="ds3 (>8 BC)",ax=ax14)
    # ax14.grid(visible=True)
    # ax14.set_xlim(0.0,0.25)
    # ax14.set_ylim(-0.1,1.1)
    # ax14.set_xlabel("IM$_{comb}$")
    # ax14.set_ylabel("Likelihood")
    # ax14.set_title("Empirical CDF, Seattle (IMcomb)")
    # plt.show()
    
# beepy.beep(sound="coin")
    








    