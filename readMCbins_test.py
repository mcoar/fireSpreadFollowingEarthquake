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

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

#%% Toggles and Options
toggle_read = False
toggle_plot = True
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

dfL = dfex.loc[dfex['Loc'].str.contains('LaGra'),:]
dfS = dfex.loc[dfex['Loc'].str.contains('Seatt'),:]
dfB = dfex

dfList = [dfB,dfL,dfS]

nbins = 7

def dfStats(df, nbins = 7):
    df['PGA groups'], h      = pd.qcut(df['PGA_g'],nbins, retbins=True)
    df['IMcomb groups'], hIM = pd.qcut(df['IMcomb'],nbins, retbins=True)
    
    ds = {
          'ds1':1,
          'ds2':4,
          'ds3':8,
          }
    
    for dskey,dsvalue in ds.items():
        df[dskey] = (df['nBC'] > dsvalue).astype(int)
        
    binwidth = np.diff(h)
    binwidthIM = np.diff(hIM)
    
    dfPGA = df.groupby('PGA groups').sum()
    dfIMcomb = df.groupby('IMcomb groups').sum()
    
    dfPGA['count']   = df['PGA groups'].value_counts()
    dfIMcomb['countIM'] = df['IMcomb groups'].value_counts()
    
    dfPGA['PGA_g'] = h[:-1] + binwidth/2.
    dfIMcomb['IMcomb'] = hIM[:-1] + binwidthIM/2.
    
    for dskey,dsvalue in ds.items():
        dfPGA[dskey+'norm'] = dfPGA[dskey]/dfPGA['count']
        dfIMcomb[dskey+'norm'] = dfIMcomb[dskey]/dfIMcomb['countIM']
    
    return dfPGA, dfIMcomb


df, dfIM = dfStats(dfS)


locName = 'Seattle'

# sys.exit()

#%% Set Up Data Fitting
# Set up lognormal function - note, scipy has a weird formulation for this. Look it up.
f_ln = lambda x,mu,sigma: stats.lognorm(s=sigma, scale=np.exp(mu)).cdf(x)
f_norm = lambda x,mu,sigma: stats.norm(loc=sigma,scale=mu).cdf(x)
f_gamma = lambda x,a,b: stats.gamma(a=a,scale=1/b).cdf(x)
f_weibull_min = lambda x,c,s: stats.weibull_min(c=c,scale=s).cdf(x)
f_weibull_max = lambda x,c,s: stats.weibull_max(c=c,scale=s).cdf(x)
f_log = lambda x,th1,th2: np.exp(th1+th2*x)/(1+np.exp(th1+th2*x))

f = {#'Norm':f_norm,
     'Lognorm':f_ln,
     # 'Gamma':f_gamma,
     # 'Weibull_min':f_weibull_min,
      # 'Weibull_max':f_weibull_max,
     # 'Logistic': f_log
     }

#%%  PGA Fitting
for k, func in f.items():
    try:
        x1 = df['PGA_g'].to_list()[:] #[1:-1]
        y1 = df['ds1norm'].to_list()[:] #[1:-1]
        mu1, sigma1 = optimize.curve_fit(func,x1,y1)[0]
        _,_,r_1,_,_ = stats.linregress(y1,[func(x,mu1,sigma1) for x in x1])
        r2_1 = r_1**2
        
        x2 = df['PGA_g'].to_list()[:] #[1:-1]
        y2 = df['ds2norm'].to_list()[:] #[1:-1]
        mu2, sigma2 = optimize.curve_fit(func,x2,y2)[0]
        _,_,r_2,_,_ = stats.linregress(y2,[func(x,mu2,sigma2) for x in x2])
        r2_2 = r_2**2
        
        x3 = df['PGA_g'].to_list()[:] #[1:-1]
        y3 = df['ds3norm'].to_list()[:] #[1:-1]
        mu3, sigma3 = optimize.curve_fit(func,x3,y3)[0]
        _,_,r_3,_,_ = stats.linregress(y3,[func(x,mu3,sigma3) for x in x3])
        r2_3 = r_3**2


        print()
        print('========================')
        print('Curve type: {:s}'.format(k))
        print('IM: {}'.format('PGA (g)'))
        print('  Loc: {}'.format(locName))
        print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1,sigma1,r2_1))
        print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2,sigma2,r2_2))
        print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3,sigma3,r2_3))    
        
        # PLOTTING
        if toggle_plot:
            x = np.linspace(0,0.7,100)
            
            plt.figure(13)
            ax13 = df.plot(x="PGA_g",y="ds1norm",kind="scatter", color='r', marker='d', s=20, label="ds1")
            ax13 = df.plot(x="PGA_g",y="ds2norm",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax13)
            ax13 = df.plot(x="PGA_g",y="ds3norm",kind="scatter", color='b', marker='o', s=10,  label="ds3",ax=ax13)
            ax13.plot(x,func(x,mu1,sigma1),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_1))
            ax13.plot(x,func(x,mu2,sigma2),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_2))
            ax13.plot(x,func(x,mu3,sigma3),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_3))
            ax13.legend(loc="lower right")
            ax13.grid(visible=True)
            ax13.set_xlim(0.0,0.7)
            ax13.set_ylim(-0.05,1.1)
            ax13.set_xlabel("PGA (g)")
            ax13.set_ylabel("$P(ds>ds_i|PGA)$")
            title = '{:s} CDF, {:s}'.format(k,locName)
            ax13.set_title(title)
            plt.show()
    
    
    except RuntimeError:
        print('Curve {:s}, parameters not found'.format(k))
    
 

        
    
#%% IMcomb Fitting
for k, func in f.items():
    try:
        x1 = dfIM['IMcomb'].to_list()[:] #[1:-1]
        y1 = dfIM['ds1norm'].to_list()[:] #[1:-1]
        mu1, sigma1 = optimize.curve_fit(func,x1,y1)[0]
        _,_,r_1,_,_ = stats.linregress(y1,[func(x,mu1,sigma1) for x in x1])
        r2_1 = r_1**2
        
        x2 = dfIM['IMcomb'].to_list()[:] #[1:-1]
        y2 = dfIM['ds2norm'].to_list()[:] #[1:-1]
        mu2, sigma2 = optimize.curve_fit(func,x2,y2)[0]
        _,_,r_2,_,_ = stats.linregress(y2,[func(x,mu2,sigma2) for x in x2])
        r2_2 = r_2**2
        
        x3 = dfIM['IMcomb'].to_list()[:] #[1:-1]
        y3 = dfIM['ds3norm'].to_list()[:] #[1:-1]
        mu3, sigma3 = optimize.curve_fit(func,x3,y3)[0]
        _,_,r_3,_,_ = stats.linregress(y3,[func(x,mu3,sigma3) for x in x3])
        r2_3 = r_3**2
    
        print()
        print('========================')
        print('Curve type: {:s}'.format(k))
        print('IM: {}'.format('IMcomb'))
        print('  Loc: {}'.format(locName))
        print('    ds1:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu1,sigma1,r2_1))
        print('    ds2:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu2,sigma2,r2_2))
        print('    ds3:: p1: {0:0.3f}, p2: {1:0.3f}, r2: {2:0.3f}'.format(mu3,sigma3,r2_3))    
        
        # PLOTTING
        if toggle_plot:
            x = np.linspace(0,0.7,100)
            
            plt.figure(23)
            ax23 = dfIM.plot(x="IMcomb",y="ds1norm",kind="scatter", color='r', marker='d', s=20, label="ds1")
            ax23 = dfIM.plot(x="IMcomb",y="ds2norm",kind="scatter", color='g', marker='s', s=10, label="ds2",ax=ax23)
            ax23 = dfIM.plot(x="IMcomb",y="ds3norm",kind="scatter", color='b', marker='o', s=10, label="ds3",ax=ax23)
            ax23.plot(x,func(x,mu1,sigma1),'r--',label='ds1_fit, $r^2$={0:0.3f}'.format(r2_1))
            ax23.plot(x,func(x,mu2,sigma2),'g--',label='ds2_fit, $r^2$={0:0.3f}'.format(r2_2))
            ax23.plot(x,func(x,mu3,sigma3),'b--',label='ds3_fit, $r^2$={0:0.3f}'.format(r2_3))
            ax23.legend(loc="lower right")
            ax23.grid(visible=True)
            ax23.set_xlim(0.0,0.7)
            ax23.set_ylim(-0.05,1.1)
            ax23.set_xlabel("IMcomb")
            ax23.set_ylabel("$P(ds>ds_i|PGA)$")
            title = '{:s} CDF, {:s}'.format(k,locName)
            ax23.set_title(title)
            plt.show()     
        
    except RuntimeError:
        print('Curve {:s}, parameters not found'.format(k))
                    
    
   
        
        
    #     # print(df)    

if toggle_plot_old:
    # sns.scatterplot(data=df, x="PGA_g", y="nLEAK")
    plt.figure(1)
    plt.grid()
    ax1 = plt.axis([0,0.5,0,50])
    sns.scatterplot(data=df, x="PGA_g", y="nBC", hue="Loc", style="Loc").set(title = 'No. of burning compartments vs. PGA')
    # ax1b = ax1.twinx()
    # plt.plot(x=[0, 0], y=[1, 1],ax=ax1b,legend=False)
    plt.figure(2)
    plt.grid()
    ax2 = plt.axis([0,0.3,0,50])
    sns.scatterplot(data=df, x="IMcomb", y="nBC", hue="Loc", style="Loc").set(title = 'No. of burning compartments vs. $IM_{comb}$')
    plt.figure(3)
    plt.grid()
    ax3 = plt.axis([0,1500,0,50])
    sns.scatterplot(data=df, x="nLEAK", y="nBC", hue="Loc", style="Loc").set(title = 'No. of burning compartments vs. No. of leaks')
    plt.figure(4)
    # ax3 = plt.axis([0,50,0,50])
    sns.histplot(data=df, x="PGA_g",y="nBC",hue="Loc")
    
    # nLEAK vs. IM
    plt.figure(6)
    plt.grid()
    sns.scatterplot(data=df,x="PGA_g",y="nLEAK",hue="Loc",style="Loc").set(title = 'No. of leaks vs. PGA')
    plt.figure(7)
    plt.grid()
    sns.scatterplot(data=df,x="IMcomb",y="nLEAK",hue="Loc",style="Loc").set(title = 'No. of leaks vs. $IM_{comb}$')
    
    
    plt.figure(8)
    sns.jointplot(data=dfex,x="nLEAK",y="nBC",kind="hex",)
    # df.sort_values(by='IMcomb',inplace=True)
    
# beepy.beep(sound="coin")
    








    