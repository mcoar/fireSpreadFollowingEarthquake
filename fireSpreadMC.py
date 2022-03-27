# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:35:44 2020

@author: Lab User
"""

#%% Import libraries

from calc_ds import *
from compartment import Compartment
import fireSpreadMain as fs
import igraph as ig
import intensity as im
from loadall import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pytictoc import TicToc; t = TicToc()
import random
import sys
from scipy import stats
import sprinkler_hydraulics as spr
from wall import Wall
from window import Window, Door
import wntr


nSamples = 100

# EQ Record path
idx = 27
username = "Lab User"
path = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/9st_mrf_mwc/"

# Grab all directory names in the path
f = []
for (dirpath, dirnames, filenames) in os.walk(path):
    f.extend(dirnames)
    break

# Split the directories into lagrande and seattle
def splitList(x):
    recdict = dict()
    for recname in x:
        rec = recname[0:8]
        if rec in recdict.keys():
            recdict[rec].append(recname)
        else:
            recdict[rec] = [recname]
    return recdict

recdict = splitList(f)
shapeErrorList = []

#%% Some options, initializations, and toggles


# Set up correlation for water network
wnfilein = "wn200ft.pickle"
username = "Lab User"
PIKIN = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/"+wnfilein
# wnfileout = 'wn-{:s}.pickle'.format(pathout)
# PIKOUT = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/PythonFiles-OS/WNpickle/"+wnfileout
# wn, wnresults = spr.Sprinkler(PFA,IDR,wnpathin=PIKIN,wnpathout=PIKOUT,plot=togglePlotSprinkler)
# pressure = wnresults.node['pressure']

f=open(PIKIN,'rb')
wn = pickle.load(f)
f.close()

def sampleWN(wn,nSamples=10000):
    """ 
    Assign correlated random variable samples to pipes in a water network.
    
    ___INPUT___
    
    wn: wntr water network file. 
        All pipes should have an attribute "floor"
    nSample: int
        number of samples
    ___OUTPUT___
    
    wn: wntr water network file 
        Identical to input except that pipes have an attribute "rvs" which
        is a vector of samples of dimension [nSamplesx1]
    
    """
    # test
    import wntr
    from scipy import stats
    
    nnode = len(wn.node_name_list)     # number of nodes
    npipe = len(wn.pipe_name_list)     # number of pipes = nn - 1 (tree-like)
    
    corrF = 0.7     # correlation on the same "F"loor
    corrG = 0.5     # correlation, "G"eneral
    
    meanvec = [0 for col in range(npipe)]
    covmat = np.zeros((npipe,npipe))
       
    i = 0
    j = 0
    
    for ename1 in wn.pipe_name_list:
        j = 0
        e1 = wn.get_link(ename1)
        f1 = e1.floor
        for ename2 in wn.pipe_name_list:
            e2 = wn.get_link(ename2)
            f2 = e2.floor
            if ename1 == ename2:
                covmat[i][j] = 1.
            elif f1 == f2:
                covmat[i][j] = corrF
            else:
                covmat[i][j] = corrG
            j += 1
        i += 1
     
    mvnorm = stats.multivariate_normal(mean = meanvec,cov=covmat)
    
    x = mvnorm.rvs(nSamples)
    
    # Uniformify the marginals
    norm = stats.norm()
    x_unif = norm.cdf(x)
    
    # Assign marginals to pipes for sampling
    i = 0
    for ename in wn.pipe_name_list:
        e = wn.get_link(ename)
        e.rvs = x_unif[:,i]
        i+=1
        
    return wn
    
BCPATH =  r"C:/Users/Lab User/OneDrive/Documents/OpenSees/PythonFiles-OS/BCpickle1/"

nBC_list = []

hardcoded = [
    # 'Output-LaGrande_009NS',
  # 'Output-LaGrande_010EW',
  # 'Output-LaGrande_010NS',
  # 'Output-LaGrande_011EW',
  # 'Output-LaGrande_011NS',
  # 'Output-LaGrande_012EW',
  # 'Output-LaGrande_012NS',
  # 'Output-LaGrande_013EW',
  # 'Output-LaGrande_013NS',
  # 'Output-LaGrande_014EW',
  # 'Output-LaGrande_014NS',
  # 'Output-LaGrande_017EW',
  # 'Output-LaGrande_017NS',
  # 'Output-LaGrande_018EW',
  # 'Output-LaGrande_018NS',
  # 'Output-LaGrande_019EW',
  # 'Output-LaGrande_019NS',
  # 'Output-LaGrande_020EW',
  # 'Output-LaGrande_020NS',
  # 'Output-LaGrande_021EW',
  # 'Output-LaGrande_021NS',
  # 'Output-LaGrande_022EW',
  # 'Output-LaGrande_022NS',
  # 'Output-LaGrande_023EW',
  # 'Output-LaGrande_023NS',
  # 'Output-LaGrande_024EW',
  # 'Output-LaGrande_024NS',
  # 'Output-LaGrande_025EW',
  # 'Output-LaGrande_025NS',
  # 'Output-LaGrande_026EW',
  # 'Output-LaGrande_026NS',
  # 'Output-LaGrande_027EW',
  # 'Output-LaGrande_027NS',
  # 'Output-LaGrande_028EW',
  # 'Output-LaGrande_028NS',
  # 'Output-LaGrande_029EW',
  # 'Output-LaGrande_029NS',
  # 'Output-LaGrande_030EW',
  # 'Output-LaGrande_030NS',
  # 'Output-LaGrande_031EW',
  # 'Output-LaGrande_031NS',
  # 'Output-LaGrande_032EW',
  # 'Output-LaGrande_032NS',
  # 'Output-LaGrande_033EW',
  # 'Output-LaGrande_033NS',  
  # ##########################
  # 'Output-Seattle_008EW',
  # 'Output-Seattle_008NS',
  # 'Output-Seattle_009EW',
  # 'Output-Seattle_009NS',
  # 'Output-Seattle_010EW',
  # 'Output-Seattle_010NS',
  # 'Output-Seattle_011EW',
  # 'Output-Seattle_011NS',
  # 'Output-Seattle_012EW',
  # 'Output-Seattle_012NS',
  # 'Output-Seattle_013EW',
  # 'Output-Seattle_013NS',
  # 'Output-Seattle_014EW',
  # 'Output-Seattle_014NS',
  # 'Output-Seattle_017EW',
  # 'Output-Seattle_017NS',
  # 'Output-Seattle_018EW', 
  # 'Output-Seattle_018NS',
  # 'Output-Seattle_019EW',
  # 'Output-Seattle_019NS',
  # 'Output-Seattle_020EW',
  # 'Output-Seattle_020NS',
  # 'Output-Seattle_021EW',
  # 'Output-Seattle_021NS',
  # 'Output-Seattle_022EW',
  # 'Output-Seattle_022NS',
  # 'Output-Seattle_023EW',
  # 'Output-Seattle_023NS',
  # 'Output-Seattle_024EW',
  # 'Output-Seattle_024NS',
  # 'Output-Seattle_025EW',
  # 'Output-Seattle_025NS',
  # 'Output-Seattle_026EW',
  # 'Output-Seattle_026NS',
  # 'Output-Seattle_027EW',
  # 'Output-Seattle_027NS',
  # 'Output-Seattle_028EW',
  # 'Output-Seattle_028NS',
  # 'Output-Seattle_029EW',
  # 'Output-Seattle_029NS',
  # 'Output-Seattle_030EW',
  # 'Output-Seattle_030NS',
  'Output-Seattle_031EW',
  'Output-Seattle_031NS',
  'Output-Seattle_032EW',
  'Output-Seattle_032NS',
  'Output-Seattle_033EW',
  'Output-Seattle_033NS']

for key, locations in recdict.items():
    # makes sure it only looks at output folders
    if key[0:6] != 'Output':
        continue
    print('Location:: ',key)
    if isinstance(locations,list):
        for record in locations:
            # if True:
            if record not in hardcoded:
                continue
            else:
                print('======================================================')
                print(record)
                pathin = path+record
                wnS = sampleWN(wn,nSamples)
                n = 1
                eps = 0.
                mean_nBC = 0.
                abs_err =100.
                nBC_vec = []
                t.tic()
                while abs_err > 0.01 or n < 30:
                    if n > nSamples:
                        wnS = sampleWN(wn,nSamples)
                    pathout = record[7::]+'_{:00005d}'.format(n)
                    bcOut,wnOut = fs.fireSpread(pathin,pathout,wnin=wnS,ns=n)
                    
                    nBC = len(bcOut)
                    nBC_vec.append(nBC)
                    mean_nBC_old = mean_nBC
                    mean_nBC = mean_nBC_old+(nBC-mean_nBC_old)/n
                    n += 1
                    
                    abs_err = mean_nBC - mean_nBC_old
                    print('Mean number of BC:' + str(mean_nBC))
                    print('======')
                    
                    #pickle
                    f=open(BCPATH+'BC_'+pathout+'.pickle','wb')
                    pickle.dump(bcOut,f)
                    f.close()
                t.toc('MC analysis for '+record)
                nBC_list.append(nBC_vec)
    
        else:
            print(record)