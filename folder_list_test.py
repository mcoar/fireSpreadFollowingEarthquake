# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 09:26:40 2022

@author: Lab User
"""

from os import walk

idx = 27
username = "Lab User"
path = r"C:/Users/"+username+r"/OneDrive/Documents/OpenSees/9st_mrf_mwc/_Output"
city = 'LaGrande'
GM_dir = 'EW'
folder = 'Output-{:s}_{:003d}{:s}'.format(city,idx,GM_dir)

pathname = path+folder

f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(dirnames)
    break

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
        