# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:02:02 2022

Takes a text file of floats and gives max absolute value.

@author: Lab User
"""



def getPGA(PATHIN):
    with open(PATHIN,'r') as f:
        data = [float(val) for val in f]
        PGA = max(data, key=abs)
    return PGA

PATH = r"C:/Users/Lab User/OneDrive/Documents/OpenSees/9st_mrf_mwc/GMs_Full_Suite/"
FILE = "LaGrande_002EW.txt"
PGA = getPGA(PATH+FILE)