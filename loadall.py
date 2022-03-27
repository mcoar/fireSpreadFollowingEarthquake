# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 08:39:58 2022

@author: Lab User
"""
import pickle

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break